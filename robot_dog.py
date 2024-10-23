# robot_dog.py
import signal
import sys
import time
import os
import base64
import threading
import queue
from pathlib import Path
import glob

from PIL import Image
import cv2
robot_interface_path = os.path.join(Path(__file__).parent, 'robot_interface/lib/python/x86_64')
sys.path.append(robot_interface_path)
import robot_interface as sdk

from vision import VisionModel
from ai_selector import AiSelector
from ai_client_base import AiClientBase, ResponseMessage
import utils
from recorder import SpeechByEnter

class Dog:
    def __init__(self, env, apikey):
        signal.signal(signal.SIGINT, self.signal_handler)

        self.env = env
        self.feedback_start = queue.Queue(maxsize=1)
        self.feedback_end = queue.Queue(maxsize=1)
        self.stop_thread1 = False
        self.stop_thread2 = False
        self.capture: cv2.VideoCapture
        self.ai_client = AiSelector.getClient(env, apikey[env["ai"]])
        self.vision_model = VisionModel(env)
        self.pause_by_trigger = False
        self.image_files = None  # To store image paths when using test_dataset

        # Initialize the communication channel and the sport client
        if self.env["connect_robot"]:
            try:
                chan = sdk.ChannelFactory.Instance()
                chan.Init(0, self.env["network_interface"])
            except Exception as e:
                print(f"Error: Failed to initialize the connection with the robot. Please check the network interface name and ensure the robot is connected.")
                print(f"Details: {e}")
                return -1
        
            self.sport_client = sdk.SportClient(False) # True enables lease management to ensure exclusive robot control by one client
            self.sport_client.SetTimeout(60.0)
            self.sport_client.Init()

    def signal_handler(self, sig, frame):
        print("SIGINT received, stopping threads and shutting down...")
        self.shutdown(True)  # Ensure proper shutdown is called
        sys.exit(0)  # Gracefully exit the program

    def setup(self):
        if self.env["use_test_dataset"]:
            target = self.env["target_in_test_dataset"]
            # # Dynamically find all subfolders inside the test_dataset folder
            # test_dataset_folder = os.path.join(Path(__file__).parent, 'test_dataset')
            # subfolders = [f.name for f in os.scandir(test_dataset_folder) if f.is_dir()]
            
            # if not subfolders:
            #     raise FileNotFoundError("No subfolders found in the test_dataset folder.")
            
            # # Iterate through all subfolders and process each one
            # for i, subfolder in enumerate(subfolders):
            #     print(f"Go2) Processing subfolder '{subfolder}' (folder {i+1}/{len(subfolders)}).")
            #     target = subfolder  # Set the current subfolder as the target
            #     print(f"Go2) Using subfolder '{target}' as target.")

            #     # Load images from the current subfolder inside test_dataset
            #     image_folder = os.path.join(test_dataset_folder, subfolder)
            #     self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
                
            #     if not self.image_files:
            #         raise FileNotFoundError(f"No images found in subfolder '{subfolder}'")
                
            #     print(f"Loaded {len(self.image_files)} images from subfolder '{subfolder}'.")

            #     # Here you would typically call the method to set up the input source and pass it the target
            #     self.setup_input_source()  # Setup the input source (camera or images)
            #     self.ai_client.set_target(target)  # Set the target for AI client (subfolder name)
            
        else:
            print("Go2) Hello! I am Go2, the robot dog. What would you like me to search for?")
            if self.env["speechable"]:
                self.speech = SpeechByEnter()
                self.speech.setup()
                container = self.speech.recording()
                target = self.ai_client.stt(container)
                print(f"User) {target}")
            else:
                target = input("User) ") # a water bottle, a TV remote controller, a smartphone
            print("Go2) Understood! Initiating search now.")

        self.setup_input_source(target)
        self.ai_client.set_target(target)

    def setup_input_source(self, target):
        """
        Setup the input source, either capture from camera or use images from test_dataset.
        """
        print("- Setting up input source")
        if self.env["use_test_dataset"]:
            # Load images from test_dataset folder
            image_folder = os.path.join(Path(__file__).parent, f'test_dataset/{target}')
            self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
            if not self.image_files:
                raise FileNotFoundError(f"No images found in folder {image_folder}")
            print(f"Loaded {len(self.image_files)} images from test_dataset.")
        else:
            # Connect to camera
            self.connect_camera()

    def connect_camera(self):
        """
        Connect to the robot camera or use built-in camera.
        """
        print("- Connecting to camera")

        # Ensure any previously opened capture is released to free the resource
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()  # Release the camera resource if it was previously used

        if self.env["connect_robot"]:
            gstreamer_str = self.env["robot_gstreamer"]  # Robot camera resolution: 1280x720
            self.capture = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)        
        else:
            self.capture = cv2.VideoCapture(0)  # Built-in webcam

            if not self.capture.isOpened():
                print("Error: Failed to open the built-in camera. Possible causes:")
                print("1) The built-in camera may be in use by another application. Close any other applications using the camera and try again.")
                print("2) The camera resource might not have been properly released from a previous session.")

    def read_frame(self):
        """
        Read a frame from the camera and return it as a PIL image in RGB format.
        Returns:
            - frame: PIL image in RGB format, or None if an error occurs.
        """
        success, capture = self.capture.read()  # Capture a frame from the camera
        if not success:
            if self.env["connect_robot"]:
                print("Failed to retrieve frame from robot camera. Possible causes:")
                print("1) OpenCV without GStreamer support may have been prioritized.")
                print("2) Network connection to the robot camera may have failed. Please check your LAN cable and try again.")
            return None 
        else:
            image_cv = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = Image.fromarray(image_cv)
            return frame  # Return the captured frame as a PIL image

    def shutdown(self, force=False):
        print("Shutting down the robot dog...")
        self.stop_thread1 = True
        self.stop_thread2 = True
        
        # Wait for thread termination (wait for threads that were signaled to stop by the flag)
        if not force:
            if hasattr(self, 'thread1') and self.thread1.is_alive():
                self.thread1.join()
            if hasattr(self, 'thread2') and self.thread2.is_alive():
                self.thread2.join()
        
        # Release resources
        # Close camera if capture exists
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        self.ai_client.close()

        print("All threads have been terminated and resources have been released.")

    def queryGPT_for_vision_test(self):
        for i in range(self.env["max_round"]):
            if not self.env["use_test_dataset"] and i >= self.env["max_round"]: # If camera capture is used, it stops after a maximum number of rounds. This limit doesn't apply to test images.
                break
            frame = self.read_frame()

            print(f"Round #{i+1}")
            assistant = self.ai_client.gpt_vision_test(frame)
            action_parsed = assistant.action.strip('* ').lower()
            print(action_parsed)
            print(assistant.reason)

        self.stop_thread1 = True

    def queryGPT_by_LLM(self):
        confused_pause = False
        if self.env["use_test_dataset"]:
            for i, image_file in enumerate(self.image_files):  # Iterate over the loaded image files
                print(f"Round #{i+1}")
                image_cv = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(image_cv)
                self.process_frame(frame, confused_pause)  # Process each frame
        else:
            for i in range(self.env["max_round"]):
                frame = self.read_frame()  # Read from the camera
                print(f"Round #{i+1}")
                self.process_frame(frame, confused_pause)  # Process each frame

        self.stop_thread1 = True

    def process_frame(self, frame, confused_pause):
        if not self.feedback_start.empty() or confused_pause: # block till feedback query ends
            confused_pause = False
            self.feedback_start.get() # Since the feedback_start queue is not empty, the get() method removes an item from the queue
            self.feedback_end.get() # Since the feedback_end queue is empty until the feedback query ends, the get() method on this queue blocks the execution of the code. When the feedback query ends, an item is placed into the feedback_end queue via feedback_end.put(1). This action makes the feedback_end queue non-empty, allowing the get() method to remove the item from the queue. Once the item is removed, the blocking ends, and the remaining code can be executed.

        # Query GPT
        if not self.feedback_start.empty(): # non-blocking; an example of blocking: input()
            return

        if not self.env["connect_gpt"]:
            self.ai_client.vision_model_test(frame)
            print("Assumed GPT answered")
        else:
            if self.env["useVLM"]:
                assistant = self.ai_client.get_response_by_image(frame)
            else:
                assistant = self.ai_client.get_response_by_LLM(frame)

            if assistant.action == "Pause":
                print("Go2) I'm confused. Please help me.")
                print("Press Enter to start and finish giving your feedback.")
                confused_pause = True
                self.pause_by_trigger = True
                return  # Exit the frame processing and wait for feedback

            if not self.feedback_start.empty():
                return
            action_parsed = assistant.action.strip('* ').lower()
            print(action_parsed)
            print(assistant.reason)
            if self.env["tts"]:
                self.ai_client.tts(action_parsed)
            
            self.activate_sportclient(action_parsed)

    def user_input(self):
        if self.env["speechable"]:
            container = self.speech.recording(lambda: self.feedback_start.put(1))
            feedback = self.ai_client.stt(container)
            print(f"User) {feedback}")
        else:
            input()
            self.feedback_start.put(1) # stop queryGPT_by_image thread
            feedback = input("User) ")
        return feedback

    def queryGPT_with_feedback(self):
        # print("queryGPT_with_feedback start")
        while not self.stop_thread2:
            feedback = self.user_input()
            if self.stop_thread2:
                break

            if not self.env["connect_gpt"]:
                print("Assumed GPT answered")
            else:
                assistant = self.ai_client.get_response_by_feedback(feedback)
                action_parsed = assistant.action.strip('* ').lower()
                print(action_parsed)
                print(assistant.reason)
                self.activate_sportclient(assistant.action)
                if self.env["tts"]:
                    self.ai_client.tts(assistant.reason)
                self.pause_by_trigger = False
                self.feedback_end.put(1)

    def VelocityMove(self, vx, vy, vyaw, elapsed_time = 1, dt = 0.01):
        for i in range(int(elapsed_time / dt)):
            self.sport_client.Move(vx, vy, vyaw)
            time.sleep(dt)
        for i in range(int(elapsed_time / dt)):
            self.sport_client.StopMove()
            time.sleep(dt)

    def activate_sportclient(self, ans):
        if not self.env["connect_robot"]:
            print("Assumed action executed: " + ans)
            return 0
        else: 
            if ans == 'move forward':
                self.VelocityMove(1, 0, 0) 
            elif ans == 'move backward':
                self.VelocityMove(-1, 0, 0) 
            elif ans == 'shift right':
                self.VelocityMove(0, -0.5, 0) 
            elif ans == 'shift left':
                self.VelocityMove(0, 0.5, 0) 
            elif ans == 'turn right':
                self.VelocityMove(0, 0, -1.04) # 1.04 radian = 60 degree/sec w/ horizontal angle of view of 100 degrees
            elif ans == 'turn left':
                self.VelocityMove(0, 0, 1.04)
            elif ans == 'stop':
                self.sport_client.StopMove()
            elif ans == 'pause':
                self.sport_client.StopMove()
            else:
                print("Action not recognized: " + ans)

            return 0

    def run_gpt(self):
        if self.env["gpt_vision_test"]:
            self.thread1 = threading.Thread(target=self.queryGPT_for_vision_test, daemon=True)
        else:
            self.thread1 = threading.Thread(target=self.queryGPT_by_LLM, daemon=True)
        self.thread1.start()
        self.thread2 = threading.Thread(target=self.queryGPT_with_feedback, daemon=True)
        self.thread2.start()

        while True:
            time.sleep(0.1)
            if self.stop_thread1 and self.stop_thread2:
                print("All threads have ended.")
                break