import signal
import sys
import time
import os
import sys
import threading
import queue
from pathlib import Path
import glob

import cv2

robot_interface_path = os.path.join(Path(__file__).parent, 'robot_interface/lib/python/x86_64')
sys.path.append(robot_interface_path)
import robot_interface as sdk

from vision_owl import OWLDepthModel
from ai_controller import AiController
from ai_client_base import AiClientBase, ResponseMessage
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
        self.ai_client = AiController.getClient(env, apikey[env["ai"]])
        self.owl_depth_model = OWLDepthModel(env)
        self.pause_by_trigger = False
        self.image_files = None  # To store image paths when using test_dataset


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
        if self.env["connect_robot"]:
            gstreamer_str = self.env["robot_gstreamer"]  # Robot camera resolution: 1280x720
            self.capture = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
        else:
            self.capture = cv2.VideoCapture(0)  # Built-in webcam

    def read_frame(self):
        """
        Read frame from the camera or load an image from test_dataset.
        Returns:
            - frame (np.ndarray): The image/frame as a NumPy array
        """
        if self.env["use_test_dataset"]:
            # Read images from the test_dataset one by one
            for image_file in self.image_files:
                frame = cv2.imread(image_file)
                if frame is None:
                    print(f"Failed to load image {image_file}.")
                    continue
                yield frame
        else:
            while True:
                success, frame = self.capture.read()
                time.sleep(3)
                if not success:
                    print("Failed to capture frame from camera.")
                    break
                yield frame  

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
        for i, frame in enumerate(self.read_frame()):
            if not self.env["use_test_dataset"] and i >= self.env["max_round"]: # If camera capture is used, it stops after a maximum number of rounds. This limit doesn't apply to test images.
                break

            rawAssistant = self.ai_client.gpt_vision_test(frame)
            print(rawAssistant)

        self.stop_thread1 = True

    def queryGPT_by_LLM(self):
        confused_pause = False
        for i, frame in enumerate(self.read_frame()):
            if not self.env["use_test_dataset"] and i >= self.env["max_round"]: # If camera capture is used, it stops after a maximum number of rounds. This limit doesn't apply to test images.
                break

            # print(f"current action #{i}")
            if not self.feedback_start.empty() or confused_pause: # block till feedback query ends
                confused_pause = False
                self.feedback_start.get() # Since the feedback_start queue is not empty, the get() method removes an item from the queue
                self.feedback_end.get() # Since the feedback_end queue is empty until the feedback query ends, the get() method on this queue blocks the execution of the code. When the feedback query ends, an item is placed into the feedback_end queue via feedback_end.put(1). This action makes the feedback_end queue non-empty, allowing the get() method to remove the item from the queue. Once the item is removed, the blocking ends, and the remaining code can be executed.
            
            # Query GPT
            if not self.feedback_start.empty(): # non-blocking; an example of blocking: input()
                continue
            if not self.env["connect_gpt"]:
                self.ai_client.vision_model_test(frame)
                print("Assumed GPT answered")
            else:
                if self.env["useVLM"]:
                    rawAssistant, assistant = self.ai_client.get_response_by_image(frame)
                else:
                    rawAssistant, assistant = self.ai_client.get_response_by_LLM(frame)

                if not self.feedback_start.empty(): # non-blocking
                    continue
                print(assistant.action)
                print(assistant.reason)

                if assistant.action == "Pause":
                    print("Go2) I'm confused. Please help me.")
                    print("Press Enter to start and finish giving your feedback.")
                    confused_pause = True
                    self.pause_by_trigger = True
                    continue

                if not self.feedback_start.empty(): # non-blocking
                    continue
                self.activate_sportclient(assistant.action)
                if self.env["tts"]:
                    self.ai_client.tts(assistant.action)

        self.stop_thread1 = True

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
                rawAssistant, assistant = self.ai_client.get_response_by_feedback(feedback, self.pause_by_trigger)
                print(rawAssistant)
                self.activate_sportclient(assistant.action)
                if self.env["tts"]:
                    self.ai_client.tts(assistant.reason)
                self.pause_by_trigger = False
                self.feedback_end.put(1)

    def activate_sportclient(self, ans):
        # if len(sys.argv) < 2:
        #     print(f"Usage: {sys.argv[0]} networkInterface")
        #     return -1
        
        # Assuming that the ChannelFactory and Init methods are properly bound in Python
        # sdk.ChannelFactory.Instance().Init(0, sys.argv[1])
        
        if not self.env["connect_robot"]:
            print("Assumed action executed: " + ans)
            time.sleep(5)
            return 0
        else: 
            chan = sdk.ChannelFactory.Instance()
            chan.Init(0, "enp58s0")

            # Initialize the sport client, assuming the translated classes have the same functionality
            sport_client = sdk.SportClient(False)
            sport_client.SetTimeout(10.0)
            sport_client.Init()
            
            ans = ans.strip().lower()
            if ans == 'move forward':
                sport_client.Move(5, 0, 0) 
            elif ans == 'move backward':
                sport_client.Move(-2.5, 0, 0) 
            elif ans == 'shift right':
                sport_client.Move(0, -1, 0) 
            elif ans == 'shift left':
                sport_client.Move(0, 1, 0) 
            elif ans == 'turn right':
                sport_client.Move(0, 0, -2) # rotate right 0.5 sec: 1.57 radian = 90 degree/sec w/ horizontal angle of view of 100 degrees
                # time.sleep(1)
                # sport_client.Move(0, 0, 3)
                # sport_client.Move(0, -0.02, -0.09)
            elif ans == 'turn left':
                sport_client.Move(0, 0, 2) # rotate left
            elif ans == 'stop':
                sport_client.StopMove()
            elif ans == 'pause':
                sport_client.StopMove()
            else:
                print("Action not recognized: " + ans)
                # sport_client.StopMove()  # stop 

            #time.sleep(2.5)
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