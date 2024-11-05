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
    
        self.session_active_event = threading.Event()
        self.session_active_event.set()
        self.feedback_complete_event = threading.Event()
        self.feedback_complete_event.set()
        self.interrupt_round_flag = threading.Event()
        
        self.capture: cv2.VideoCapture
        self.ai_client = AiSelector.getClient(env, apikey[env["ai"]])
        self.vision_model = VisionModel(env)
        self.image_files = None  # To store image paths when using test_dataset
        self.feedback = None
        self.round_number = 1

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
            self.sport_client.SetTimeout(600.0)
            self.sport_client.Init()

    def signal_handler(self, sig, frame):
        print("SIGINT received, stopping threads and shutting down...")
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        self.ai_client.close()
        print("All resources released.")
        print("Program exited.")
        
        # self.shutdown(True)  # Ensure proper shutdown is called
        sys.exit(0)  # Gracefully exit the program

    def setup(self):
        if self.env["use_test_dataset"]:
            target = self.env["target_in_test_dataset"]
        else:
            print("Go2) Hello! I am Go2, the robot dog. What would you like me to search for?")
            if self.env["speechable"]:
                self.speech = SpeechByEnter()
                self.speech.setup()
                container = self.speech.recording()
                target = self.ai_client.stt(container)
                print(f"User) {target}")
            else:
                target = input("User) ")
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
            self.connect_camera()

    def connect_camera(self):
        print("- Connecting to camera")

        # Ensure any previously opened capture is released to free the resource
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()  # Release the camera resource if it was previously used

        if self.env["connect_robot"]:
            gstreamer_str = self.env["robot_gstreamer"]  # Robot camera resolution: 1280x720
            self.capture = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)        
        else:
            self.capture = cv2.VideoCapture(0)

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
        success, capture = self.capture.read()
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
        self.robot_auto_thread.join()
        self.feedback_thread.join()
        print("All threads completed.")

        # Release resources
        # Close camera if capture exists
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        self.ai_client.close()
        print("All resources released.")
        print("Program exited.")

    def queryGPT_for_vision_test(self):
        for i in range(self.env["max_round"]):
            if not self.env["use_test_dataset"] and i >= self.env["max_round"]: # If camera capture is used, it stops after a maximum number of rounds. This limit doesn't apply to test images.
                break
            frame = self.read_frame()

            print(f"Round #{i+1}")
            assistant = self.ai_client.gpt_vision_test(frame)
            print(assistant.action)
            print(assistant.reason)

        self.stop_thread1 = True

    def check_feedback_and_interruption(self):
        """Waits if feedback is in progress and checks for interruption.
        Returns True if an interruption is flagged, False otherwise."""
        self.feedback_complete_event.wait()  # Block until feedback_complete_event is set
        
        if self.interrupt_round_flag.is_set():
            self.interrupt_round_flag.clear()  # Reset the flag for future tasks
            return True  # Indicates that the current task should be skipped
        return False  # No interruption, proceed with the current task   

    def queryGPT_by_LLM(self):
        if self.env["use_test_dataset"]:
            for i, image_file in enumerate(self.image_files):
                print(f"Round #{i+1}")
                image_cv = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(image_cv)
                self.process_frame(frame)
        else:
            while self.round_number <= self.env["max_round"]:
                self.feedback_complete_event.wait()

                frame = self.read_frame()
                print(f"Starting round #{self.round_number}")

                if (self.round_number) % self.env["feedback_interval"] == 0:
                    print("Go2) Would you give any feedback? [Y/n]")
                    if input("User) ").strip().lower() == "y":
                        print("Go2) Thanks for your help! Please provide your feedback.")
                        self.feedback = input("User) ")

                if self.check_feedback_and_interruption():
                    self.round_number += 1  # Increment round number before continuing
                    continue  # Skip the rest of the current iteration and proceed to the next one.

                if not self.env["connect_gpt"]:
                    self.ai_client.vision_model_test(frame)
                    print("Assumed GPT answered")
                else:
                    if self.env["useVLM"]:
                        assistant = self.ai_client.get_response_by_image(frame)
                    else:
                        assistant = self.ai_client.get_response_by_LLM(frame, dog_instance=self)

                        # If get_response_by_LLM returned None, skip the rest of the loop
                        if assistant is None:
                            continue  # Skip to the next round if interrupted
            
                    if self.check_feedback_and_interruption():
                        self.round_number += 1
                        continue

                    print(assistant.action)
                    # print(assistant.reason)
                    if self.env["tts"]:
                        self.ai_client.tts(assistant.action)
                    
                    self.activate_sportclient(assistant.action, int(assistant.move), int(assistant.shift), int(assistant.turn))
                self.feedback = None
                print(f"Round {self.round_number} completed\n")
                self.round_number += 1
            
            print("All rounds completed. Press Enter to end session.")
            self.session_active_event.clear()  # Indicate that the session is now inactive

    def process_frame(self, frame, feedback):
        if not self.env["connect_gpt"]:
            self.ai_client.vision_model_test(frame)
            print("Assumed GPT answered")
        else:
            if self.env["useVLM"]:
                assistant = self.ai_client.get_response_by_image(frame)
            else:
                assistant = self.ai_client.get_response_by_LLM(frame, feedback)
 
            self.feedback_complete_event.wait()  # Blocks until feedback_complete_event is set

            print(assistant.action)
            print(assistant.reason)
            if self.env["tts"]:
                self.ai_client.tts(assistant.action)
            
            self.activate_sportclient(assistant.action, int(assistant.move), int(assistant.shift), int(assistant.turn))

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
        while self.session_active_event.is_set():  # Only active while session is running
            feedback_input = input("Type 'feedback' to give feedback: \n").strip().lower()
            if feedback_input == 'feedback':
                print("Feedback command received; checking session_active_event...")  # Debug: Confirm input matched
                if self.session_active_event.is_set():
                    self.feedback_complete_event.clear()  # Pause queryGPT_by_LLM while feedback is in progress
                    self.interrupt_round_flag.set()  # Set the flag to skip the current round
                    print("Giving feedback... (Press Enter when done)")
                    feedback = input("User) ") # Wait for feedback completion
                    
                    assistant = self.ai_client.get_response_by_feedback(feedback)
                    print(assistant.action)
                    print(assistant.reason)
                    self.activate_sportclient(assistant.action, int(assistant.move), int(assistant.shift), int(assistant.turn))
                    if self.env["tts"]:
                        self.ai_client.tts(assistant.action)
                            
                    self.feedback_complete_event.set()  # Allow round_sequence to continue after feedback
                    print("Feedback complete. Moving to next round...")

    def VelocityMove(self, vx, vy, vyaw, elapsed_time = 1, dt = 0.01):
        for i in range(int(elapsed_time / dt)):
            self.sport_client.Move(vx, vy, vyaw)
            time.sleep(dt)
        for i in range(int(elapsed_time / dt)):
            self.sport_client.StopMove()
            time.sleep(dt)

    def activate_sportclient(self, action, move, shift, turn):
        if not self.env["connect_robot"]:
            print("Assumed action executed.")
            return 0
        else: 
            if move + shift + turn == 0:
                self.sport_client.StopMove()
            else:
                action_map = {
                    'move forward': (0.5, 0, 0),
                    'move backward': (-0.5, 0, 0),
                    'shift right': (0, -0.5, 0),
                    'shift left': (0, 0.5, 0),
                    'turn right': (0, 0, -1.04),
                    'turn left': (0, 0, 1.04)
                }
                
                for ans in action:
                    if ans in action_map:
                        velocity = action_map[ans]
                        for _ in range(move if 'move' in ans else shift if 'shift' in ans else turn):
                            self.VelocityMove(*velocity)
                    else:
                        print("Action not recognized: " + ans)
        return 0

    def run_gpt(self):
        # self.queryGPT_by_LLM()
        
        if self.env["gpt_vision_test"]:
            self.robot_auto_thread = threading.Thread(target=self.queryGPT_for_vision_test)
        else:
            self.robot_auto_thread = threading.Thread(target=self.queryGPT_by_LLM)
        self.feedback_thread = threading.Thread(target=self.queryGPT_with_feedback)

        self.robot_auto_thread.start()
        self.feedback_thread.start()
