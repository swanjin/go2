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
from ai_client_base import AiClientBase, ResponseMsg
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
        self.ai_client.dog = self  # Give the client a reference to the dog instance
        self.vision_model = VisionModel(env)
        self.image_files = None  # To store image paths when using test_dataset
        # self.feedback = None
        # self.window = None  # Will be set by the UI
        self.tts_engine = None

        # Initialize the communication channel and the sport client
        if self.env["connect_robot"]:
            try:
                chan = sdk.ChannelFactory.Instance()
                chan.Init(0, self.env["network_interface"])
            except Exception as e:
                print(f"Error: Failed to connect to the robot. Ensure the network interface is correct and the robot is properly connected.")
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
            self.setup_input_source(self.env["target_in_test_dataset"])
        else:
            # Just setup the camera without setting target
            self.setup_input_source(None)
            self.target = None  # Initialize target as None

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

        # Ensure any previously opened capture is released
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()

        if self.env["connect_robot"]:
            try:
                # Modified GStreamer pipeline with error checking
                gstreamer_str = self.env["robot_gstreamer"]
                self.capture = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
                
                if not self.capture.isOpened():
                    raise RuntimeError("Failed to open GStreamer pipeline")
                    
                # Test frame capture
                ret, _ = self.capture.read()
                if not ret:
                    raise RuntimeError("Failed to read frame from GStreamer pipeline")
                    
            except Exception as e:
                print(f"Error connecting to robot camera: {e}")
                print("Falling back to default camera...")
                self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print("Error: Failed to open any camera. Possible causes:")
            print("1) The camera may be in use by another application")
            print("2) The camera resource might not have been properly released")
            print("3) No compatible camera found")
            return False
        
        return True

    def read_frame(self):
        if not hasattr(self, 'capture') or self.capture is None:
            print("Error: Camera not initialized")
            return None

        try:
            if self.env["use_test_dataset"]:
                # Use test dataset logic here
                return None
            
            max_retries = 3
            for attempt in range(max_retries):
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
                print(f"Frame capture failed, attempt {attempt + 1}/{max_retries}")
                time.sleep(0.1)  # Short delay between retries
            
            print("Failed to capture frame after multiple attempts")
            return None
        
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

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

    def check_feedback_and_interruption(self):
        """Waits if feedback is in progress and checks for interruption.
        Returns True if an interruption is flagged, False otherwise."""
        self.feedback_complete_event.wait()  # Block until feedback_complete_event is set
        
        if self.interrupt_round_flag.is_set():
            self.interrupt_round_flag.clear()  # Reset the flag for future tasks
            return True  # Indicates that the current task should be skipped
        return False  # No interruption, proceed with the current task   

    def format_actions(self, actions):
        if isinstance(actions, list):
            return 'and then '.join(map(str, actions))
        return str(actions)

    def queryGPT_by_LLM(self):
        if self.env["woz"]:
            self.env["max_round"] = 1
        while self.ai_client.round_number <= self.env["max_round"]:
            self.feedback_complete_event.wait()

            frame = self.read_frame()
            print(f"Starting round #{self.ai_client.round_number}")

            if self.check_feedback_and_interruption():
                continue

            if not self.env["connect_ai"]:
                print("Assumed GPT answered")
            else:
                assistant = self.ai_client.get_response_by_LLM(frame, dog_instance=self)

                if assistant is None:
                    continue
        
                if self.check_feedback_and_interruption():
                    continue
                
                formatted_action = self.format_actions(assistant.action)
                print(f"formatted_action: {formatted_action}")
                combined_message = f"I'm going to {formatted_action}. {assistant.reason}."
                if self.env["interactive"] or self.env["vn"]:
                    pass
                
                self.activate_sportclient(assistant.action)

                if formatted_action == 'stop':
                    end_message = "I found the apple, so I'm stopping here. You can now end the chat."

            print(f"Round {self.ai_client.round_number} completed.\n")
        
        print("All rounds completed. Press Enter to end session.")
        self.session_active_event.clear()  # Indicate that the session is now inactive

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
            if feedback_input == "feedback":
                print("Feedback command received; checking session_active_event...")  # Debug: Confirm input matched
                if self.session_active_event.is_set():
                    self.feedback_complete_event.clear()  # Pause queryGPT_by_LLM while feedback is in progress
                    self.interrupt_round_flag.set()  # Set the flag to skip the current round
                    print("Giving feedback... (Press Enter when done)")
                    
                    # frame = self.read_frame()
                    # assistant = self.ai_client.get_response_non_command(frame)
                    # if assistant is not None:
                    #     print(f"Printing {assistant.action} from robot_dog.")
                    #     self.activate_sportclient(assistant.action)
                    #     # if self.env["tts"]:
                    #     #     self.ai_client.tts(assistant.action)
                            
                    self.feedback_complete_event.set()  # Allow round_sequence to continue after feedback
                    print("Feedback complete. Moving to next round...")

    def VelocityMove(self, vx, vy, vyaw, elapsed_time = 1, dt = 0.01):
        for i in range(int(elapsed_time / dt)):
            self.sport_client.Move(vx, vy, vyaw)
            time.sleep(dt)
        if self.env["woz"]:
            elapsed_time = 5 # intentional delay for woz
        for i in range(int(elapsed_time / dt)):
            self.sport_client.StopMove()
            time.sleep(dt)

    def activate_sportclient(self, actions):
        if not self.env["connect_robot"]:
            print("Assumed action executed.")
        else:      
            if self.env["woz"]:
                print("Executing WOZ movement sequence:")
                print("1. Move forward sequence")
                self.VelocityMove(0.5, 0, 0)
                self.VelocityMove(0.5, 0, 0)
                print("2. Turn left")
                self.VelocityMove(0, 0, 1.65)
                # print("2. Turn right sequence") # extended version of woz
                # self.VelocityMove(0, 0, -1.65)
                # self.VelocityMove(0, 0, -1.65)
                # self.VelocityMove(0, 0, -1.65)
                print("3. Move forward sequence")
                self.VelocityMove(0.5, 0, 0)
                self.VelocityMove(0.5, 0, 0)
                self.VelocityMove(0.5, 0, 0)
                self.VelocityMove(0.5, 0, 0)
                self.VelocityMove(0.5, 0, 0)
                print("4. Turn left")
                self.VelocityMove(0, 0, 1.65)
                # print("4. Turn right sequence") # extended version of woz
                # self.VelocityMove(0, 0, -1.65)
                # self.VelocityMove(0, 0, -1.65)
                # self.VelocityMove(0, 0, -1.65)
                print("5. Move forward sequence")
                self.VelocityMove(0.5, 0, 0)
                self.VelocityMove(0.5, 0, 0)
                print("6. Final stop")
                self.VelocityMove(0, 0, 0)
                
                # stop_message = "Stop. I found an apple."
                # if self.env["tts"]:
                #     self.ai_client.tts(stop_message) 
            else:                
                if actions == ['stop']:
                    self.sport_client.StopMove()
                else:
                    action_map = {
                        'move forward': (0.5, 0, 0),
                        'move backward': (-0.5, 0, 0),
                        'shift right': (0, -0.35, 0),
                        'shift left': (0, 0.35, 0),
                        'turn right': (0, 0, -1.65),
                        'turn left': (0, 0, 1.65)
                    }
                    
                    for action in actions:
                        if action in action_map:
                            velocity = action_map[action]
                            self.VelocityMove(*velocity)
                        else:
                            print("Action not recognized: " + action)

    def run_gpt(self):
        self.robot_auto_thread = threading.Thread(target=self.queryGPT_by_LLM)
        self.feedback_thread = threading.Thread(target=self.queryGPT_with_feedback)

        self.robot_auto_thread.start()
        self.feedback_thread.start()

    def tts(self, text):
        if not self.env.get('tts', True):
            return
        
        try:
            import pyttsx3
            
            # Convert text to string if it's a list
            if isinstance(text, list):
                text = self.parse_action_tts(text)
            
            # Improve pronunciation if you have such a method
            if hasattr(self, 'improve_pronunciation'):
                text = self.improve_pronunciation(text)
            
            print(f"[TTSWorker] Starting TTS with text: {text}")
            
            # Create a new engine instance each time to avoid reference issues
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
                
            self.tts_engine = pyttsx3.init()
            
            # Configure the engine
            rate = int(200 * self.env.get('tts_speed', 0.8))
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', 1.0)
            
            # Set voice to English if available
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "english" in voice.name.lower() or "en-" in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
                
            # Use the engine to say the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            print("[TTSWorker] TTS finished")
            
            # Signal that TTS is finished if using events
            if hasattr(self, 'tts_finished_event'):
                self.tts_finished_event.set()
            
        except Exception as e:
            print(f"[TTSWorker] Error in TTS: {e}")