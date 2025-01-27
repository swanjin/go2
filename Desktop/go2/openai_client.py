# openai_client.py
import os
import base64
import datetime
import wave
import io
import pyaudio

from openai import OpenAI
import cv2

from ai_client_base import AiClientBase, ResponseMsg
from vision import VisionModel
from navigation import NaviModel, Mapping
# from round import Round
import utils
from round import Round

class OpenaiClient(AiClientBase):
    def __init__(self, env, key):
        # Call the parent class's constructor to initialize system_prompt and other attributes
        super().__init__(env)
        self.env = env
        self.image_counter = 0
        self.msg = []
        self.msg_feedback = []
        self.round_number = 1
        self.chat = []
        self.curr_state = utils.string_to_tuple(self.env['curr_state'])
        self.memory_list = []
        self.is_initial_prompt_landmark_or_non_command = True
        self.is_initial_response_format_non_command = True

        self.client = OpenAI(api_key=key)
        self.vision_model = VisionModel(self.env)
        self.navi_model = NaviModel()
        self.mapping = Mapping()

        try:
            os.makedirs('test', exist_ok=True)
            self.save_dir = f"test/test_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.mkdir(self.save_dir)
            self.log_file = open(f"{self.save_dir}/log.log", "a+") # append: a+ overwrite: w+
        except Exception as e:
            print(f"Failed to create directory: {e}")

    def set_target(self, target):
        self.target = target

    def update_memory_list(self, detected_objects, chat, assistant):
        round = Round(self.round_number, detected_objects, chat, assistant)
        self.memory_list.append(round)

        self.log_file.write(
            f"Round {round.round_number}:\n"
            f"- Detected Objects: {round.detected_objects if round.detected_objects else 'None'}\n"
            f"- Chat: {round.chat if round.chat else 'None'}\n"
            f"- Initial State: {round.assistant.initial_state}\n"
            f"- Action: {round.assistant.action}\n"
            f"- New State: {round.assistant.new_state}\n"
            f"- Reason: {round.assistant.reason if round.assistant.reason else 'None'}\n\n"
        )

        self.log_file.flush()
        self.round_number += 1
        self.curr_state = round.assistant.new_state

    def construct_detection_auto(self, description):
        return (
            f"Detection:\n"
            f"(The image size is {self.env['captured_width']}x{self.env['captured_height']}, "
            f"with the pixel index (0, 0) located at the top-left corner.)\n"
            f"{description} \n\n"
        )

    def construct_detection_feedback(self, detected_objects):
        return (
            f"Detection:\n"
            f"{detected_objects} \n\n"
        )

    def construct_memory(self, memory):
        return (
            f"Memory:\n {memory}\n\n"
        )

    def analyze_image(self, image_pil):
        image_analysis = self.vision_model.describe_image(image_pil)
        return image_analysis.frame, image_analysis.detected_objects, image_analysis.description # array, list, list

    def append_message(self, message, message_role: str, message_content: str):
        message.append({"role": message_role, "content": message_content})

    def get_ai_response(self, message):
        # print(message)
        result = self.client.chat.completions.create(
            model=self.env['ai_model'],
            messages=message
        )
        return result.choices[0].message.content
    
    def string_to_tuple(self, input_string):
        # Remove markdown code block formatting if present
        cleaned_string = input_string.replace('```', '').strip()
        # Remove any newlines
        cleaned_string = cleaned_string.replace('\n', '')
        # Parse the tuple
        return tuple(map(int, cleaned_string.strip("()").split(",")))

    def initialize_prompt_auto(self, description):
        self.msg.clear()
        self.append_message(self.msg, "user", self.prompt_auto(self.curr_state))
        self.append_message(self.msg, "user", self.construct_detection_auto(description))
        self.append_message(self.msg, "user", self.construct_memory(self.memory_list))
        self.append_message(self.msg, "user", self.response_format_auto())
    
    def get_response_by_LLM(self, image_pil, dog_instance):   
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            return None
        
        # Analyze image
        frame_bboxes_array, detected_objects, description = self.analyze_image(image_pil)
        
        # Initialize messages
        self.initialize_prompt_auto(description)
        
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            return None

        rawAssistant = self.get_ai_response(self.msg)
        assistant = ResponseMsg.parse(rawAssistant)

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            return None  

        # Update data
        self.store_image(frame_bboxes_array)
        self.update_memory_list(detected_objects, self.chat, assistant)

        return assistant
    
    def initial_prompt_feedback(self, detected_objects):
        # Initialize messages
        if self.is_initial_prompt_landmark_or_non_command:
            self.append_message(self.msg_feedback, "user", self.prompt_landmark_or_non_command(self.curr_state))
            self.append_message(self.msg_feedback, "user", self.construct_detection_feedback(detected_objects))
            self.append_message(self.msg_feedback, "user", self.construct_memory(self.memory_list))      
            self.is_initial_prompt_landmark_or_non_command = False

    def initial_response_format_non_command(self):  
        if self.is_initial_response_format_non_command:
            self.append_message(self.msg_feedback, "user", self.response_format_non_command())
            self.is_initial_response_format_non_command = False

    def feedback_mode_on(self, image_pil):
        # Analyze image
        frame_bboxes_array, detected_objects, description = self.analyze_image(image_pil)

        # Initialize messages
        self.initial_prompt_feedback(detected_objects)
        
        return frame_bboxes_array, detected_objects

    def get_response_non_command(self, user_input):
        self.initial_response_format_non_command()
        self.append_message(self.msg_feedback, "user", user_input)
        self.append_message(self.chat, "user", user_input)

        rawAssistant = self.get_ai_response(self.msg_feedback)
        self.append_message(self.msg_feedback, "assistant", rawAssistant)
        self.append_message(self.chat, "assistant", rawAssistant)

        return rawAssistant
        
    def get_response_landmark_or_non_command(self, user_input, frame_bboxes_array, detected_objects):
        # Append user input to messages
        self.append_message(self.msg_feedback, "user", user_input)
        self.append_message(self.chat, "user", user_input)

        # Determine if the feedback is a landmark or general
        if self.is_landmark(user_input):
            print("❗ Executing landmark command")
            self.append_message(self.msg_feedback, "user", self.response_format_landmark_command())
            new_state = utils.string_to_tuple(self.get_ai_response(self.msg_feedback))
            action_to_goal = self.navi_model.navigate_to(self.curr_state, new_state, self.mapping.obstacles)
            assistant = ResponseMsg(self.curr_state, new_state, action_to_goal, "")
        else:
            print("❗ Executing general command")
            self.msg_feedback[0]['content'] = self.prompt_general_command(self.curr_state) # replace user prompt
            self.append_message(self.msg_feedback, "user", self.response_format_general_command())
            rawAssistant = self.get_ai_response(self.msg_feedback)
            assistant = ResponseMsg.parse(rawAssistant)

        # Update data
        image_pil_fmode = utils.put_text_top_left(frame_bboxes_array, text="Feedback mode")
        self.store_image(image_pil_fmode)
        self.update_memory_list(detected_objects, self.chat, assistant)
        self.msg_feedback.clear()
        self.chat.clear()
        self.is_initial_prompt_landmark_or_non_command = True
        self.is_initial_response_format_non_command = True

        return assistant

    def stt(self, voice_buffer):
        container = voice_buffer
        transcription = self.client.audio.transcriptions.create(
			model="whisper-1", 
			file=container,
			language='en'
		)
        return transcription.text

    def parse_action_tts(self, action):
        sentence = " then ".join(action)
        return sentence

    def tts(self, text):
        CHUNK = 1024
        if not isinstance(text, list):
            text = text
        else:
            text = self.parse_action_tts(text)

        # Open `/dev/null` and redirect `stderr` to it at the OS level
        devnull = os.open(os.devnull, os.O_WRONLY)
        original_stderr = os.dup(2)  # Save original `stderr` file descriptor
        os.dup2(devnull, 2)          # Redirect `stderr` to `/dev/null`

        try:
            with self.client.with_streaming_response.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input = text,
            response_format= "wav"
            ) as response:
                container = io.BytesIO(response.read())
                with wave.open(container) as wf:
                    p = pyaudio.PyAudio()
                    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)

                    while len(data := wf.readframes(CHUNK)): 
                        stream.write(data)
                    stream.close()
                    p.terminate()

        finally:
            # Restore original `stderr` and close `/dev/null`
            os.dup2(original_stderr, 2)
            os.close(devnull)
            os.close(original_stderr)

    def close(self):
        self.log_file.close()

    def is_instruction_command(self, input): 
        msg = []
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            "Your task: Determine if the user input is an request or command asking you to do something. "
            "If yes, respond with 'true'. If no, respond with 'false'."
        )
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)

        try:
            rawAssistant = self.get_ai_response(msg)
            is_command = rawAssistant.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_instruction_command: {e}")
            return False
        return is_command
    
    def is_landmark(self, input): 
        msg = []
        prompt = "You are Go2, a helpful robot dog assistant who only speaks English. Given the user input, determine if it indicates the user wants you to move by referencing any of the following landmarks: refrigerator, kitchen, TV, desk, cabinet, sofa, banana, bottle, box. If the user input references any of these, respond with 'true'. Otherwise, respond with 'false'."
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)

        try:
            rawAssistant = self.get_ai_response(msg)
            is_landmark = rawAssistant.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_instruction_command: {e}")
            return False
        return is_landmark
