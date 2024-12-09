# openai_client.py
import os
import base64
import datetime
import wave
import io
import pyaudio

from openai import OpenAI
import cv2

from ai_client_base import AiClientBase, ResponseMessage
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
        self.history = "None."
        self.msg = []
        self.msg_feedback = []
        self.round_list = []
        self.curr_state = (0,0,0)
        self.client = OpenAI(api_key=key)
        self.vision_model = VisionModel(self.env)
        self.navi_model = NaviModel(self.curr_state)
        self.mapping = Mapping()

        try:
            os.makedirs('test', exist_ok=True)
            self.save_dir = f"test/test_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.mkdir(self.save_dir)
            self.log_file = open(f"{self.save_dir}/history.log", "a+") # append: a+ overwrite: w+
        except Exception as e:
            print(f"Failed to create directory: {e}")

    def set_target(self, target):
        self.target = target

    def update_log(self, assistant, feedback, image_description_str):
        self.log_file.write(f"\n=== image{len(self.round_list)+1} ===\n")
        
        if image_description_str == "No objects related to the target is detected.":
            self.log_file.write(f"Image Analysis: \n None. \n")
        else: 
            self.log_file.write(f"Image Analysis: \n{image_description_str} \n")

        if feedback:
            self.log_file.write(f"Feedback: \n {feedback} \n")

        if assistant is None:
            self.log_file.write(f"Response: \n None.")  # This prevents logging False
        else:
            self.log_file.write(
                f"Response: \n"
                f"Action) {assistant.action} \n"
                f"Move) {assistant.move} \n"
                f"Shift) {assistant.shift} \n"
                f"Turn) {assistant.turn} \n"
                f"New state) {assistant.new_state} \n"
                f"Reason) {assistant.reason} \n"
            )

        self.log_file.flush()

    def update_history_prompt(self, assistant, feedback, image_description_str):
        self.round_list.append(Round(len(self.round_list) + 1, assistant, feedback))
        round_number = len(self.round_list)
        
        self.history = self.history if round_number > 1 else "None."   
        self.history += (
            f"- Round {round_number}: "
            f"Conversation between Go2: {feedback}. "
            f"From the state {assistant.curr_state}, {image_description_str} "
            f"The likelihood of target presence at this state was {assistant.likelihood}. "
            f"You executed the '{assistant.action}' action which led to the updated state of {assistant.new_state}. \n"
        )
        # self.history += "None."

    def construct_image_analysis(self, image_description_str):
        return (
            f"### Image analysis:\n"
            f"(The image size is {self.env['captured_width']}x{self.env['captured_height']}, "
            f"with the pixel index (0, 0) located at the top-left corner.):\n"
            f"{image_description_str} \n\n"
        )

    def construct_history(self, history):
        return (
            f"### History:\n {history}\n\n"
        )

    def analyze_image(self, image_pil):
        image_analysis = self.vision_model.describe_image(image_pil)
        if image_analysis.description == "":
            image_analysis.description = "No objects related to the target is detected."
        return image_analysis.frame, image_analysis.description # array, string

    def append_message(self, message, message_role: str, message_content: str):
        message.append({"role": message_role, "content": message_content})

    def get_ai_response(self, message):
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

    def get_response_by_LLM(self, image_pil, dog_instance, feedback = None):   
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None
        
        if feedback is None:
            feedback = "None"

        # Analyze image
        image_bboxes_array, image_description_str = self.analyze_image(image_pil)
        
        # Initialize messages
        self.msg = []
        self.append_message(self.msg, "user", self.user_prompt_auto(self.curr_state))
        self.append_message(self.msg, "user", self.construct_image_analysis(image_description_str))
        self.append_message(self.msg, "user", self.construct_history(self.history))
        self.append_message(self.msg, "user", self.response_format_auto())
            
        if self.env["print_history"]:
            print(self.history)
        
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None

        rawAssistant = self.get_ai_response(self.msg)
        assistant = ResponseMessage.parse(rawAssistant)

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None  

        # update data and reset messages
        self.curr_state = assistant.new_state
        self.store_image(image_bboxes_array)
        self.update_log(assistant, feedback, image_description_str)
        self.update_history_prompt(assistant, feedback, image_description_str)
        self.msg = []

        return assistant

    def feedback_mode_on(self, image_pil):
        # Analyze image
        image_bboxes_array, image_description_str = self.analyze_image(image_pil)

        # Initialize messages
        self.append_message(self.msg_feedback, "user", self.user_prompt_feedback(self.curr_state))
        self.append_message(self.msg_feedback, "user", self.construct_image_analysis(image_description_str))
        self.append_message(self.msg_feedback, "user", self.construct_history(self.history))      

        # Save conversation
        self.log_file.write(f"\n=== Conversation ===\n")

        return image_bboxes_array, image_description_str

    def get_response_by_feedback(self, user_input):
        self.append_message(self.msg_feedback, "user", self.response_format_feedback())
        self.append_message(self.msg_feedback, "user", user_input)      

        rawAssistant = self.get_ai_response(self.msg_feedback)
        self.append_message(self.msg_feedback, "assistant", rawAssistant)

        # Save conversation
        self.log_file.write(f"User: \n {user_input} \n")
        self.log_file.write(f"Assistant: \n {rawAssistant} \n")

        return rawAssistant
        
    def execute_feedback(self, user_input, image_bboxes_array, image_description_str):     
        self.append_message(self.msg_feedback, "user", self.response_format_execute_feedback()) 
        self.append_message(self.msg_feedback, "user", user_input)
        print(self.msg_feedback)
        new_state = self.string_to_tuple(self.get_ai_response(self.msg_feedback))
        action_to_goal = self.navi_model.navigate_to(self.navi_model.position, new_state, self.mapping.obstacles)

        # Put feedback mode label on the image
        image_pil_fmode = utils.put_text_top_left(image_bboxes_array, text="feedback mode")
        
        # Save conversation
        self.log_file.write(f"User: \n {user_input} \n")

        # store data and reset messages
        self.curr_state = new_state
        # self.store_image(image_pil_fmode)
        # self.update_log(assistant, user_input, image_description_str)
        # self.update_history_prompt(assistant, user_input, image_description_str)
        self.msg_feedback = []

        return action_to_goal

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

    def is_feedback_mode_exit(self, input): 
        messages = [
            {"role": "system", "content": """
        You are Go2, a robot dog assistant. You can only speak English even if the user speaks other languages. Your task is to search for a target object. You operate in two modes:
        1. Automatic Search Mode: Search independently based on your discretion.
        2. Feedback Mode: Follow user guidance to locate the object.

        You are currently in Feedback Mode. Your job is to evaluate the user's input and decide whether to switch back to Automatic Search Mode or stay in Feedback Mode.

        If the user's input indicates a desire to switch back to Automatic Search Mode, respond with 'true'. For all other inputs, respond with 'false'.
        """},
            {'role': 'user', 'content': input}
        ]

        params_for_interpreter = {
            "model": self.env['ai_model'],
            "messages": messages,
        }

        try:
            result = self.client.chat.completions.create(**params_for_interpreter)
            assistant_response = result.choices[0].message.content.strip()  # Strip unnecessary whitespace
            exit_fmode = assistant_response.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_feedback_mode_exit: {e}")
            return False
        return exit_fmode
