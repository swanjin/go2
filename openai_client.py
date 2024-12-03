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
# from round import Round
import utils
from round import Round

class OpenaiClient(AiClientBase):
    def __init__(self, env, key):
        # Call the parent class's constructor to initialize system_prompt and other attributes
        super().__init__(env)
        
        self.client = OpenAI(api_key=key)
        self.env = env
        self.image_counter = 0
        self.history = "None."
        self.vision_model = VisionModel(env)
        self.openai_params_for_LLM = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ""}
            ],
            "max_tokens": 200,
            "temperature": 0
        }
        self.round_list = []

        try:
            os.makedirs('test', exist_ok=True)
            self.save_dir = f"test/test_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.mkdir(self.save_dir)
            self.history_log_file = open(f"{self.save_dir}/history.log", "a+") # append: a+ overwrite: w+
        except Exception as e:
            print(f"Failed to create directory: {e}")

    def set_target(self, target):
        self.target = target

    def save_round(self, assistant, feedback, image_description_text):
        # Update history.log
        self.history_log_file.write(f"\n=== image{len(self.round_list)+1} ===\n")
        
        if image_description_text == "No objects related to the target is detected.":
            self.history_log_file.write(f"Image Analysis: \n None. \n")
        else: 
            self.history_log_file.write(f"Image Analysis: \n{image_description_text} \n")

        if feedback:
            self.history_log_file.write(f"Feedback: \n {feedback} \n")

        if assistant is None:
            self.history_log_file.write(f"Response: \n None.")  # This prevents logging False
        else:
            self.history_log_file.write(
                f"Response: \n"
                f"Likelihood) {assistant.likelihood} \n"
                f"Action) {assistant.action} \n"
                f"Move) {assistant.move} \n"
                f"Shift) {assistant.shift} \n"
                f"Turn) {assistant.turn} \n"
                f"Reason) {assistant.reason} \n"
            )

        self.history_log_file.flush()

    def update_history_prompt(self, assistant, feedback, image_description_text):
        self.round_list.append(Round(len(self.round_list) + 1, assistant, feedback))
        round_number = len(self.round_list)
        
        self.history = self.history if round_number > 1 else ""
        
        # #### Test feedback prompt
        # if round_number == 1:
        #     image_description_text = "You detected banana at pixel index (640, 360) with a distance of 5 meters."
    
        self.history += (
            f"- Round {round_number}: "
            f"The user provided feedback: {feedback}. "
            f"From the tuple {assistant.curr_tuple}, {image_description_text} "
            f"The likelihood of target presence at this tuple was {assistant.likelihood}. "
            f"You executed the '{assistant.action}' action which led to the updated tuple of {assistant.new_tuple}. \n"
            # f"The rationale behind this action you told me was: '{assistant.reason}' \n"
        )
        # self.history += "None."

    def get_response_by_LLM(self, image_pil, dog_instance, feedback = None):
        # Reset to the initial state
        self.reset_messages()
        
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None
        
        image_analysis = self.vision_model.describe_image(image_pil)

        if image_analysis.description == "":
            image_description_text = "No objects related to the target is detected."
        else:
            image_description_text = image_analysis.description

        if feedback is None:
            feedback = "None"

        # ### test 
        # if dog_instance.round_number == 1:
        #     image_description_text = "You detected bottle at pixel index (665, 236) with a distance of 2.65 meters."
        # if dog_instance.round_number == 2:
        #     image_description_text = "You detected plant at pixel index (665, 236) with a distance of 2.65 meters."

        # input prompt
        self.openai_params_for_LLM["messages"][1]["content"] = (
            f"{self.action_auto_format()}\n\n" # Answer format for robot's action in automatic mode
            f"### Image analysis:\n (The image size is {self.env['captured_width']}x{self.env['captured_height']}, with the pixel index (0, 0) located at the top-left corner.):\n {image_description_text} \n\n"
            f"### History:\n {self.history}\n\n"
        )
        # print(self.openai_params_for_LLM["messages"][1:])
        
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None

        if self.env["print_history"]:
            print(self.history)
        
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None

        result = self.client.chat.completions.create(**self.openai_params_for_LLM)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1
            return None  

        image_array_bboxes = image_analysis.frame
        if feedback == "None":
            self.store_image(image_array_bboxes)

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1 
            return None

        self.save_round(assistant, feedback, image_description_text)
        self.update_history_prompt(assistant, feedback, image_description_text)

        return assistant

    def append_prompt(self, message_role: str, message_content: str):
        self.openai_params_for_LLM["messages"].append({"role": message_role, "content": message_content})

    def feedback_mode_on(self, image_pil):
        # Modify system prompt
        self.openai_params_for_LLM["messages"][0]["content"] = self.system_prompt_feedback
        
        # Start new feedback mode
        image_analysis = self.vision_model.describe_image(image_pil)
        image_array = image_analysis.frame

        if image_analysis.description == "":
            image_description_text = "No objects related to the target is detected."
        else:
            image_description_text = image_analysis.description
        
        self.openai_params_for_LLM["messages"][1]["content"] = (
            f"### Image analysis:\n (The image size is {self.env['captured_width']}x{self.env['captured_height']}, with the pixel index (0, 0) located at the top-left corner.):\n {image_description_text} \n\n"
            f"### History:\n {self.history}\n\n"
            f"### Conversation:\n Refer to the below conversation between you and the user."
        )
        
        if self.env["print_history"]:
            print(self.history)

        return image_array, image_description_text

    def get_response_by_feedback(self, questions):
        self.append_prompt("user", questions)
        
        # Answer format for user's questions
        self.append_prompt("user", self.questions_feedback_format(questions))
        # print(self.openai_params_for_LLM["messages"][2:]) # for debug

        result = self.client.chat.completions.create(**self.openai_params_for_LLM)
        response = result.choices[0].message.content

        # Replace answer format with AI's response
        self.openai_params_for_LLM["messages"] = self.openai_params_for_LLM["messages"][:-1] 
        self.append_prompt("assistant", response)
        # print(self.openai_params_for_LLM["messages"][2:]) # for debug

        return response
        
    def feedback_to_action(self, feedback, image_array_bboxes, image_description_text):
        self.append_prompt("user", feedback)
        
        # # Answer format for user's questions
        # self.append_prompt("user", self.questions_feedback_format(feedback)) 

        # result = self.client.chat.completions.create(**self.openai_params_for_LLM)
        # confirmation_msg = result.choices[0].message.content

        # # Replace answer format with AI's response
        # self.openai_params_for_LLM["messages"] = self.openai_params_for_LLM["messages"][:-1] 
        # self.append_prompt("assistant", confirmation_msg)

        # Answer format for robot's action
        self.append_prompt("user", self.action_feedback_format()) 
        result = self.client.chat.completions.create(**self.openai_params_for_LLM)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)

        image_pil_fmode = utils.put_text_top_left(image_array_bboxes, text="feedback mode")
        self.store_image(image_pil_fmode)
        # self.history_log_file.write(f"Go2: {confirmation_msg} \n")
        # self.history_log_file.flush()
        self.save_round(assistant, feedback, image_description_text)
        self.update_history_prompt(assistant, feedback, image_description_text)

        # return confirmation_msg, assistant
        return assistant
    
    def reset_messages(self):
        self.openai_params_for_LLM["messages"].clear()
        self.openai_params_for_LLM = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ""}
            ],
            "max_tokens": 200,
            "temperature": 0
        }

    def reset_messages_feedback(self):
        self.openai_params_for_LLM["messages"].clear()
        self.openai_params_for_LLM = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": self.system_prompt_feedback},
                {"role": "user", "content": ""}
            ],
            "max_tokens": 200,
            "temperature": 0
        }
  
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
        self.history_log_file.close()

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
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0
        }

        try:
            result = self.client.chat.completions.create(**params_for_interpreter)
            assistant_response = result.choices[0].message.content.strip()  # Strip unnecessary whitespace
            exit_fmode = assistant_response.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_feedback_mode_exit: {e}")
            return False
        return exit_fmode
