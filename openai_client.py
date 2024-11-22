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

        self.openai_prompt_messages = [
            {"role": "system", "content": self.system_prompt 
        }]
        self.openai_prompt_messages_for_text = [
            {"role": "system", "content": self.system_prompt 
        }]
        self.openai_params = {
            "model": "gpt-4o",
            "messages": self.openai_prompt_messages,
            "max_tokens": 200,
            # "temperature": 0
        }
        self.openai_params_for_text = {
            "model": "gpt-4o",
            "messages": self.openai_prompt_messages_for_text,
            "max_tokens": 200,
            # "temperature": 0
        }
        self.openai_goal = {
            "role": "user",
            "content": [None, None, ]}
        self.openai_goal_for_text = {
            "role": "user",
            "content": ""
        }
        self.openai_prompt_messages.append(self.openai_goal)
        self.openai_prompt_messages_for_text.append(self.openai_goal_for_text)
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
        # self.openai_goal["content"][0] = self.get_user_prompt() + "\n# History: \n None."

    def save_round(self, assistant, feedback, image_description_text):
        # Update history.log
        self.history_log_file.write(f"======= image{len(self.round_list)+1} =======\n")
        
        if image_description_text == "No objects detected in the image.":
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
                f"Turn) {assistant.turn} \n\n"
                f"Reason) {assistant.reason} \n"
            )

            self.history_log_file.flush()

    def vision_model_test(self, image_pil):
        image_analysis = self.vision_model.describe_image(image_pil)
        self.store_image(image_analysis.frame)
        self.save_round(image_analysis=image_analysis)

    def gpt_vision_test(self, image_pil):
        _, buffer = cv2.imencode(".jpg", image_pil)
        decoded = base64.b64encode(buffer).decode("utf-8")
        self.openai_goal["content"][-1] = {"image": decoded}
        result = self.client.chat.completions.create(**self.openai_params)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)

        return assistant

    def update_history_prompt(self, assistant, feedback, image_description_text):
        self.round_list.append(Round(len(self.round_list) + 1, assistant, feedback))
        round_number = len(self.round_list)
        
        self.history = self.history if round_number > 1 else ""
        
        # #### Test feedback prompt
        # if round_number == 1:
        #     image_description_text = "You detected banana at coordinates (640, 360) with a distance of 5 meters."
    
        self.history += (
            f"- Round {round_number}: "
            f"The user provided feedback: {feedback}. "
            f"From the position {assistant.curr_position}, {image_description_text} "
            f"The likelihood of target presence at this position was {assistant.likelihood}. "
            f"You executed the '{assistant.action}' action which led to the updated position of {assistant.new_position}. \n"
            # f"The rationale behind this action you told me was: '{assistant.reason}' \n"
        )
        # self.history += "None."

    def get_response_by_LLM(self, image_pil, dog_instance, feedback = None):

        # self.openai_prompt_messages_for_text = [
        #     {"role": "system", "content": self.system_prompt 
        # }]

        image_analysis = self.vision_model.describe_image(image_pil)

        if image_analysis.description == "":
            image_description_text = "No objects detected in the image."
        else:
            image_description_text = image_analysis.description

        # ## Test likelihood for invisible cases
        # if image_analysis.description == "" and not dog_instance.round_number == 2:
        #     image_description_text = "No objects detected in the image."
        # elif image_analysis.description == "" and dog_instance.round_number == 2:
        #     image_description_text = "You detected refrigerator at coordinates (640, 360) with a distance of 5 meters."
        # else:
        #     image_description_text = image_analysis.description
        
        # if self.history is None or self.env["use_test_dataset"]:
        #     self.history = "# History:\n None."

        if feedback is None:
            feedback = "None"

        # #### test
        # if dog_instance.round_number == 2:
        #     image_description_text = "You detected refrigerator at coordinates (665, 236) with a distance of 2.65 meters."

        # input prompt
        print(self.openai_prompt_messages_for_text)

        self.openai_goal_for_text["content"] = (
            f"{self.get_user_prompt()}\n\n"
            f"### Image analysis:\n (The image size is {self.env['captured_width']}x{self.env['captured_height']}, with the coordinate (0, 0) located at the top-left corner.):\n {image_description_text} \n\n"
            f"### History:\n {self.history}\n\n"
            f"### Feedback:\n {feedback}."
        )

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

        result = self.client.chat.completions.create(**self.openai_params_for_text)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1 
            return None  

        if feedback == "None":
            self.store_image(image_analysis.frame)
        else:
            self.store_image()

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            dog_instance.round_number += 1 
            return None  

        self.save_round(assistant, feedback, image_description_text)
        self.update_history_prompt(assistant, feedback, image_description_text)

        return assistant

    def get_response_by_image(self, image_pil):
        image_analysis = self.vision_model.describe_image(image_pil, False)
        if image_analysis.frame.shape[1] != self.env["captured_width"] or image_analysis.frame.shape[0] != self.env["captured_height"]:
            resized_frame = cv2.resize(image_analysis.frame, (self.env["captured_width"], self.env["captured_height"]))
        else:
            resized_frame = image_analysis.frame

        _, buffer = cv2.imencode(".jpg", resized_frame)
        decoded = base64.b64encode(buffer).decode("utf-8")
        self.openai_goal["content"][0] = self.get_user_prompt() + self.history # for VLM
        self.openai_goal["content"][-1] = {"image": decoded}

        result = self.client.chat.completions.create(**self.openai_params)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)

        resized_image_pil = utils.OpenCV2PIL(resized_frame)
        self.store_image(resized_image_pil)
        self.save_round(assistant)

        return assistant
    
    def feedback_mode_on(self, image_pil):
        
        # self.openai_prompt_messages_for_text = [
        #     {"role": "system", "content": self.system_prompt 
        # }]

        image_analysis = self.vision_model.describe_image(image_pil)

        if image_analysis.description == "":
            image_description_text = "No objects detected in the image."
        else:
            image_description_text = image_analysis.description
        
        self.openai_goal_for_text["content"] = (
            f"### Image analysis:\n (The image size is {self.env['captured_width']}x{self.env['captured_height']}, with the coordinate (0, 0) located at the top-left corner.):\n {image_description_text} \n\n"
            f"### History:\n {self.history}\n\n"
            f"### Conversation:\n Refer to the below conversation between you and the user."
        )
        self.openai_prompt_messages_for_text.append(self.openai_goal_for_text)
        
        if self.env["print_history"]:
            print(self.history)


    def get_response_by_feedback(self, text):

        self.openai_prompt_messages_for_text.append({"role": "user", "content": text})
        self.openai_prompt_messages_for_text.append({"role": "user", "content": self.get_user_prompt_for_questions()})

        result = self.client.chat.completions.create(**self.openai_params_for_text)
        response = result.choices[0].message.content
        self.openai_prompt_messages_for_text.append({"role": "assistant", "content": response})

        return response
        
    
    def feedback_to_action(self, feedback):
        
        self.openai_prompt_messages_for_text.append({"role": "user", "content": feedback})
        self.openai_prompt_messages_for_text.append({"role": "user", "content": self.get_user_prompt_for_questions()})
        result = self.client.chat.completions.create(**self.openai_params_for_text)
        confirmation_msg = result.choices[0].message.content
        self.openai_prompt_messages_for_text.append({"role": "assistant", "content": confirmation_msg})
        print(confirmation_msg)

        self.openai_prompt_messages_for_text.append({"role": "user", "content": self.get_user_prompt()})
        # print(self.openai_prompt_messages_for_text)
        
        result = self.client.chat.completions.create(**self.openai_params_for_text)
        rawAssistant = result.choices[0].message.content
        self.openai_prompt_messages_for_text.append({"role": "assistant", "content": rawAssistant})
        
        assistant = ResponseMessage.parse(rawAssistant)

        self.store_image()
        self.save_round(assistant, feedback, None)
        self.update_history_prompt(assistant, feedback=feedback, image_description_text="No objects detected in the image.")

        return confirmation_msg, assistant
    
    def clear_gpt_message(self):
        self.openai_prompt_messages_for_text.clear()

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