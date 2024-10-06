import os
import base64
import datetime
import wave
import io
import pyaudio

from openai import OpenAI
import cv2

from ai_client_base import AiClientBase, ResponseMessage
from vision_owl import OWLDepthModel
from round import Round

class OpenaiClient(AiClientBase):
    def __init__(self, env, key):
        self.client = OpenAI(api_key=key)
        self.env = env
        self.image_counter = 0
        self.history = None
        self.owl_depth_model = OWLDepthModel(env)


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
        }
        self.openai_params_for_text = {
            "model": "gpt-4o",
            "messages": self.openai_prompt_messages_for_text,
            "max_tokens": 200,
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
        self.openai_goal["content"][0] = self.get_user_prompt() + "\n# History: \n" + "None"

    def save_round(self, assistant, feedback=None, feedback_factor=None, owl_response=None):
        self.history_log_file.write(f"======= image{len(self.round_list)+1} =======\n")
        if owl_response is not None:
            self.history_log_file.write(f"owl_response: {owl_response.description}\n")
        self.history_log_file.write(f"response: {assistant}\n")
        self.history_log_file.flush()

        self.round_list.append(Round(len(self.round_list) + 1, assistant, feedback, feedback_factor))
        if (len(self.round_list) == 1):
            self.history = "# History"
        if feedback is not None:
            pauseText = "By trigger" if feedback_factor else "By voluntary"
            feedbackText = f"There was a user feedback {pauseText}: {feedback}."
            self.history += f"\nRound {len(self.round_list)}: {feedbackText} You performed the action '{assistant.action}' and updated the position to {assistant.new_position} because {assistant.reason}"
        else:
            self.history += f"\nRound {len(self.round_list)}: From the position {assistant.curr_position}, {owl_response.description} You performed the action '{assistant.action}' and updated the position to {assistant.new_position} because {assistant.reason}"

        # The target had been {assistant.target} from the position of the previous round {assistant.curr_position}, where its likelihood score was {assistant.likelihood}. Therefore, in this round, Go2 performed the action '{assistant.action}' updating the position to {assistant.new_position}

        self.openai_goal["content"][0] = self.get_user_prompt() + self.history # for VLM?!

    def vision_model_test(self, cv2_image):
        owl_response = self.owl_depth_model.describe_image(cv2_image)
        self.store_image(owl_response.frame)

    def gpt_vision_test(self, cv2_image):
        _, buffer = cv2.imencode(".jpg", cv2_image)
        decoded = base64.b64encode(buffer).decode("utf-8")
        self.openai_goal["content"][-1] = {"image": decoded}
        result = self.client.chat.completions.create(**self.openai_params)
        rawAssistant = result.choices[0].message.content
        return rawAssistant

    def get_response_by_LLM(self, cv2_image):
        owl_response = self.owl_depth_model.describe_image(cv2_image)

        # Ensure self.history is initialized if not already
        if self.history is None or self.env["use_test_dataset"]:
            self.history = "# History \nNone."  # Initialize the history if it's missing

        if owl_response.description == "":
            image_description_text = "No objects detected in the image."
        else:
            image_description_text = owl_response.description
        # prompt input
        self.openai_goal_for_text["content"] = f"{self.get_user_prompt()}\n \n# Image description (The coordinate (0, 0) is located at the top-left corner of the image.)\n{image_description_text}\n \n{self.history}"
        if self.env["print_history"]:
            print(self.history)

        result = self.client.chat.completions.create(**self.openai_params_for_text)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)
        
        self.store_image(owl_response.frame)
        self.save_round(assistant, owl_response=owl_response)

        return rawAssistant, assistant

    def get_response_by_image(self, cv2_image):
        owl_response = self.owl_depth_model.describe_image(cv2_image, False)
        if owl_response.frame.shape[1] != self.env["captured_width"] or owl_response.frame.shape[0] != self.env["captured_height"]:
            resized_frame = cv2.resize(owl_response.frame, (self.env["captured_width"], self.env["captured_height"]))
        else:
            resized_frame = owl_response.frame

        _, buffer = cv2.imencode(".jpg", resized_frame)
        decoded = base64.b64encode(buffer).decode("utf-8")
        self.openai_goal["content"][-1] = {"image": decoded}
        result = self.client.chat.completions.create(**self.openai_params)
        rawAssistant = result.choices[0].message.content
        
        assistant = ResponseMessage.parse(rawAssistant)
        self.store_image(resized_frame)
        self.save_round(assistant, owl_response=owl_response)

        return rawAssistant, assistant

    def get_response_by_feedback(self, feedback, feedback_factor):
        self.openai_goal_for_text["content"] = self.get_user_prompt() + self.history + f"\n\n# Feedback:{feedback}"
        result = self.client.chat.completions.create(**self.openai_params_for_text)
        rawAssistant = result.choices[0].message.content
        assistant = ResponseMessage.parse(rawAssistant)
        
        self.save_round(assistant, feedback_factor)

        return rawAssistant, assistant

    def stt(self, voice_buffer):
        container = voice_buffer
        transcription = self.client.audio.transcriptions.create(
			model="whisper-1", 
			file=container,
			language='en'
		)
        return transcription.text

    def tts(self,reason):
        CHUNK = 1024
        with self.client.with_streaming_response.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=reason,
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

    def close(self):
        self.history_log_file.close()