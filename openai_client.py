# openai_client.py
import os
import base64
import datetime
import wave
import io
import pyaudio
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.playback import play

from openai import OpenAI
import cv2

from ai_client_base import AiClientBase, ResponseMsg
from vision import VisionModel
from navigation import NaviModel, Mapping
from navi_config import NaviConfig

# from round import Round
import utils
from round import Round

class OpenaiClient(AiClientBase):
    def __init__(self, env, key):
        # Call the parent class's constructor to initialize system_prompt and other attributes
        super().__init__(env)
        self.env = env
        self.msg = []
        self.msg_feedback = []
        self.round_number = 1
        self.chat = []
        self.curr_state = utils.string_to_tuple(self.env['curr_state'])
        self.memory_list = []
        self.is_initial_prompt_landmark_or_non_command = True
        self.is_initial_response_format_non_command = True
        self.is_landmark_action = False
        self.is_landmark_state = None

        # Specify detectable areas
        self.snack_area1 = self.detectable_area(range(NaviConfig.snack1_bottom_left[0], NaviConfig.snack1_bottom_left[0]+NaviConfig.snack1_width+1), range(NaviConfig.snack1_bottom_left[1], NaviConfig.snack1_bottom_left[1]+NaviConfig.snack1_height+1), 90)
        self.sofa_area = self.detectable_area(range(NaviConfig.sofa_bottom_left[0], NaviConfig.sofa_bottom_left[0]+NaviConfig.sofa_width+1), range(NaviConfig.sofa_bottom_left[1], NaviConfig.sofa_bottom_left[1]+NaviConfig.sofa_height+1), 90)
        self.desk_area = self.detectable_area(range(NaviConfig.desk_bottom_left[0], NaviConfig.desk_bottom_left[0]+NaviConfig.desk_width+1), range(NaviConfig.desk_bottom_left[1], NaviConfig.desk_bottom_left[1]+NaviConfig.desk_height+1), 180)
        self.tv_area = self.detectable_area(range(NaviConfig.tv_bottom_left[0], NaviConfig.tv_bottom_left[0]+NaviConfig.tv_width+1), range(NaviConfig.tv_bottom_left[1], NaviConfig.tv_bottom_left[1]+NaviConfig.tv_height+1), 270)
        self.banana_area = self.detectable_area(range(NaviConfig.banana_bottom_left[0], NaviConfig.banana_bottom_left[0]+NaviConfig.banana_width+1), range(NaviConfig.banana_bottom_left[1], NaviConfig.banana_bottom_left[1]+NaviConfig.banana_height+1), 0)    
        self.fridge_area = self.detectable_area(range(NaviConfig.fridge_bottom_left[0], NaviConfig.fridge_bottom_left[0]+NaviConfig.fridge_width+1), range(NaviConfig.fridge_bottom_left[1], NaviConfig.fridge_bottom_left[1]+NaviConfig.fridge_height+1), 90)
        self.snack_area2 = self.detectable_area(range(NaviConfig.snack2_bottom_left[0], NaviConfig.snack2_bottom_left[0]+NaviConfig.snack2_width+1), range(NaviConfig.snack2_bottom_left[1], NaviConfig.snack2_bottom_left[1]+NaviConfig.snack2_height+1), 180)

        # Combine all detectable areas into a single set to remove duplicates
        self.all_detectable_areas = list(set(
            self.snack_area1 +
            self.sofa_area +
            self.desk_area +
            self.tv_area +
            self.banana_area +
            self.fridge_area +
            self.snack_area2
        ))

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

    def update_memory_list(self, detected_objects, distances, description, chat, assistant):
        round = Round(self.round_number, detected_objects, distances, description, chat, assistant)
        self.curr_state = round.assistant.new_state

        self.log_file.write(
            f"Round {round.round_number}:\n"
            f"- Detected Objects: {round.detected_objects if round.detected_objects else 'None'}\n"
            f"- Distances: {round.distances if round.distances else 'None'}\n"
            f"- Description: {round.description if round.description else 'None'}\n"
            f"- Chat: {round.chat if round.chat else 'None'}\n"
            f"- Initial State: {round.assistant.initial_state}\n"
            f"- Action: {round.assistant.action}\n"
            f"- New State: {round.assistant.new_state}\n"
            f"- Reason: {round.assistant.reason if round.assistant.reason else 'None'}\n\n"
        )

        self.log_file.flush()
        self.round_number += 1
        self.memory_list.append(round)

        return round.assistant

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

    def detectable_area(self, x_range, y_range, z_value):
        points_inside = [(x, y, z_value) for x in x_range for y in y_range]
        return points_inside
    
    def analyze_image(self, image_pil):
        image_analysis = self.vision_model.describe_image(image_pil)

        self.check_and_update_analysis(
            image_analysis, 
            self.snack_area1, 
            self.env['object4']
        )
        self.check_and_update_analysis(
            image_analysis, 
            self.sofa_area, 
            self.env['object5']
        )
        self.check_and_update_analysis(
            image_analysis, 
            self.desk_area, 
            self.env['object6']
        )
        self.check_and_update_analysis(
            image_analysis, 
            self.tv_area, 
            self.env['object7']
        )
        self.check_and_update_analysis(
            image_analysis, 
            self.banana_area, 
            self.env['object2']
        )
        self.check_and_update_analysis(
            image_analysis, 
            self.fridge_area, 
            self.env['object3']
        )
        self.check_and_update_analysis(
            image_analysis, 
            self.snack_area2, 
            self.env['object4']
        )

        return image_analysis.frame, image_analysis.detected_objects, image_analysis.distances, image_analysis.description

    def check_and_update_analysis(self, image_analysis, detectable_area, object_name):
        if self.curr_state in detectable_area and object_name not in image_analysis.detected_objects:
            distance = self.calculate_distance(object_name)
            description = f"You detected {object_name} with a distance of {distance} meters."
            image_analysis.detected_objects.append(object_name)
            image_analysis.distances.append(distance)
            image_analysis.description.append(description)

    def calculate_distance(self, object_name):
        curr_x = self.curr_state[0]
        curr_y = self.curr_state[1]
        if self.curr_state in self.snack_area1 and object_name == self.env['object4']:
            if curr_x in [-3, -2]:
                return '3.1'  # 2 steps
            elif curr_x in [-1]:
                return '2.6'  # 1 steps
        elif self.curr_state in self.sofa_area and object_name == self.env['object5']:
            if curr_x in [-2]:
                return '3.1'  
            elif curr_x in [-1]:
                return '2.6'
        elif self.curr_state in self.desk_area and object_name == self.env['object6']:
            if curr_y in [-3,-2]:
                return '1'
        elif self.curr_state in self.tv_area and object_name == self.env['object7']:
            if curr_x in [-1,0]:
                return '1'
        elif self.curr_state in self.banana_area and object_name == self.env['object2']:
            if curr_y in [-3]:
                return '5.1'  # 7 steps
            elif curr_y in [-2]:
                return '4.6'  # 6 steps
            elif curr_y in [-1]:
                return '4.1'  # 5 steps
            elif curr_y in [0]:
                return '3.6'  # 4 steps
            elif curr_y in [1]:
                return '3.1'  # 3 steps
            elif curr_y in [2]:
                return '2.6'  # 2 steps
        elif self.curr_state in self.fridge_area and object_name == self.env['object3']:
            if curr_x in [-1, 0]:
                return '3.1'  # 2 steps
            elif curr_x in [1]:
                return '2.6'  # 1 steps
        elif self.curr_state in self.snack_area2 and object_name == self.env['object4']:
            if curr_y in [4]:
                return '3.1'  # 2 steps
            elif curr_y in [3]:
                return '2.6'  # 1 steps

    def append_message(self, message, message_role: str, message_content: str):
        message.append({"role": message_role, "content": message_content})

    def get_ai_response(self, message):
        # print(message)
        result = self.client.chat.completions.create(
            model=self.env['ai_model'],
            messages=message,
            temperature=self.env['temperature']
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
    
    def check_action_same_as_previous_round(self, action, reason):
        if self.memory_list:
            prev_action = self.memory_list[-1].assistant.action
            if prev_action == action or (prev_action in [['move forward'], ['move forward', 'move forward']] and action in [['move forward'], ['move forward', 'move forward']]):
                reason = "Similar reason as the previous round."
        return reason

    def correct_next_position(self, current, actions):
        for action in actions:
            next_pos = NaviModel.get_next_position(current, action)
            current = next_pos
        return current
    
    def correct_action(self, action, detected_objects, distances, description):
        if not distances:
            return action

        try:
            if self.curr_state in self.snack_area1:
                distance_value = float(distances[detected_objects.index(self.env['object4'])])
            elif self.curr_state in self.sofa_area:
                distance_value = float(distances[detected_objects.index(self.env['object5'])])
            elif self.curr_state in self.desk_area:
                distance_value = float(distances[detected_objects.index(self.env['object6'])])
            elif self.curr_state in self.tv_area:
                distance_value = float(distances[detected_objects.index(self.env['object7'])])
            elif self.curr_state in self.banana_area:
                distance_value = float(distances[detected_objects.index(self.env['object2'])])
            elif self.curr_state in self.fridge_area:
                distance_value = float(distances[detected_objects.index(self.env['object3'])])
            elif self.curr_state in self.snack_area2:
                distance_value = float(distances[detected_objects.index(self.env['object4'])])

        except (ValueError, TypeError, IndexError) as e:
            print(f"Error converting distance to float: {e}")
            return action

        stop_hurdle = float(self.env.get('stop_landmark', 0))
        threshold_range = float(self.env['threshold_range'])

        # Subcase 2.1 logic
        if self.env['object2'] in detected_objects:
            print(f"Distance value: {distance_value}")
            if distance_value > (stop_hurdle + threshold_range * 6):
                action = ['move forward'] * 7
            elif distance_value > (stop_hurdle + threshold_range * 5):
                action = ['move forward'] * 6
            elif distance_value > (stop_hurdle + threshold_range * 4):
                action = ['move forward'] * 5
            elif distance_value > (stop_hurdle + threshold_range * 3):
                action = ['move forward'] * 4
            elif distance_value > (stop_hurdle + threshold_range * 2):
                action = ['move forward'] * 3
            elif distance_value > (stop_hurdle + threshold_range):
                action = ['move forward'] * 2
            # elif distance_value < stop_hurdle:
            #     action = ['turn right']  # or 'turn left', depending on previous action

        # Subcase 2.2 logic
        elif any(obj in detected_objects for obj in [self.env['object3'], self.env['object4'], self.env['object5']]):
            print(f"Distance value: {distance_value}")
            if distance_value > (stop_hurdle + threshold_range * 2):
                action = ['move forward'] * 2
            elif distance_value > (stop_hurdle + threshold_range):
                action = ['move forward']
            # elif distance_value < stop_hurdle:
            #     action = ['turn right']  # or 'turn left', depending on previous action

        # # Subcase 2.3 logic
        # elif any(obj in detected_objects for obj in [self.env['object6'], self.env['object7']]):
        #     action = ['turn right']  # or 'turn left', depending on previous action

        return action
    
    def get_response_by_LLM(self, image_pil, dog_instance):   
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            return None
        
        # Analyze image
        frame_bboxes_array, detected_objects, distances, description = self.analyze_image(image_pil)
        
        # Initialize messages
        self.initialize_prompt_auto(description)
        
        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            return None

        rawAssistant = self.get_ai_response(self.msg)
        assistant = ResponseMsg.parse(rawAssistant)

        # Post-processing assistant
        if self.curr_state in self.all_detectable_areas:
            assistant.action = self.correct_action(assistant.action, detected_objects, distances, description)

        assistant.new_state = self.correct_next_position(self.curr_state, assistant.action)
        # assistant.reason = self.check_action_same_as_previous_round(assistant.action, assistant.reason)

        # Check for feedback interruption early in the function
        if dog_instance.check_feedback_and_interruption():
            return None  

        # Update data
        self.store_image(frame_bboxes_array)
        self.update_memory_list(detected_objects, distances, description, self.chat, assistant)

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
        frame_bboxes_array, detected_objects, distances, description = self.analyze_image(image_pil)

        # Initialize messages
        self.initial_prompt_feedback(detected_objects)
        
        return frame_bboxes_array, detected_objects, distances, description

    def get_response_non_command(self, user_input):
        self.initial_response_format_non_command()
        self.append_message(self.msg_feedback, "user", user_input)
        self.append_message(self.chat, "user", user_input)

        rawAssistant = self.get_ai_response(self.msg_feedback)
        self.append_message(self.msg_feedback, "assistant", rawAssistant)
        self.append_message(self.chat, "assistant", rawAssistant)

        return rawAssistant
        
    def get_response_landmark_or_general_command(self, user_input, frame_bboxes_array, detected_objects, distances, description):
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
            self.is_landmark_action = True
            self.is_landmark_state = new_state
            print(f"Is landmark action: {self.is_landmark_action}")
            print(f"Is landmark state: {self.is_landmark_state}")
        else:
            print("❗ Executing general command")
            self.msg_feedback[0]['content'] = self.prompt_general_command(self.curr_state) # replace user prompt
            self.append_message(self.msg_feedback, "user", self.response_format_general_command())
            rawAssistant = self.get_ai_response(self.msg_feedback)
            assistant = ResponseMsg.parse(rawAssistant)

        # Update data
        image_pil_fmode = utils.put_text_top_left(frame_bboxes_array, text="Feedback mode")
        self.store_image(image_pil_fmode)
        self.update_memory_list(detected_objects, distances, description, self.chat, assistant)
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
        # env에서 tts가 false면 바로 리턴
        if not self.env.get('tts', True):  # tts 설정이 없으면 기본값 True
            return
            
        CHUNK = 1024
        if not isinstance(text, list):
            text = text
        else:
            text = self.parse_action_tts(text)

        # Open `/dev/null` and redirect stderr temporarily
        devnull = os.open(os.devnull, os.O_WRONLY)
        original_stderr = os.dup(2)
        os.dup2(devnull, 2)

        try:
            with self.client.with_streaming_response.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="wav"
            ) as response:
                container = io.BytesIO(response.read())
                # Load the entire audio using pydub
                audio_segment = AudioSegment.from_file(container, format="wav")
                
                # 속도
                faster_segment = speedup(audio_segment, playback_speed=self.env['tts_speed'])

                play(faster_segment)
        finally:
            os.dup2(original_stderr, 2)
            os.close(devnull)
            os.close(original_stderr)

    # def tts(self, text):
    #     # Check if TTS is enabled in the environment settings
    #     if not self.env.get('tts', True):  # Default to True if not specified
    #         return

    #     try:
    #         import pyttsx3

    #         # Convert text to string if it's a list
    #         if isinstance(text, list):
    #             text = self.parse_action_tts(text)

    #         # Improve pronunciation
    #         text = self.improve_pronunciation(text)

    #         # Initialize pyttsx3 engine
    #         engine = pyttsx3.init()

    #         # Set speech rate
    #         rate = int(200 * self.env.get('tts_speed', 0.8))
    #         engine.setProperty('rate', rate)

    #         # Set volume
    #         engine.setProperty('volume', 1.0)

    #         # Set voice to English if available
    #         voices = engine.getProperty('voices')
    #         for voice in voices:
    #             if "english" in voice.name.lower() or "en-" in voice.id.lower():
    #                 engine.setProperty('voice', voice.id)
    #                 break

    #         # Use the engine to say the text
    #         engine.say(text)
    #         engine.runAndWait()

    #         # Properly stop the engine to avoid callback issues
    #         engine.stop()

    #         print("[TTSWorker] TTS finished")
    #     except Exception as e:
    #         print(f"[TTSWorker] Error during TTS: {str(e)}")

    # def improve_pronunciation(self, text):
    #     """특정 단어나 구문의 발음을 개선하기 위한 텍스트 전처리 함수"""
    #     # Go2를 "Go two"로 변환
    #     text = text.replace("Go2", "Go two")
        
    #     # 필요한 경우 다른 발음 개선 규칙 추가
    #     # 예: text = text.replace("특정단어", "발음하기 쉬운 형태")
        
    #     return text

    def close(self):
        self.log_file.close()

    def is_instruction_command(self, input): 
        msg = []
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            "Your task: Determine if the user input is requesting you to perform any action or movement, OR providing location information that implies movement. "
            "Respond with 'true' if:\n"
            "- The user wants you to move somewhere\n"
            "- The user give you a direction\n"
            "- The user give you a landmark\n"
            "- The user wants you to perform any physical action\n"
            "- The user gives any kind of instruction or command\n"
            "- The user provides information about where an object is located (implying you should go there)\n"
            "- The user mentions 'between' landmarks or objects (implying a location to visit)\n"
            "Respond with 'false' if:\n"
            "- The user is asking a question NOT about the action or direction\n"
            "- The user is making a statement with no implication for movement\n"
            "- The user is just providing general information unrelated to locations\n"
            "Examples:\n"
            "- 'go to the apple' -> true\n"
            "- 'can you turn around' -> true\n"
            "- 'move forward' -> true\n"
            "- 'the apple (or target) is behind you' -> true\n"
            "- 'get close to the fridge (or other landmark)' -> true\n"
            "- 'go straight as far as you can' -> true\n"
            "- 'I think the apple is between banana and fridge' -> true\n"
            "- 'the target is located near the sofa' -> true\n"
            "- 'can you see the apple?' -> false\n"
            "- 'what is in front of you?' -> false"
        )
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)

        try:
            rawAssistant = self.get_ai_response(msg)
            is_command = rawAssistant.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_instruction_command: {e}")
            return False
        print(f"is_command: {is_command}")
        return is_command
    
    def is_yes(self, input):
        msg = []
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            "Your task: Determine if the user's response indicates agreement or confirmation. "
            "Respond with 'true' if the input expresses:\n"
            "- Agreement (e.g., 'yes', 'okay', 'sure', 'go ahead', 'do it', 'that's right')\n"
            "- Confirmation (e.g., 'that's what I want', 'exactly', 'perfect')\n"
            "- Positive acknowledgment (e.g., 'sounds good', 'that works')\n"
            "Respond with 'false' if the input expresses:\n"
            "- Disagreement (e.g., 'no', 'wait', 'stop', 'not quite')\n"
            "- Uncertainty (e.g., 'I'm not sure', 'let me think')\n"
            "- Different instructions or questions\n"
            "Examples:\n"
            "- 'that's exactly what I want' -> true\n"
            "- 'yes, go ahead' -> true\n"
            "- 'not what I meant' -> false\n"
            "- 'let me explain again' -> false"
        )
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)

        try:
            rawAssistant = self.get_ai_response(msg)
            return rawAssistant.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_yes: {e}")
            return False
        
    def is_landmark(self, input): 
        msg = []
        # Dynamically retrieve landmarks from NaviConfig
        landmarks_list = ", ".join(NaviConfig.landmarks.keys())
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            "Given the user input, determine if it indicates the user wants you to move by referencing box obstacles or any of the following landmarks: "
            f"{landmarks_list}. If the user input references any of these, respond with 'true'. Otherwise, respond with 'false'."
        )
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)

        try:
            rawAssistant = self.get_ai_response(msg)
            is_landmark = rawAssistant.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_landmark: {e}")
            return False
        print(f"is_landmark: {is_landmark}")
        return is_landmark
    
    def is_no_command(self, input):
        msg = []
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            "Your task: Determine if the user's response is a simple negation without any new instruction. "
            "\nRespond with 'true' if the input:\n"
            "- Only expresses disagreement (e.g., 'no', 'nope', 'wrong')\n"
            "- Indicates rejection without new instruction (e.g., 'that's not right', 'I don't want that')\n"
            "- Shows disapproval without alternative (e.g., 'that's incorrect', 'not what I meant')\n"
            "\nRespond with 'false' if the input:\n"
            "- Contains any new instruction (e.g., 'no, go to the apple instead')\n"
            "- Provides alternative action (e.g., 'no, you should turn right')\n"
            "- Includes specific correction (e.g., 'no, I meant the other direction')\n"
            "\nExamples:\n"
            "- 'no' -> true\n"
            "- 'that's not what I want' -> true\n"
            "- 'no, go to the apple' -> false\n"
            "- 'no, you should turn left instead' -> false"
        )
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)

        try:
            rawAssistant = self.get_ai_response(msg)
            return rawAssistant.lower() == "true"
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error in is_no_command: {e}")
            return False

    def get_landmark_name(self, input):
        msg = []
        # NaviConfig에서 랜드마크 목록 가져오기
        landmarks_list = ", ".join(NaviConfig.landmarks.keys())
        
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            f"Given the following landmarks: {landmarks_list}\n"
            "Determine which landmark the user is referring to in their input. "
            "Return EXACTLY one of the landmark names from the list above, or 'none' if no match is found.\n"
            "Examples:\n"
            "- 'can you go to the coffee machine?' -> coffee machine\n"
            "- 'move to the coffee maker' -> coffee machine\n"
            "- 'head towards the fridge' -> fridge\n"
            "- 'go to the sofa' -> sofa\n"
            "- 'move to the table' -> none"
        )
        
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)
        
        try:
            response = self.get_ai_response(msg).lower()
            # response가 landmarks 중 하나와 일치하면 그 landmark 반환
            for landmark in NaviConfig.landmarks.keys():
                if landmark.lower() == response:
                    return landmark
            return None
        except Exception as e:
            print(f"Error in get_landmark_name: {e}")
            return None

    def get_multiple_landmark_names(self, input):
        msg = []
        # NaviConfig에서 랜드마크 목록 가져오기
        landmarks_list = ", ".join(NaviConfig.landmarks.keys())
        
        prompt = (
            "You are Go2, a helpful robot dog assistant who only speaks English. "
            f"Given the following landmarks: {landmarks_list}\n"
            "Identify ALL landmarks the user is referring to in their input. "
            "Return a comma-separated list of ONLY the landmark names from the list above that appear in the input. "
            "If no landmarks are mentioned, return 'none'.\n"
            "Examples:\n"
            "- 'can you go to the coffee machine?' -> coffee machine\n"
            "- 'the apple is between the fridge and banana' -> fridge, banana\n"
            "- 'move to the area between the sofa and fridge' -> sofa, fridge\n"
            "- 'go to the table' -> none"
        )
        
        self.append_message(msg, "user", prompt)
        self.append_message(msg, "user", input)
        
        try:
            response = self.get_ai_response(msg).lower()
            if response == 'none':
                return []
                
            # 응답을 쉼표로 분리하고 각 항목을 정리
            landmark_candidates = [item.strip() for item in response.split(',')]
            
            # 실제 랜드마크와 일치하는 항목만 필터링
            found_landmarks = []
            for candidate in landmark_candidates:
                for landmark in NaviConfig.landmarks.keys():
                    if landmark.lower() == candidate:
                        found_landmarks.append(landmark)
                        break
            
            return found_landmarks
        except Exception as e:
            print(f"Error in get_multiple_landmark_names: {e}")
            return []
