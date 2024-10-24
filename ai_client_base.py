# ai_cllient_base.py
from dataclasses import dataclass, field, asdict
import datetime
from PIL import Image
import os
import re

import utils

class AiClientBase:
    def __init__(self, env):
        self.client = None
        self.env = env
        self.image_counter = 0
        self.round_list = []

        # try:
        #     os.makedirs('test', exist_ok=True)
        #     self.save_dir = f"test/test_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        #     os.mkdir(self.save_dir)
        #     self.history_log_file = open(f"{self.save_dir}/history.log", "a+") # append: a+ overwrite: w+
        # except Exception as e:
        #     print(f"Failed to create directory: {e}")
        
        # self.target = None  # Initialize target to None

        #### Use w/ gpt_vsion_test() ####        
        # self.system_prompt = """You are Go2, a robot dog."""
        self.system_prompt = f"""You are Go2, a robot dog. Your coordinates and orientation are represented by a position tuple (x, y, orientation), where x and y are your coordinates on a grid, and orientation describes your facing direction in degrees:
- 0 degrees or 360 degrees means facing north (along the positive Y-axis).
- 90 degrees or -270 degrees means facing east (along the positive X-axis).
- 180 degrees or -180 degrees means facing south (along the negative Y-axis).
- 270 degrees or -90 degrees means facing west (along the negative X-axis).   
You start at (0, 0, 0). One unit shift in x or y corresponds to 0.5 meters. 

Here is the action dictionary, formatted as 'action name: (x shift, y shift, clockwise rotation)':

- 'move forward': (0, 0.5, 0) 
- 'move backward': (0, -0.5, 0) 
- 'shift right': (1, 0, 0) 
- 'shift left': (-1, 0, 0) 
- 'turn right': (0, 0, 60) 
- 'turn left': (0, 0, -60)

# Instructions for Target Search and Navigation

1. Search for the Target Object:
- Use image analysis and historical data to guide your decisions. Follow real-time feedback to determine the most appropriate action.
2. Target Alignment:
- If the x-coordinate of at least one detected target is within the middle third of the image (i.e., x-coordinates between {self.env['captured_width']*(1/3)} and {self.env['captured_width']*(2/3)}), adjust your y-coordinate to move closer to that target. 
- If no targets are within this middle range, adjust your x-coordinate to center the detected target within your field of view. For example, if the target is in the left third of the image, 'shift left' to bring it closer to the center. On the other hand, if the target is in the right third of the image, 'shift right' to center it.
3. Handling Invisible Targets:
- If the target becomes invisible:
-- First, explore all possible orientations at the current x and y coordinates before moving to new ones. Use the following orientations: 0°, 60°, 120°, 180°, 240°, 300°.
-- Refer to the search history to avoid revisiting orientations that have already been explored without success.
-- If all orientations at the current x and y coordinates have been explored without finding the target, choose the orientation with the highest likelihood of success based on historical data at that location. Revisit that orientation and continue exploring in that direction and its neighboring orientations.
""" 
# 발견한 이후, keep centered가 move forward보다 먼저하도록 명시 안해야, move forward하다가 시야에서 사라져서 사람 도움 필요한 failure 상황이 생김.
# # Instructions
# - Search for the target object. Use the image analysis and history to guide your decisions, and follow feedback to determine your action.
# - If the x coodrinate of at least one detected target is in the middle of the image, that is, between {self.env['captured_width']*(1/3)} and {self.env['captured_width']*(2/3)}, adjust your y coordinates to move closer to it. If none of them is in the middle, adjust your x coordinates to center the detected target within your field of view.
# - If the distance of at least one detected target in the middle is less than {self.env['stop_hurdle_meter']} meters, execute the 'Stop' action. If the distances to all detected targets in the middle are greater than or equal to {self.env['stop_hurdle_meter']} meters, avoid the 'Stop' action.
# - If the target is invisible, first explore all possible orientations at the same x and y coordinates before moving to new ones. By referring to the history, avoid revisiting orientations already explored without success. If you have explored all possible orientations(0,60,120,180,240,300) at the same x and y coordinates without success, choose your action to revisit the orientation with the highest likelihood based on your history at that x and y coordinates. Once you reach that specific orientation, explore further in that direction and its neighboring ones.




    def get_user_prompt(self):
#### Use w/ gpt_vsion_test() ####        
#         return f"""Go2, find {self.target}. Respond with the specified format:
# Go2)
# Target: Please assess whether the target is visible in the captured image. If the target object is detected, mark it as 'Visible'. If the target is not detected, mark it as 'Invisible'.
# Confidence: If visible, provide how much you are sure that the detected object is the target based on the scale 0-100. Please output only the number.
# Location: If visible, explain its location in the image in one concise short sentence.
# """
        return f"""Your target object is '{self.target}'. When you respond, follow the structued format:
Current Position: Tuple (x, y, orientation) before the action.
Target Status: If any target is detected in the image analysis of this round, not those of previous rounds in the history, mark 'Visible'; otherwise, 'Invisible.'
Likelihood: If the target status is 'Visible', set likelihood to 100. If not, assign a score from 0-100 based on how likely the target is to be near detected objects or environments, considering contextual correlations.
Action: The exact action name in the action dictionary you choose by considering all the instructions.
New Position: Updated tuple (x, y, orientation) after the action.
Reason: Explain your choice in one concise sentence by mentioning which instructions affected your decision.
Step: If there is feedback, interpret the feedback to determine the step. If the distance to at least of one detected targets in the middle third of the image is less than the defined stop distance (i.e., {self.env['stop_hurdle_meter']} meters), execute the step = 0. If the distance is between {self.env['stop_hurdle_meter']} and 1.70 meters, execute step = 1, if the distance is between 1.70 meters and 2.3 meters, excute step = 2, else execute step = 3. If the target status is 'Invisible', exectute step = 1.
"""

    def set_target(self, target):
        self.target = target

    def stt(self, voice_buffer):
        return None

    def get_response_by_image(self, image_pil):
        pass

    def get_response_by_feedback(self, feedback):
        pass

    def store_image(self, image_array = None):
        if image_array is None:
            text = "The user provided feedback; no image was captured."
            image = Image.new('RGB', (self.env["captured_width"], self.env["captured_height"]), 'black')
            image = utils.put_text_in_the_middle(image, text, self.env["captured_width"], self.env["captured_height"])
        else:
            image = utils.OpenCV2PIL(image_array)
 
        ## add 'assistant' as a parameter into the function
        # if (self.env["text_insertion"]):
        #     text = "\n".join([f"Current Position: {assistant.curr_position}", f"Target: {assistant.target}", f"Likelihood: {assistant.likelihood}", f"Action: {assistant.action}", f"New Position: {assistant.new_position}", f"Reason: {assistant.reason}"])
        #     image = utils.image_text_append(image, self.env["captured_width"], self.env["captured_height"], text)
        
        # Format the filename based on the number of images stored
        self.image_counter += 1
        filename = f"image{self.image_counter:02d}.jpg"  # Pad with zeros to two digits
        image_path = os.path.join(self.save_dir, filename)

        # Save the image
        image.save(image_path)
        # print(f"Image saved to {image_path}")

    # def save_round(self, assistant, history, feedback=None, feedback_factor=None, image_analysis=None):
    #     self.history_log_file.write(f"======= image{len(self.round_list)+1} =======\n")
        
    #     if image_analysis:
    #         self.history_log_file.write(f"Image Analysis:\n {image_analysis.description}\n")
    #     else: 
    #         self.history_log_file.write(f"Image Analysis:\n None\n")

    #     if feedback:
    #         self.history_log_file.write(f"Feedback:\n {feedback}\n")
    #     else:
    #         self.history_log_file.write(f"Feedback:\n None\n")  # This prevents logging False

    #     self.history_log_file.write(f"Response:\n Action) {assistant.action}\n Reason) {assistant.reason}\n\n")
    #     self.history_log_file.flush()

    #     self.round_list.append(Round(len(self.round_list) + 1, assistant, history, feedback, feedback_factor, image_analysis))
        
    #     if (len(self.round_list) == 1):
    #         history = "# History"
    #     else:
    #         # pauseText = "by trigger" if feedback_factor else "by voluntary"
    #         feedbackText = f"The user provided feedback: {feedback}."
    #         history += (
    #             f"\nRound {len(self.round_list)}: "
    #             f"{feedbackText if feedback is not None else ''} "
    #             f"From the position {assistant.curr_position}, "
    #             f"{image_analysis.description if image_analysis is not None else ''} "
    #             f"The detection likelihood score for this position was {assistant.likelihood}. "
    #             f"You executed the '{assistant.action}' action and updated the position to {assistant.new_position}. "
    #             f"The rationale behind this action you told me was: '{assistant.reason}'"
    #         )

    def close(self):
        pass

@dataclass
class ResponseMessage:
    curr_position: str
    target: str
    likelihood: str
    action: str
    new_position: str
    reason: str
    step: str

    @staticmethod
    def parse(message: str):
        try:
            # Filter out lines that do not contain ':' and strip empty spaces
            parts = [line.split(":", 1)[1].strip() for line in message.split('\n') if ':' in line and len(line.strip()) > 0]
            if len(parts) != 7:
                raise ValueError("Message does not contain exactly seven parts")
            curr_position, target, likelihood, action, new_position, reason, step = parts
            print(step)
            step = re.findall(r'\d', step)
            if step:
                step = int(''.join(step))
            else:
                print("No Step.")
            if step == 0:
                action = "stop"
            if step == None:
                step = 1
            action = action.replace("** ", "").strip()
            print(ResponseMessage(curr_position, target, likelihood, action, new_position, reason, step))
        except Exception as e:
            print("parse failed. Message: ", message, "\nError: ", e)
            return ResponseMessage()
        return ResponseMessage(curr_position, target, likelihood, action, new_position, reason, step)
    
    def to_dict(self):
        return asdict(self)




# @dataclass
# class ResponseMessage:
#     fields: dict = field(default_factory=dict)
    
#     @staticmethod
#     def parse(message: str):
#         try:
#             # Split message by lines and filter out any lines that do not contain ':' and strip empty spaces
#             parts = [line.split(":", 1) for line in message.split('\n') if ':' in line and len(line.strip()) > 0]
#             # Create a dictionary with dynamic keys and their corresponding values
#             fields = {key.strip(): value.strip() for key, value in parts}
#         except Exception as e:
#             print("parse failed. Message: ", message, "\nError: ", e)
#             return ResponseMessage()

#         return ResponseMessage(fields)

#     def to_dict(self):
#         return asdict(self)
    
#     def get_field(self, field_name: str):
#         return self.fields.get(field_name, None)

#     def set_field(self, field_name: str, value: str):
#         self.fields[field_name] = value
