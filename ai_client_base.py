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

    def get_user_prompt(self):
#### Use w/ gpt_vsion_test() ####        
#         return f"""Go2, find {self.target}. Respond with the specified format:
# Go2)
# Target: Please assess whether the target is visible in the captured image. If the target object is detected, mark it as 'Visible'. If the target is not detected, mark it as 'Invisible'.
# Confidence: If visible, provide how much you are sure that the detected object is the target based on the scale 0-100. Please output only the number.
# Location: If visible, explain its location in the image in one concise short sentence.
# """
        return f"""Your target object is '{self.target}'. Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format.
Current Position: Tuple (x, y, orientation) before the action.
Target Status: If any target is detected in the image analysis of this round, not those of previous rounds in the history, mark 'Visible'; otherwise, 'Invisible.'
Likelihood: If the target status is 'Visible', set likelihood to 100. If not, assign a score from 0-100 based on how likely the target is to be near detected objects or environments, considering contextual correlations.
Action: If the # Feedback section has the comment "None.", select the precise action name from the action dictionary based on all given instructions. If there are additional comments in the # Feedback section (anything other than "None."), interpret the feedback to determine all action names. If there's one action in feedback, execute the exact action name in the action dictionary. If there's two or more actions in feedback, execute the exact action names with comma in order. (ex. Feedback: move forward 3 times and turn right 2 times ----> execute: move forward, turn right).
New Position: Updated tuple (x, y, orientation) after the action.
Reason: Explain your choice in one concise sentence by mentioning which instructions affected your decision.
Move: If there is feedback, interpret the feedback to only determine the number of move for "move forward" or "move backward". If you think there's no feedback for number of move, execute 0. If there's no feedback and the distance to at least of one detected targets in the middle third of the image is less than the defined stop distance (i.e., {self.env['stop_hurdle_meter']} meters), execute the step = 0. If the distance is between {self.env['stop_hurdle_meter']} and 1.70 meters, execute step = 1, if the distance is between 1.70 meters and 2.3 meters, excute step = 2, else execute step = 3. If the target status is 'Invisible', exectute step = 1.
Shift: If there is feedback, interpret the feedback to only determine the number of shift for "shift left" or "shift right". If you think there's no feedback for number of shift, execute 0. If there's no feedback, execute shift = 1.
Turn: If there is feedback, interpret the feedback to only determine the number of move for "turn right" or turn left". If you think there's no feedback for number of turn, execute 0. If there's no feedback, execute turn = 1.
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
    move: str
    shift: str
    turn: str

    def parse_step(x: str):
        x = re.findall(r'\d', x)
        if x:
            x = int(''.join(x))
        else:
            print("No " + x)
        return x
    
    def parse_action(action: str):
        actions = [act.strip() for act in action.split('.')]
        return actions

    @staticmethod
    def parse(message: str):
        try:
            # Filter out lines that do not contain ':' and strip empty spaces
            parts = [line.split(":", 1)[1].strip() for line in message.split('\n') if ':' in line and len(line.strip()) > 0]
            if len(parts) != 9:
                raise ValueError("Message does not contain exactly nine parts")
            curr_position, target, likelihood, action, new_position, reason, move, shift, turn = parts
            
            # parse action
            action = ResponseMessage.parse_action(action)

            # parse step
            # print(move, shift, turn)
            move = ResponseMessage.parse_step(move)
            shift = ResponseMessage.parse_step(shift)
            turn = ResponseMessage.parse_step(turn)
            # print(move, shift, turn)
            
            total_step = move + shift + turn
            if total_step == 0:
                action = "stop"
            # print(ResponseMessage(curr_position, target, likelihood, action, new_position, reason, move, shift, turn))
        except Exception as e:
            print("parse failed. Message: ", message, "\nError: ", e)
            return ResponseMessage()
        return ResponseMessage(curr_position, target, likelihood, action, new_position, reason, move, shift, turn)
    
    def to_dict(self):
        return asdict(self)