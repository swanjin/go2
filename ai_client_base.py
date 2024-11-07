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

        #### Use w/ gpt_vsion_test() ####  #      
        # self.system_prompt = """You are Go2, a robot dog."""
        self.system_prompt = f"""You are Go2, a robot dog. Your coordinates and orientation are represented by a position tuple (x, y, orientation), where x and y are your coordinates on a grid, and orientation describes your facing direction in degrees:
- 0 degrees or 360 degrees means facing north (along the positive Y-axis).
- 90 degrees or -270 degrees means facing east (along the positive X-axis).
- 180 degrees or -180 degrees means facing south (along the negative Y-axis).
- 270 degrees or -90 degrees means facing west (along the negative X-axis).   
You start at (0, 0, 0). One unit shift in x or y corresponds to 1 meters. 

Here is the action dictionary, formatted as 'action name: (x shift, y shift, clockwise rotation)':

- 'move forward': (0, 1, 0) 
- 'move backward': (0, -1, 0) 
- 'shift right': (1, 0, 0) 
- 'shift left': (-1, 0, 0) 
- 'turn right': (0, 0, 60) 
- 'turn left': (0, 0, -60)

# Instructions for Action:
Choose the precise action name from the action dictionary to search for the '{self.env['target']}' object based on the guidance below.
- Case 1: the '{self.env['target']}' is detected in the '# Image Analysis' section
  - If none of the '{self.env['target']}' is within this middle range, adjust your x-coordinate to center the detected target within your field of view. For example, if the target is in the left third of the image, 'shift left' to bring it closer to the center. On the other hand, if the target is in the right third of the image, 'shift right' to center it.
  - If the x-coordinate of at least one '{self.env['target']}' is within the middle third of the image (i.e., x-coordinates between {self.env['captured_width']*(1/3)} and {self.env['captured_width']*(2/3)}), adjust your y-coordinate to move closer to the target.
- Case 2: the '{self.env['target']}' is not detected in the '# Image Analysis' section
  - Case 2.1: the '# Feedback' section has a comment "None."
    a. You should explore the different orientations only if the exact comment "No objects detected in the image." is present in the '# Image Analysis' section. Refer to the '# History' section to avoid revisiting orientations that have already been explored without success.
    b. If '{self.env['object1']}' are detected in the '# Image Analysis' section **with at least one distance less than {self.env['hurdle_meter_for_non_target']} meters and the likelihood at the current orientation is over 50%**, then **do not change the y-coordinate. Instead, explore different orientations**. Refer to the '# History' section to avoid revisiting orientations that have already been explored without success.
    c. If '{self.env['object1']}' are detected in the '# Image Analysis' section and **all of their distances are more than {self.env['hurdle_meter_for_non_target']} meters with likelihood > 50%**, then **adjust the y-coordinate in the direction of the detected objects, as movement is now prioritized over orientation**.
    d. For b. and c., ensure that the action taken aligns exactly with the criteria specified above:
      - **If at least one proximity is less than {self.env['hurdle_meter_for_non_target']} meters and likelihood > 50%, then explore orientations (Case 2.1.b)**.
      - **If all distances are greater than {self.env['hurdle_meter_for_non_target']} meters and likelihood > 50%, then adjust the y-coordinate (Case 2.1.c)**.
  - Case 2.2: the '# Feedback' section has a comment, anything other than "None."
    - If different actions are involved in the feedback comment, list each unique action name once, separated by commas, in the order given.
    - If the feedback comment contains something related to previous rounds, use the information in the '# History:' section. For example, if the feedback says, 'Return to the previous position.', check the position of the very previous round and adjust your action to get there by comparing your current position at this round.

# Instructions for Move: for deciding the number of steps of 'move forward' or 'move backward' 
- Case 1: 
  - If the chosen action is 'move forward' and the distance to at least one of the detected targets in the middle third of the image is less than the defined stop distance (i.e., {self.env['stop_hurdle_meter_for_target']}), execute 0.
  - If the chosen action is 'move forward' and the distance is between {self.env['stop_hurdle_meter_for_target']} and 1.70 meters, execute 1. 
  - If the chosen action is 'move forward' and the distance is between 1.70 meters and 2.3 meters, execute 2. 
  - Otherwise, execute 3.
- Case 2:
  - Case 2.1:
    - If 'move forward' or 'move backward' is in the chosen actions, 
      - If the chosen action is 'move forward' and the distance to '{self.env['object1']}' is between {self.env['hurdle_meter_for_non_target']} and 2.0 meters, execute 1. 
      - If the chosen action is 'move forward' and the distance is between 2.0 meters and 2.5 meters, execute 2. 
      - Otherwise, execute 3.
    - If 'move forward' or 'move backward' is not in the chosen actions, execute 0.
  - Case 2.2: 'move forward' or 'move backward' is in the chosen actions, interpret the feedback to only determine the number of move; otherwise, execute 0.

# Instructions for Shift: for deciding the number of steps of 'shift right' or 'shift left'
- Case 1:
  - If the x-coordinate of at least one detected target is within the middle third of the image (i.e., x-coordinates between 427 and 854), execute 0.
  - If none of the detected targets is within this middle range, execute 1.
- Case 2: 
  - Case 2.1: 
    - Execute 1.
  - Case 2.2: 
    - If 'shift right' or 'shift left' is in the chosen actions, interpret the feedback to only determine the number of move; otherwise, execute 0.

# Instructions for Turn: for deciding the number of steps of 'turn right' or 'turn left' 
- Case 2: 
  - Case 2.1: 
    - If 'turn right' or 'turn left' is in the chosen actions, execute 1; otherwise, execute 0.
  - Case 2.2: 
    - If 'turn right' or 'turn left' is in the chosen actions, interpret the feedback to only determine the number of move; otherwise, execute 0.
""" 

    def get_user_prompt(self):
#### Use w/ gpt_vsion_test() ####        
        # return f"""Go2, find {self.target}. Respond with the specified format:
# Go2)
# Target: Please assess whether the target is visible in the captured image. If the target object is detected, mark it as 'Visible'. If the target is not detected, mark it as 'Invisible'.
# Confidence: If visible, provide how much you are sure that the detected object is the target based on the scale 0-100. Please output only the number.
# Location: If visible, explain its location in the image in one concise short sentence.
# """
        return f"""Your target object is '{self.target}'. Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format.
    Current Position: Tuple (x, y, orientation) before the action.
    Target Status: If the target is detected in the 'Image Analysis' section, mark 'Visible'; otherwise, 'Invisible.'
    
    Contextual Likelihood: If the target status is 'Visible', set the likelihood as 100. If it is 'Invisible' and there are no detected objects, set 0. If the target status is 'Invisible' but there are some detected objects, assign a score from 0-100 based on how likely the target is contextually correlated with the other detected objects in the image at this round. For example, if the target is '{self.env['target']}' and '{self.env['object1']}' is detected, the likelihood should be 80.
    Action: Follow the guideline in the '# Instructions for Action' section.
    New Position: Updated tuple (x, y, orientation) after the action.
    Reason: Explain your choice in one concise sentence by mentioning which instructions affected your decision.
    Move: Follow the guideline in the '# Instructions for Move' section.
    Shift: Follow the guideline in the '# Instructions for Shift' section.
    Turn: Follow the guideline in the '# Instructions for Turn' section.
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
    action: list
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
        import ast
        if action.startswith("[") or action.endswith("]"):
            action = action[2:-2]
        actions = [act.strip() for act in action.split(',')]
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