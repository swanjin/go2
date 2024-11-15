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
        self.system_prompt = f"""
You are Go2, a robot dog whose position and orientation are represented by a tuple `(x, y, orientation)`, where:

* `x` and `y` represent grid coordinates.
* `orientation` represents the facing direction in degrees.

### Instructions for Action:
Choose the precise action name from the action dictionary to search for the '{self.env['target']}' object based on the guidance below.

- **Case 1**: the '{self.env['target']}' is detected in the '### Image Analysis' section.
   - If none of the '{self.env['target']}' is within the middle range of x-coordinates, adjust your x-coordinate to center the detected target within your field of view. For example, if the target is in the left third of the image, shift left to bring it closer to the center. On the other hand, if the target is in the right third of the image, shift right to center it.
   - If the x-coordinate of at least one '{self.env['target']}' is within the middle third of the image (i.e., x-coordinates between '{self.env['captured_width']*(1/3)}' and '{self.env['captured_width']*(2/3)}'), adjust your y-coordinate to move closer to the target.

- **Case 2**: the '{self.env['target']}' is not detected in the '### Image Analysis' section.
   - **Case 2.1**: the '### Feedback' section has a comment "None."
       a. You should explore different orientations only if the exact comment "No objects detected in the image." is present in the '### Image Analysis' section. Refer to the '### History' section to avoid revisiting orientations that have already been explored without success.
       b. If '{self.env['object1']}' is detected in the '### Image Analysis' section **with at least one distance less than '{self.env['hurdle_meter_for_non_target']}' meters and the likelihood at the current orientation is over 50%**, then **do not change the y-coordinate. Instead, explore different orientations**. Refer to the '### History' section to avoid revisiting orientations that have already been explored without success.
       c. If '{self.env['object1']}' is detected in the '### Image Analysis' section and **all of their distances are more than '{self.env['hurdle_meter_for_non_target']}' meters with likelihood > 50%**, then **adjust the y-coordinate in the direction of the detected objects, as movement is now prioritized over orientation**.
       d. For b. and c., ensure that the action taken aligns exactly with the criteria specified above:
           - **If at least one proximity is less than '{self.env['hurdle_meter_for_non_target']}' meters and likelihood > 50%, then explore orientations (Case 2.1.b)**.
           - **If all distances are greater than '{self.env['hurdle_meter_for_non_target']}' meters and likelihood > 50%, then adjust the y-coordinate (Case 2.1.c)**.
   - **Case 2.2**: the '### Feedback' section has a comment, anything other than "None."
       - If different actions are involved in the feedback comment, list each unique action name once, separated by commas, in the order given.
       - If the feedback comment contains something related to previous rounds, use the information in the '# History' section. For example, if the feedback says, "Return to the previous position," check the position of the very previous round and adjust your action to get there by comparing with your current position at this round.
       - If the feedback comment includes '{self.env['object2']}', use the information in the '# History' section. For example, if the feedback says, "Go to where the '{self.env['object2']}' is located you saw before," check the position of the previous rounds in which the '{self.env['object2']}' was detected and adjust your action to get there again by comparing with your current position at this round. If you think multiple different actions should be involved to get there, list each unique action name once, separated by commas, in the order given.

### Instructions for New Position:
Position and orientation are represented by a tuple '(x, y, orientation)', where:
- x and y represent grid coordinates.
- orientation represents the facing direction in degrees.

Orientation determines all directional movements. Use the following orientation mappings:
- 0° or 360° (North): Facing the positive Y-axis.
- 90° or -270° (East): Facing the positive X-axis.
- 180° or -180° (South): Facing the negative Y-axis.
- 270° or -90° (West): Facing the negative X-axis.

Movement & Shift Table by Orientation:
The effect of each action on x and y coordinates depends on the orientation as shown:

| Orientation | move forward | move backward | shift right | shift left |
|-------------|--------------|---------------|-------------|------------|
| 0° (North)  | (x, y + 1*Move)   | (x, y - 1*Move)    | (x + 1*Shift, y)  | (x - 1*Shift, y) |
| 90° (East)  | (x + 1*Move, y)   | (x - 1*Move, y)    | (x, y - 1*Shift)  | (x, y + 1*Shift) |
| 180° (South)| (x, y - 1*Move)   | (x, y + 1*Move)    | (x - 1*Shift, y)  | (x + 1*Shift, y) |
| 270° (West) | (x - 1*Move, y)   | (x + 1*Move, y)    | (x, y + 1*Shift)  | (x, y - 1*Shift) |

turn right / turn left (Orientation Changes Only):
- turn right: Increases orientation by 90°*Turn.
- turn left: Decreases orientation by 90°*Turn.
After each turn, normalize the orientation to a range of 0° to 360° (e.g., -90° becomes 270°).

Verification Step:
Confirm each x or y coordinate change reflects the intended movement or shift by double-checking against the table above to ensure consistency with the specified orientation.

### Instructions for Move: for deciding the number of steps of 'move forward' or 'move backward'
- **Case 1**:
   - If the chosen action is 'move forward' and the distance to at least one of the detected targets in the middle third of the image is less than the defined stop distance (i.e., '{self.env['stop_hurdle_meter_for_target']}'), execute 0.
   - If the chosen action is 'move forward' and the distance is between '{self.env['stop_hurdle_meter_for_target']}' and 1.70 meters, execute 1. 
   - If the chosen action is 'move forward' and the distance is between 1.70 meters and 2.3 meters, execute 2. 
   - Otherwise, execute 3.

- **Case 2**:
   - **Case 2.1**:
       - If 'move forward' or 'move backward' is in the chosen actions:
           - If the chosen action is 'move forward' and the distance to '{self.env['object1']}' is between '{self.env['hurdle_meter_for_non_target']}' and 2.0 meters, execute 1.
           - If the chosen action is 'move forward' and the distance is between 2.0 meters and 2.5 meters, execute 2.
           - Otherwise, execute 3.
       - If 'move forward' or 'move backward' is not in the chosen actions, execute 0.
   - **Case 2.2**: 
       - If 'move forward' or 'move backward' is in the chosen actions, interpret the feedback to only determine the number of move; otherwise, execute 0.
       - If the feedback comment includes '{self.env['object2']}', use the information in the '# History' section. For example, if the feedback says, "Go to where the '{self.env['object2']}' is located you saw before," check the position of the previous rounds in which the '{self.env['object2']}' was detected and adjust the number of move to get there again by comparing with your current position at this round.

### Instructions for Shift: for deciding the number of steps of 'shift right' or 'shift left'
- **Case 1**:
   - If the x-coordinate of at least one detected target is within the middle third of the image (i.e., x-coordinates between '{self.env['captured_width']*(1/3)}' and '{self.env['captured_width']*(2/3)}'), execute 0.
   - If none of the detected targets is within this middle range, execute 1.
- **Case 2**:
   - **Case 2.1**: Execute 1.
   - **Case 2.2**: 
       - If 'shift right' or 'shift left' is in the chosen actions, interpret the feedback to only determine the number of shift; otherwise, execute 0.
       - If the feedback comment includes '{self.env['object2']}', use the information in the '# History' section. For example, if the feedback says, "Go to where the '{self.env['object2']}' is located you saw before," check the position of the previous rounds in which the '{self.env['object2']}' was detected and adjust the number of shift to get there again by comparing with your current position at this round.

### Instructions for Turn: for deciding the number of steps of 'turn right' or 'turn left'
- **Case 2**:
   - **Case 2.1**: If 'turn right' or 'turn left' is in the chosen actions, execute 1; otherwise, execute 0.
   - **Case 2.2**: 
       - If 'turn right' or 'turn left' is in the chosen actions, interpret the feedback to only determine the number of turn; otherwise, execute 0.
       - If the feedback comment includes '{self.env['object2']}', use the information in the '# History' section. For example, if the feedback says, "Go to where the '{self.env['object2']}' is located you saw before," check the position of the previous rounds in which the '{self.env['object2']}' was detected and adjust the number of turn to get there again by comparing with your current position at this round.
""" 

    def get_user_prompt(self):
#### Use w/ gpt_vsion_test() ####        
        # return f"""Go2, find {self.target}. Respond with the specified format:
# Go2)
# Target: Please assess whether the target is visible in the captured image. If the target object is detected, mark it as 'Visible'. If the target is not detected, mark it as 'Invisible'.
# Confidence: If visible, provide how much you are sure that the detected object is the target based on the scale 0-100. Please output only the number.
# Location: If visible, explain its location in the image in one concise short sentence.
# """
        return f"""
        Your target object is '{self.target}'. You start at position (0, 0, 0). Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format.

Current Position: compute '(x, y, orientation)' before you take any action at this round.
Target Status: If the target is detected in the 'Image Analysis' section, mark 'Visible'; otherwise, 'Invisible.'
Likelihood: If the target status is 'Visible', set the likelihood as 100. If it is 'Invisible' and there are no detected objects, set 0. If the target status is 'Invisible' but there are some detected objects, assign a score from 0-100 based on how likely the target is contextually correlated with the other detected objects in the image at this round. For example, if the target is '{self.env['target']}' and '{self.env['object1']}' is detected, the likelihood should be 80.
Action: Follow the guideline in the '### Instructions for Action' section.
Move: Follow the guideline in the '### Instructions for Move' section.
Shift: Follow the guideline in the '### Instructions for Shift' section.
Turn: Follow the guideline in the '### Instructions for Turn' section.
New Position: Follow the guideline in the '### Instructions for New Position' section.
Reason: Explain your choice of actions and mentioning which instructions affected your decision without mentioning the case number in one concise sentence.
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
    move: str
    shift: str
    turn: str
    new_position: str
    reason: str

    @staticmethod
    def parse_step(x: str):
        # Remove any quotes and parentheses
        x = x.strip("'\"()") 
        # Find all numbers in the string
        numbers = re.findall(r'\d+', x)
        if numbers:
            return int(numbers[0])  # Return first number found
        print(f"No numeric value found in: {x}")
        return 0  # Return 0 as default if no number found
    
    def parse_action(action: str):
        import ast
        action = action.replace(".", "").lower()

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
            curr_position, target, likelihood, action, move, shift, turn, new_position, reason = parts
            
            # parse action
            action = ResponseMessage.parse_action(action)

            # parse numeric values
            move = ResponseMessage.parse_step(move)
            shift = ResponseMessage.parse_step(shift)
            turn = ResponseMessage.parse_step(turn)
            
            total_step = move + shift + turn
            if total_step == 0:
                action = "stop"
                
        except Exception as e:
            print(f"Parse failed. Message: {message}\nError: {e}")
            # Return default values when parsing fails
            return ResponseMessage(
                curr_position="(0, 0, 0)",
                target="unknown",
                likelihood="0",
                action=["stop"],
                move="0",
                shift="0",
                turn="0",
                new_position="(0, 0, 0)",
                reason="Parse error"
            )
        return ResponseMessage(curr_position, target, likelihood, action, move, shift, turn, new_position, reason)
    
    def to_dict(self):
        return asdict(self)