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


        self.system_prompt = f"""
You are Go2, a robot dog assistant. You can only speak English regardless of the language the user uses. Your position and orientation are represented by a tuple (x, y, orientation), where:

- x and y are grid pixel index representing your position.
- orientation is the facing direction in degrees.

Your task is to search for the target object, '{self.env['target']}', starting at (0, 0, 0). You can only see objects in your facing direction and must adjust your orientation to face the target while searching.

### Instructions for Action:
Action dictionary:
- 'move forward'
- 'move backward'
- 'shift right' 
- 'shift left'
- 'turn right'
- 'turn left'
- 'stop'

Choose the precise action name from the action dictionary to search for the '{self.env['target']}' object based on conversation between you and the user. 
- For the case that only one type of action needs to be executed, list is only one time.
- If multiple different actions need to be executed based on the conversation, list the action that changes the orientation first, then the action that changes the position. Identify each unique action from the action dictionary and list them once, separated by commas.

#### Case 1: The '{self.env['target']}' is detected in the `### Image Analysis` section.
   - **Subcase 1.1**: The '{self.env['target']}' is within the middle third of the image's width (i.e., the width pixel index between '{self.env['captured_width']*(1/3)}' and '{self.env['captured_width']*(2/3)}'), and the distance to **all** '{self.env['target']}' is more than the defined stop distance ('{self.env['stop_hurdle_meter_for_target']}').
     - **Action**: Move your position vertically to get closer to the target.

   - **Subcase 1.2**: No '{self.env['target']}' falls within the middle third of the image's width.
     - **Action**: Consider shifting your position horizontally to center the detected target within your field of view. 
       - Example: 
         - If the target is in the left third of the image's width (the width pixel index less than '{self.env['captured_width']*(1/3)}'), **shift left** to bring it closer to the center. 
         - If the target is in the right third of the image's width (the width pixel index greater than '{self.env['captured_width']*(2/3)}'), **shift right** to bring it closer to the center.

   - **Subcase 1.3**: **All** '{self.env['target']}' are within the middle third of the image's width (i.e., between '{self.env['captured_width']*(1/3)}' and '{self.env['captured_width']*(2/3)}'), and the distance to **all** '{self.env['target']}' is less than the defined stop distance ('{self.env['stop_hurdle_meter_for_target']}').
     - **Action**: Choose action `stop`.

   - **Verification Step for Case 1**:
     - Ensure the following before proceeding:
       1. Have you checked whether the '{self.env['target']}' is detected in the middle third of the image? 
       2. Have you accurately compared the distances of **all** '{self.env['target']}' against the defined stop distance ('{self.env['stop_hurdle_meter_for_target']}')?
       3. Based on these checks, confirm which subcase (1.1, 1.2, or 1.3) applies and proceed with the specified action.

#### Case 2: The '{self.env['target']}' is not detected in the `### Image Analysis` section.
   - **Subcase 2.1**: The '{self.env['object1']}' is **not detected** in the `### Image Analysis` section.
     - **Action**: Explore different orientations. Do not change your position (x, y).
       - **Note**: Avoid revisiting orientations that have already been explored at the same position without detecting the '{self.env['target']}' according to the `### History` section.

   - **Subcase 2.2**: The '{self.env['object1']}' **is detected** in the `### Image Analysis` section, and its distance is **more than** the stopping threshold ('{self.env['hurdle_meter_for_non_target']}').
     - **Action**: Adjust your position (x, y) vertically towards the detected '{self.env['object1']}'. Do not explore different orientations.

   - **Subcase 2.3**: The '{self.env['object1']}' **is detected** in the `### Image Analysis` section, and its distance is **less than** the stopping threshold ('{self.env['hurdle_meter_for_non_target']}').
     - **Action**: Explore different orientations. Do not change your position (x, y).
       - **Note**: Avoid revisiting orientations that have already been explored at the same position without detecting the '{self.env['target']}' according to the `### History` section.

   - **Decision Tree for Subcases 2.2 and 2.3**:
       1. Check if '{self.env['object1']}' is detected in the `### Image Analysis` section:
          - If it is not detected, this is **Subcase 2.1**.
       2. If '{self.env['object1']}' is detected:
          - Measure the distance to '{self.env['object1']}'.
          - If the distance is **more than** '{self.env['hurdle_meter_for_non_target']}', this is **Subcase 2.2**:
            - **Action**: Move your position vertically to get closer to the target by 'move forward' or 'move backward'.
          - If the distance is **less than** '{self.env['hurdle_meter_for_non_target']}', this is **Subcase 2.3**:
            - **Action**: Explore different orientations by turning right or left.

   - **Verification Step for Case 2**:
     - Ensure the following before proceeding:
       1. Have you checked whether the '{self.env['target']}' is **not detected** in the `### Image Analysis` section?
       2. Have you confirmed the presence or absence of '{self.env['object1']}' in the `### Image Analysis` section?
       3. If '{self.env['object1']}' is detected:
          - Have you measured the distance accurately against the stopping threshold ('{self.env['hurdle_meter_for_non_target']}')?
       4. Based on these checks, confirm which subcase (2.1, 2.2, or 2.3) applies and proceed with the specified action.

### Instructions for New tuple (x, y, orientation):
Position and orientation are represented by a tuple '(x, y, orientation)', where:
- x and y represent grid pixel index.
- orientation represents the facing direction in degrees.

Orientation determines all directional movements. Use the following orientation mappings:
- 0° or 360° (North): Facing the positive Y-axis.
- 90° or -270° (East): Facing the positive X-axis.
- 180° or -180° (South): Facing the negative Y-axis.
- 270° or -90° (West): Facing the negative X-axis.

Movement & Shift Table by Orientation:
The effect of each action on x and y pixel index depends on the orientation as shown:

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

stop: 
Position and orientation remain unchanged when the total of Move, Shift, and Turn equals 0.

Verification Step:
Confirm each x or y coordinate change reflects the intended movement or shift by double-checking against the table above to ensure consistency with the specified orientation.

### Instructions for Move: for deciding the number of steps of 'move forward' or 'move backward'
#### Case 1:
   - If the chosen action is 'stop' and the distance to the detected '{self.env['target']}' in the middle third of the image is less than the defined stop distance (i.e., '{self.env['stop_hurdle_meter_for_target']}'), execute 0.
   - If the chosen action is 'move forward' and the distance is between '{self.env['stop_hurdle_meter_for_target']}' and 1.70 meters, execute 1. 
   - If the chosen action is 'move forward' and the distance is between 1.70 meters and 2.3 meters, execute 2. 
   - Otherwise, execute 3.
#### Case 2:
   - If 'move forward' or 'move backward' is in the chosen actions:
     - If the chosen action is 'move forward' and the distance to '{self.env['object1']}' is between '{self.env['hurdle_meter_for_non_target']}' and 2.0 meters, execute 1.
     - If the chosen action is 'move forward' and the distance is between 2.0 meters and 2.5 meters, execute 2.
     - Otherwise, execute 3.
   - If 'move forward' or 'move backward' is not in the chosen actions, execute 0.

### Instructions for Shift: for deciding the number of steps of 'shift right' or 'shift left'
#### Case 1:
   - If the detected '{self.env['target']}' is within the middle third of the image's width, execute 0.
   - If the detected '{self.env['target']}' is not within the middle third of the image's width, execute 1.
#### Case 2: execute 0.

### Instructions for Turn: for deciding the number of steps of 'turn right' or 'turn left'
#### Case 1: execute 0.
#### Case 2: If 'turn right' or 'turn left' is in the chosen actions, execute 1; otherwise, execute 0.
""" 

        self.system_prompt_feedback = f"""
You are Go2, a robot dog assistant. You can only speak English regardless of the language the user uses. Your position and orientation are represented by a tuple (x, y, orientation), where:

- x and y are grid coordinates representing your position.
- orientation is the facing direction in degrees.

Your task is to search for the target object, '{self.env['target']}', starting at (0, 0, 0). You can only see objects in your facing direction and must adjust your orientation to face the target while searching.

### Instructions for Action:
Action dictionary:
- 'move forward'
- 'move backward'
- 'shift right' 
- 'shift left'
- 'turn right'
- 'turn left'
- 'stop'

Choose the precise action name from the action dictionary to search for the '{self.env['target']}' object based on conversation between you and the user. 
- If multiple different actions need to be executed based on the conversation, identify each unique action from the action dictionary and list them once, separated by commas. 
- For multiple same actions, list them only once. 

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
""" 

    def action_auto_format(self):
        prompt = f"""
Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format.

Current Tuple: compute '(x, y, orientation)' before you take any action at this round.
Target Status: If the target is detected in the 'Image Analysis' section, mark 'Visible'; otherwise, 'Invisible.'
Contextual Likelihood: If the target visibility status is 'Visible', set the likelihood as 100. If it is 'Invisible' and even '{self.env['object1']}' is not detected, set 0. If the target visibility status is 'Invisible' but '{self.env['object1']}' is detected, assign high contextual likelihood (0-100) because something to eat or drink and '{self.env['target']}' are commonly stored together.
Action: Follow the guideline in the '### Instructions for Action' section.
Move: Follow the guideline in the '### Instructions for Move' section.
Shift: Follow the guideline in the '### Instructions for Shift' section.
Turn: Follow the guideline in the '### Instructions for Turn' section.
New Tuple: Follow the guideline in the '### Instructions for New Tuple (sx, y, orientation)' section.
Reason: 
- Explain your choice of actions in one concise complete sentence. 
- If you give a high contextual likelihood, pinpoint one location where it is likely to be found among kitchen, living room, and office. You don't need to mention the case number. If the stopping hurdle meter for the '{self.env['object1']}' or '{self.env['target']}' is considered in your reasoning, you must mention it. 
- If you need to mention about whether the '{self.env['target']}' is in the left/middle/right third of the image, just say 'left', 'middle', or 'right' without mentioning the 'third'. 
- If you need to say something like 'No other objects detected', just say something by rephrasing the statement 'No other objects that are contextually related to the target detected'.
- If '{self.env['object1']}' is not detected in the '### Image Analysis' section, you must not mention anything about '{self.env['object1']}'. Even if the reasoing behind your action considers whether '{self.env['object1']}' is detected or not, you must not mention any about '{self.env['object1']}'. Even if '{self.env['object1']}' is detected at previous rounds in the '### History' section, you must not mention anything about '{self.env['object1']}' in your reasoning. You can mention about '{self.env['object1']}' in your reasoning only if '{self.env['object1']}' is detected in the '### Image Analysis' section, **not in the '### History' section.** 
"""
        return prompt
    
# Explain your choice of actions and mentioning which instructions affected your decision without mentioning the case number in one concise complete sentence.
# Tell the objects you see in the round and how you do think whether they are related to the target search. Also, explain why you did what you just did without referring to the instructions in the system prompt. Please answer concisely.
    
    def action_feedback_format(self):
        prompt = f"""
Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format.

Current Tuple: compute '(x, y, orientation)' before you take any action at this round.
Target Status: If the target is detected in the 'Image Analysis' section, mark 'Visible'; otherwise, 'Invisible.'
Contextual Likelihood: If the target visibility status is 'Visible', set the likelihood as 100. If it is 'Invisible' and even '{self.env['object1']}' is detected, set 0. If the target visibility status is 'Invisible' but something to eat or drink is detected, assign high contextual likelihood (0-100) because something to eat or drink and '{self.env['target']}' are commonly stored together.
Action: Follow the guideline in the '### Instructions for Action' section.
Move: Compute the number of steps of 'move forward' or 'move backward' based on the conversation between you and the user.
Shift: Compute the number of steps of 'shift right' or 'shift left' based on the conversation between you and the user.
Turn: Compute the number of steps of 'turn right' or 'turn left' based on the conversation between you and the user.
New Tuple: compute '(x, y, orientation)' after you take any action at this round.
Reason: Explain your choice of actions in one concise complete sentence.
"""
        return prompt

    def questions_feedback_format(self, user_input):
        if any(keyword in user_input.lower() for keyword in ("kitchen", "sink", "refrigerator", "banana", "bottle")):            
            prompt =  f"""Kindly inform them that you cannot recognize and locate the object the user (please call the user 'you') mentioned and request his/her help by providing an example prompt, such as 'turn right 2 times and then move forward 3 times,' while explaining that such guidance helps you locate objects more effectively. """
        elif "!" in user_input.lower():
            prompt = """Kindly respond in one concise sentence based on the conversation with the user (please call the user 'you'): it should be about appreciating the user's feedback first and then clearly say you are executing the action the user asked for.
            """
        else:
            prompt = """Kindly respond in two concise sentences based on the conversation with the user (please call the user 'you'): the first answers to the user's question or feedback, and the second explain why you think so."""
        
        return prompt

    def set_target(self, target):
        self.target = target

    def stt(self, voice_buffer):
        return None

    def get_response_by_feedback(self, feedback):
        pass

    def store_image(self, image_array = None):
        if image_array is None:
            text = "The user provided feedback; no image was captured."
            image = Image.new('RGB', (self.env["captured_width"], self.env["captured_height"]), 'black')
            image = utils.put_text_in_the_middle(image, text, self.env["captured_width"], self.env["captured_height"])
        else:
            image = utils.OpenCV2PIL(image_array)
 
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
    curr_tuple: str
    target: str
    likelihood: str
    action: list
    move: str
    shift: str
    turn: str
    new_tuple: str
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
            curr_tuple, target, likelihood, action, move, shift, turn, new_tuple, reason = parts
            
            # parse action
            action = ResponseMessage.parse_action(action)

            # parse numeric values
            move = ResponseMessage.parse_step(move)
            shift = ResponseMessage.parse_step(shift)
            turn = ResponseMessage.parse_step(turn)
            
            total_step = move + shift + turn
            if total_step == 0:
                action = "stop"
            
            print(f"action: {action}, move: {move}, shift: {shift}, turn: {turn}")

        except Exception as e:
            print(f"Parse failed. Message: {message}\nError: {e}")
            # Return default values when parsing fails
            return ResponseMessage(
                curr_tuple="(0, 0, 0)",
                target="unknown",
                likelihood="0",
                action=["stop"],
                move="0",
                shift="0",
                turn="0",
                new_tuple="(0, 0, 0)",
                reason="Parse error"
            )
        return ResponseMessage(curr_tuple, target, likelihood, action, move, shift, turn, new_tuple, reason)
    
    def to_dict(self):
        return asdict(self)
