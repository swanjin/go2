# ai_cllient_base.py
from dataclasses import dataclass, field, asdict
import datetime
from PIL import Image
import os
import re

import utils
from navi_config import NaviConfig
from navigation import Mapping, NaviModel


class AiClientBase:
    def __init__(self, env):
        self.client = None
        self.env = env
        self.image_counter = 0
        self.is_first_response = True
        self.border_points = Mapping().generate_border_points(NaviConfig.border_size)  # Assuming border_size is 7

    def initial_prompt(self, curr_state):
        return (f"""
        You are Go2, a robot dog assistant. You can only speak English regardless of the language the user uses. Your position and orientation are represented by a state (x, y, orientation), where:

        - x and y are grid coordinates representing your position.
        - orientation is the facing direction in degrees.

        Your task is to search for the target object, {self.env['target']}. Current state is {curr_state}. You can only see objects in your facing direction and must adjust your orientation to face the target while searching.

        You should avoid obstacles.
        {self.get_obstacles()}
        """)
    
    def get_action_dictionary(self):
        return ("""
        Action dictionary:
        - 'move forward'
        - 'move backward'
        - 'shift right' 
        - 'shift left'
        - 'turn right'
        - 'turn left'
        - 'stop'
        
        Choose the precise action name from the action dictionary to search for the {self.env['target']}. If the action needs to be executed several times, identify each unique action from the action dictionary and list them several times, separated by commas.
        """)
    
    def get_new_state(self):
        return ("""
        Orientation determines all directional movements. Use the following orientation mappings:
        - 0° or 360° (North): Facing the positive Y-axis.
        - 90° or -270° (East): Facing the positive X-axis.
        - 180° or -180° (South): Facing the negative Y-axis.
        - 270° or -90° (West): Facing the negative X-axis.

        The effect of executing each action once on x and y grid coordinates depends on the orientation as shown:

        | Orientation | move forward | move backward | shift right | shift left |
        |-------------|--------------|---------------|-------------|------------|
        | 0° (North)  | (x, y + 1)   | (x, y - 1)    | (x + 1, y)  | (x - 1, y) |
        | 90° (East)  | (x + 1, y)   | (x - 1, y)    | (x, y - 1)  | (x, y + 1) |
        | 180° (South)| (x, y - 1)   | (x, y + 1)    | (x - 1, y)  | (x + 1, y) |
        | 270° (West) | (x - 1, y)   | (x + 1, y)    | (x, y + 1)  | (x, y - 1) |

        The effect of executing turn right or turn left once (Orientation Changes Only):
        - turn right: Increases orientation by 90°.
        - turn left: Decreases orientation by 90°.
        After each turn, normalize the orientation to a range of 0° to 360° (e.g., -90° becomes 270°).

        The effect of executing stop:
        Position and orientation remain unchanged.

        Verification Step:
        - If the action needs to be executed several times, the new state must be updated for each action.
        - Confirm each x or y coordinate change reflects the intended movement or shift by double-checking against the table above to ensure consistency with the specified orientation.
        """)
    
    def get_landmarks(self):
        landmarks_str = ",\n".join([f'"{name}": {coords}' for name, coords in NaviConfig.landmarks.items()])
        return f"Landmarks:\n{landmarks_str}\n"
    
    def get_obstacles(self):
        border_lines = ", ".join([f"({x}, {y})" for x, y in self.border_points])
        
        # Dynamically fetch all obstacles from NaviConfig
        obstacles = ", ".join([f"({coords[0]}, {coords[1]})" for coords in NaviConfig.obstacles.values()])
        
        return (f"""
        Obstacles:
        1. Border lines
        {border_lines}
        2. Box
        {obstacles}
        """)

    def prompt_auto(self, curr_state):
        stop_target = self.env.get('stop_target')
        threshold_range = self.env.get('threshold_range')

        if stop_target is None or threshold_range is None:
            raise ValueError("Environment variables 'stop_target' and 'threshold_range' must be set.")

        return (f"""
        {self.initial_prompt(curr_state)}

        ### Instructions for Action:
        {self.get_action_dictionary()}

        #### Case 1: The target object {self.env['target']} is detected in the 'Detection' section.
        - **Subcase 1.1**: The {self.env['target']} is in the middle of the frame, and the distance to the {self.env['target']} is greater than '{stop_target}'.
            - **Action**: Move your position vertically to get closer to the target with the number of times as below.
                - If the chosen action is 'move forward' and the distance to the target is within ({stop_target}, {stop_target + threshold_range}) meters, execute 1 times.
                - Otherwise, execute 2 times.

        - **Subcase 1.2**: The {self.env['target']} is in the middle of the frame, and the distance to the {self.env['target']} is less than '{stop_target}'.
            - **Action**: 'stop'.

        - **Subcase 1.3**: The {self.env['target']} is on the left side of the frame.
            - **Action**: 'shift left'.

        - **Subcase 1.4**: The {self.env['target']} is on the right side of the frame.
            - **Action**: 'shift right'.

        - **Verification Step for Case 1**:
            - Ensure the following before proceeding:
            1. Have you checked whether the {self.env['target']} is detected in the middle of the frame?
            2. Have you accurately compared the distances of  {self.env['target']} against the defined stop distance ('{stop_target}')?
            3. Have you checked whether the {self.env['target']} is on the left side of the frame?
            4. Have you checked whether the {self.env['target']} is on the right side of the frame?
            5. Based on these checks, confirm which subcase (1.1, 1.2, 1.3, or 1.4) applies and proceed with the specified action.

        #### Case 2: The {self.env['target']} is **not detected** in the 'Detection' section.
        - **Subcase 2.1**: Neither {self.env['object1']} nor {self.env['object2']} nor {self.env['object3']} is detected in the 'Detection' section.
            - **Action**: Rotate once to explore a different orientation without changing your position (x, y). Avoid action that would update your state (x, y, orientation) as the same state that you have already visited according to the 'Memory' section.

        - **Subcase 2.2**: Either {self.env['object1']} or {self.env['object2']} or {self.env['object3']} is detected in the 'Detection' section, and its distance is **more than** the stopping threshold ('{self.env['stop_landmark']}').
            - **Action**: Move forward once or twice.
            - Strictly follow the following rules: 
                - If the distance to the detected object is between '{self.env['stop_landmark']}' and '{(self.env['stop_landmark']+self.env['threshold_range'])}' meters, move forward **once**. 
                - If the distance to the detected object is greater than '{(self.env['stop_landmark']+self.env['threshold_range'])}' meters, move forward **twice**.

        - **Subcase 2.3**: Either {self.env['object1']} or {self.env['object2']} or {self.env['object3']} is detected in the 'Detection' section, and its distance is **less than** the stopping threshold ('{self.env['stop_landmark']}').
            - **Action**: Rotate once to explore a different orientation without changing your position (x, y). Avoid action that would update your state (x, y, orientation) as the same state that you have already visited according to the 'Memory' section.
        
        - **Verification Step for Case 2**:
            - Ensure the following before proceeding:
            1. Have you checked whether the {self.env['target']} is **not detected** in the 'Detection' section?
            2. Have you confirmed the presence or absence of {self.env['object1']} or {self.env['object2']} or {self.env['object3']} in the 'Detection' section?
            3. If either {self.env['object1']} or {self.env['object2']} or {self.env['object3']} is detected:
                - Have you measured the distance accurately against the stopping threshold ('{self.env['stop_landmark']}')?
            4. Based on these checks, confirm which subcase (2.1, 2.2, or 2.3) applies and proceed with the specified action.

        ### Instructions for New state:
        {self.get_new_state()}
        """)

    def response_format_auto(self):
        return (f"""
        Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format.

        Response Format:
        - **Initial State**: (x, y, orientation)
        - **New State**: (x, y, orientation)
        - **Action**: action1, action2, ...
        - **Reason**: 
          - Explain your choice of actions in one concise sentence by pinpointing the everyday association between the detected objects and the {self.env['target']} as a contextual reasoning.
          - Do not mention case numbers, subcase numbers, section names or distances.
          - If referring to the {self.env['target']} position in the image, use 'left', 'middle', or 'right' without mentioning 'third'.
          - If neither {self.env['object1']}, {self.env['object2']}, nor {self.env['object3']} is detected, do not mention them in your reasoning.
        """)

    def prompt_landmark_or_non_command(self, curr_state):
        return (f"""
        You are Go2, a robot dog assistant who only speaks English. Your task is to search for the target object, {self.env['target']}. Current state is {curr_state}. You can only see objects in your facing direction. You can only see objects in your facing direction.

        State: (x, y, orientation)
        - Grid x: -7 to 7
        - Grid y: -9 to 7
        - Orientation: 0°=N, 90°=E, 180°=S, 270°=W
        - Current state: {curr_state}

        {self.get_landmarks()}\n\n
        {self.get_obstacles()}
        """)

    def response_format_landmark_command(self): # landmark command: go close to the refrigerator
        return (f"""        
        Rules:
        1. Target state within grid bounds, not an obstacle.
        2. If target state based on a landmark/obstacle, set orientation to its orientation.
        3. If target state invalid, find nearest valid spot.
        4. If ties in distance, pick randomly.
        
        Response Format: 
        - Respond only with "(x,y,orientation)" without extra text.
        """)

    def response_format_non_command(self): # non-command: what can you see?
        return (f"""
        Rules:
        - If the user asks a question, one sentence should provide a concise answer to the user's question.
        - Do not mention numbers and cardinal directions for states, obstacles, or landmarks. 
        - Describe landmarks relative to your current state only if the user asks for it.
        """)

    def prompt_general_command(self, curr_state):
        return (f"""
        {self.initial_prompt(curr_state)}

        ### Instructions for Action/New state:
        {self.get_action_dictionary()}\n\n
        {self.get_new_state()}
        """)

    def response_format_general_command(self): # general command: move forward 3 times/ turn around
        return (f"""
        Ensure each response follows the following format precisely. Do not deviate. Before responding, verify that your output exactly matches the structured format. 

        - **Initial State**: (x, y, orientation)
        - **New State**: (x, y, orientation)
        - **Action**: action1, action2, ... (If the user requests a precise command requiring multiple actions to be executed several times, identify each unique action from the action dictionary and list them accordingly, repeating them as needed and separating them with commas.)
        - **Reason**: 
          - Explain your choice of actions in one concise sentence.
        """)

    def set_target(self, target):
        self.target = target

    def stt(self, voice_buffer):
        return None

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
class ResponseMsg:
    initial_state: tuple
    new_state: tuple
    action: list
    reason: str

    @staticmethod
    def parse(message: str):
        try:
            # Filter out lines that do not contain ':' and strip empty spaces
            parts = [line.split(":", 1)[1].strip() for line in message.split('\n') if ':' in line and len(line.strip()) > 0]
            if len(parts) != 4:
                raise ValueError("Message does not contain exactly four parts")
            initial_state, new_state, action, reason = parts
            
            # Convert action string to list
            actions = utils.string_to_list(action)

            # Convert state strings to tuples
            initial_state = utils.string_to_tuple(initial_state)
            new_state = utils.string_to_tuple(new_state)

        except Exception as e:
            print(f"Parse failed. Message: {message}\nError: {e}")
            # Return default values when parsing fails
            return ResponseMsg(
                initial_state=(0, 0, 0),
                new_state=(0, 0, 0),
                action=["stop"],
                reason="Parse error"
            )
        return ResponseMsg(initial_state, new_state, actions, reason)