from dataclasses import dataclass, asdict
import utils
from PIL import Image
import os

class AiClientBase:
#### Use w/ gpt_vsion_test() ####        
    # system_prompt = """You are Go2, a robot dog."""
    system_prompt = """You are Go2, a robot dog. Your position and orientation are represented by a tuple (x, y, orientation), where x and y are its coordinates on a grid, and orientation describes its facing direction in degrees:
0 degrees or 360 degrees means facing north (along the positive Y-axis)
90 degrees or -270 degrees means facing east (along the positive X-axis)
180 degrees or -180 degrees means facing south (along the negative Y-axis)
270 degrees or -90 degrees means facing west (along the negative X-axis).
        
You start at (0, 0, 0). Your task is to search for target objects as instructed. One unit shift in the X and Y directions corresponds to 0.5 meters. Here is the action dictionary formatted as 'Action: (X shift, Y shift, clockwise rotation)':

Stop: (0, 0, 0)
Pause: (0, 0, 0)
Move forward: (0, 1, 0) — Moves 1 meter forward
Move backward: (0, -0.5, 0) — Moves 0.5 meter backward
Shift right: (1, 0, 0) — Moves 1 meter to the right
Shift left: (-1, 0, 0) — Moves 1 meter to the left
Turn right: (0, 0, 90) — Rotates 90 degrees clockwise
Turn left: (0, 0, -90) — Rotates 90 degrees counterclockwise"""

    def __init__(self, env):
        self.client = None
        self.env = env
        self.image_counter = 0

    def get_user_prompt(self):
#### Use w/ gpt_vsion_test() ####        
#         return f"""Go2, find {self.target}. Respond with the specified format:
# Go2)
# Target: Please assess whether the target is visible in the captured image. If the target object is detected, mark it as 'Visible'. If the target is not detected, mark it as 'Invisible'.
# Confidence: If visible, provide how much you are sure that the detected object is the target based on the scale 0-100. Please output only the number.
# Location: If visible, explain its location in the image in one concise short sentence.
# """
        return f"""Go2, find {self.target}. Respond with the specified format:
Go2)
Current Position: The tuple (x, y, orientation) before doing Action.
Target: If the {self.target} is detected in the image description, mark it as 'Visible'. If the target is not detected, mark it as 'Invisible'.
Likelihood: If the target object is detected, assign a likelihood of 100. If the target is not detected, assess the likelihood of its presence on a scale of 0 to 100, using contextual and semantic information from the detected objects in the household setting, and provide a one-sentence rationale for your evaluation.
Action: action name you choose.
New Position: The updated tuple (x, y, orientation) after doing Action.
Reason: Why you choose the action.

# Instruction
- Use the image description below to make a decision.
- If the target is not visible in the image, explore another orientation.
- If there is a feedback, use the feedback to determine your action."""

    def set_target(self, target):
        self.target = target

    def stt(self, voice_buffer):
        return None

    def get_response_by_image(self, cv2_image):
        pass

    def get_response_by_feedback(self, feedback):
        pass

    def store_image(self, cv2_image = None):
        if cv2_image is None:
            image = Image.new('RGB', (self.env["captured_width"], self.env["captured_height"]), 'black')
        else:
            image = utils.OpenCV2PIL(cv2_image)
 
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

    @staticmethod
    def parse(message: str):
        try:
            # Filter out lines that do not contain ':' and strip empty spaces
            parts = [line.split(":", 1)[1].strip() for line in message.split('\n') if ':' in line and len(line.strip()) > 0]
            if len(parts) != 6:
                raise ValueError("Message does not contain exactly four parts")
            curr_position, target, likelihood, action, new_position, reason = parts
        except Exception as e:
            print("parse failed. Message: ", message, "\nError: ", e)
            return ResponseMessage()
        return ResponseMessage(curr_position, target, likelihood, action, new_position, reason)
    
    def to_dict(self):
        return asdict(self)
