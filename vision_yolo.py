import math
from dataclasses import dataclass

import cv2
from ultralytics import YOLO
from transformers import pipeline
import torch
from PIL import Image
import numpy as np

@dataclass
class YOLOResponse:
    frame: "cv2.typing.MatLike"
    description: str

class YOLODepthModel:
    def __init__(self, env, yolo_model_path="models/yolo-Weights/yolov10n.pt", depth_model_checkpoint="Intel/zoedepth-nyu-kitti"):
        """
        Args:
            depth_model_checkpoint
            - Absolute Depth: Use the checkpoint `"Intel/zoedepth-nyu-kitti"` for models that return depth in real-world units (e.g., meters).
            - Relative Depth: Use the checkpoint `"depth-anything/Depth-Anything-V2-base-hf"` for models that return normalized relative depth values.
        """
        self.env = env

        # YOLO model for object detection
        self.yolo_model = YOLO(yolo_model_path)
        
        # Define YOLO object classes
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
                           "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
                           "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
                           "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        # Depth Estimation pipeline
        self.depth_pipe = pipeline("depth-estimation", model=depth_model_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

    def detect_objects(self, frame):
        # Detect objects and return the class names and bounding boxes
        results = self.yolo_model(frame, stream=True)
        detected_classes = []
        bounding_boxes = []
        height, width, _ = frame.shape
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0] * 100
                if confidence >= self.env["detect_confidence"]:  # Include only class names with a confidence level greater than the specified threshold
                    detected_classes.append(box)
                    
                    # Extract and clamp bounding box coordinates within the image dimensions
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1, x2, y2 = max(0, min(x1, width-1)), max(0, min(y1, height-1)), max(0, min(x2, width-1)), max(0, min(y2, height-1))
                    
                    bounding_boxes.append([x1, y1, x2, y2])
        
        return detected_classes, bounding_boxes

    # def draw_on_frame(self, frame):
    #     # Get detected objects and their bounding boxes using detect_objects method
    #     detected_class_names, bounding_boxes = self.detect_objects(frame)
        
    #     for i, box in enumerate(bounding_boxes):
    #         x1, y1, x2, y2 = box
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
    #         # Assuming the same confidence logic as detect_objects
    #         confidence = math.ceil(self.env["detect_confidence"]) / 100  # Use a fixed confidence or retrieve it from detect_objects if needed
    #         cls = detected_class_names[i]
    #         org = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(frame, f"{cls}: {confidence}", org, font, 1, (255, 0, 0), 2)

    #     return frame

    def depth_estimation(self, frame, bounding_boxes):
        # Perform depth estimation and return the average depth for each bounding box
        # Convert the OpenCV frame (BGR format NumPy array class w/ dtype uint8) to an RGB PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get depth estimation predictions
        predictions = self.depth_pipe(image) # input image should be in a file path, a URL, a base64 string, or a PIL.Image object
        
        # Extract the predicted (absolute/relative) depth as a PyTorch tensor
        depth_values_tensor = predictions["predicted_depth"]
        # depth_map = predictions["depth"] # PIL image
        
        # Move tensor to CPU (if it's on GPU) and convert it to a NumPy array
        depth_values_np = depth_values_tensor.cpu().numpy()

        # Get the dimensions of the depth map
        depth_height, depth_width = depth_values_np.shape[1:]

        # Get the dimensions of the original frame
        frame_height, frame_width, _ = frame.shape

        # Calculate the scaling factors for the coordinates
        scale_x = depth_width / frame_width
        scale_y = depth_height / frame_height

        average_depths = []  # To store average depth values for each object

        # Loop through each bounding box and calculate the average depth
        for box in bounding_boxes:
            x1, y1, x2, y2 = box

            # Scale the bounding box coordinates to match the depth map size
            x1_depth = int(x1 * scale_x)
            y1_depth = int(y1 * scale_y)
            x2_depth = int(x2 * scale_x)
            y2_depth = int(y2 * scale_y)

            # Extract the depth values within the bounding box
            box_depth_values = depth_values_np[0, y1_depth:y2_depth, x1_depth:x2_depth]

            # Calculate the average depth for the bounding box
            if box_depth_values.size > 0:
                average_depth = np.mean(box_depth_values)
            else:
                average_depth = 0  # If the bounding box is empty, return 0 depth

            average_depths.append(average_depth)

        return average_depths

    def describe_image(self, frame, draw_on_frame=True):
        # Get detected objects and their bounding boxes using detect_objects method
        detected_classes, bounding_boxes = self.detect_objects(frame)
        # Get the average depth for each bounding box
        average_depths = self.depth_estimation(frame, bounding_boxes)

        # Generate a description of the image based on detected objects and their depths
        description = []
        
        for i, detected in enumerate(detected_classes):
            x1, y1, x2, y2 = bounding_boxes[i]
            
            if average_depths[i] <= self.env["depth_threshold"]:
                avg_depth = average_depths[i] * eval(self.env["depth_scale_for_under"])
            else:
                avg_depth = average_depths[i] * eval(self.env["depth_scale_for_over"])

            # Calculate the center coordinates of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Generate a simple sentence describing each object using center coordinates
            description.append(f"You detected {self.classNames[int(detected_classes[i].cls[0])]} at coordinates ({center_x:.0f}, {center_y:.0f}) with a depth of {avg_depth:.2f} meters.")
        
        if draw_on_frame:
            for i, box in enumerate(bounding_boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)                
                org = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"{self.classNames[int(detected_classes[i].cls[0])]}: {detected_classes[i].conf[0]:.2f}", org, font, 1, (255, 0, 0), 2)

        # Join all the descriptions into a single paragraph
        return YOLOResponse(frame, "\n".join(description))