import math
from dataclasses import dataclass
import datetime

import cv2
import torch
from transformers import pipeline
from torchvision.ops import nms
from PIL import Image, ImageDraw
import numpy as np
import os

import utils

@dataclass
class OWLResponse:
    frame: "cv2.typing.MatLike"
    description: str

class OWLDepthModel:
    def __init__(self, env, owl_model_checkpoint="google/owlv2-base-patch16-ensemble", depth_model_checkpoint="Intel/zoedepth-nyu-kitti"):
        """
        Args:
            depth_model_checkpoint
            - Absolute Depth: Use the checkpoint `"Intel/zoedepth-nyu-kitti"` for models that return depth in real-world units (e.g., meters).
            - Relative Depth: Use the checkpoint `"depth-anything/Depth-Anything-V2-base-hf"` for models that return normalized relative depth values.
        """
        self.env = env
        self.image_counter = 0

        torch.cuda.empty_cache()  # Clear the cache before model initialization

        # Initialize the OWL-V2 model for object detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = pipeline(model=owl_model_checkpoint, task="zero-shot-object-detection", device=0 if torch.cuda.is_available() else -1)

        # Define custom candidate labels for object detection
        self.candidate_labels = ["apple"] # "tv", "potted plant", "coffee machine", "block", "table", "person", "chair", "plant", "bottle", "person"
        # self.candidate_labels = [self.env["target_in_test_dataset"]], 

        # Depth Estimation pipeline
        self.depth_pipe = pipeline("depth-estimation", model=depth_model_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

        # try:
        #     os.makedirs('test', exist_ok=True)
        #     self.save_dir = f"test/test_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        #     os.mkdir(self.save_dir)
        # except Exception as e:
        #     print(f"Failed to create directory: {e}")

    def detect_objects(self, frame):
        # Convert OpenCV frame to PIL image
        image = Image.fromarray(np.uint8(frame)).convert("RGB")

        # Perform object detection using the OWL-V2 model (handled by the pipeline)
        predictions = self.detector(image, candidate_labels=self.candidate_labels)

        boxes = []
        scores = []
        labels = []

        # Collect bounding boxes, scores, and labels
        for prediction in predictions:
            box = prediction["box"]
            score = prediction["score"]
            label = prediction["label"]

            # Add the box coordinates, score, and label
            boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            scores.append(score)
            labels.append(label)

        if len(boxes) == 0:
            return [], [], []  # Return empty if no boxes are detected

        # Convert lists to tensors and move them to the GPU for NMS
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(self.device)
        scores_tensor = torch.tensor(scores, dtype=torch.float32).to(self.device)

        # Ensure tensors are 2D (even if there's only one box)
        if boxes_tensor.ndim == 1:
            boxes_tensor = boxes_tensor.unsqueeze(0)

        # Apply NMS - this will also be performed on the GPU
        keep_indices = nms(boxes_tensor, scores_tensor, self.env["iou_threshold"])

        # Prepare lists to store the final output after NMS
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        # Process the kept indices and clamp the bounding boxes to image dimensions (done on CPU)
        height, width = frame.shape[:2]
        for i in keep_indices:
            if scores[i] >= self.env["detect_confidence"]:
                filtered_labels.append(labels[i])
                filtered_scores.append(scores[i])  # No need to call .cpu().item() since scores[i] is already a float

                # Extract and clamp bounding box coordinates to stay within image bounds
                x1, y1, x2, y2 = boxes_tensor[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Ensure the box coordinates are within valid image dimensions
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                filtered_boxes.append([x1, y1, x2, y2])

        return filtered_labels, filtered_boxes, filtered_scores


    def depth_estimation(self, frame, boxes):
        # Convert the OpenCV frame to a PIL image for depth estimation
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get depth estimation predictions (this returns a tensor on the GPU if available)
        predictions = self.depth_pipe(image)

        # Extract depth map as a GPU tensor
        depth_values_gpu = predictions["predicted_depth"]

        # Get dimensions of depth map and frame
        depth_height, depth_width = depth_values_gpu.shape[-2:]  # Assuming depth tensor is in [batch, height, width] format
        frame_height, frame_width = frame.shape[:2]

        # Calculate scaling factors on the CPU
        scale_x = depth_width / frame_width
        scale_y = depth_height / frame_height

        average_depths = []

        # Calculate the average depth for each bounding box
        for box in boxes:
            # Scale bounding box to match depth map size (done on CPU for simplicity)
            x1_depth = int(box[0] * scale_x)
            y1_depth = int(box[1] * scale_y)
            x2_depth = int(box[2] * scale_x)
            y2_depth = int(box[3] * scale_y)

            # Ensure the indices are within valid range (done on CPU)
            x1_depth = max(0, min(x1_depth, depth_width - 1))
            y1_depth = max(0, min(y1_depth, depth_height - 1))
            x2_depth = max(0, min(x2_depth, depth_width - 1))
            y2_depth = max(0, min(y2_depth, depth_height - 1))

            # Extract depth values within the bounding box (on GPU)
            box_depth_values = depth_values_gpu[0, y1_depth:y2_depth, x1_depth:x2_depth]

            # Calculate average depth on the GPU
            if box_depth_values.numel() > 0:
                average_depth = box_depth_values.mean().item()  # Compute mean on the GPU and retrieve the scalar value
            else:
                average_depth = 0
            average_depths.append(average_depth)

        return average_depths

    def get_label(self, frame):
        labels, boxes, scores = self.detect_objects(frame)
        return labels[0]

    def describe_image(self, frame, draw_on_frame=True):
        # Get detected objects and their bounding boxes
        labels, boxes, scores = self.detect_objects(frame)
        
        # Get depth for each bounding box (already using GPU in depth_estimation)
        average_depths = self.depth_estimation(frame, boxes)

        description = []
        depth_threshold = self.env["depth_threshold"]

        # Process each detected object on the CPU (text formatting, logic is more efficient on CPU)
        for i, label in enumerate(labels):
            x1, y1, x2, y2 = boxes[i]

            # Adjust depth using environment thresholds (on CPU)
            avg_depth = average_depths[i] * (
                eval(self.env["depth_scale_for_under"]) if average_depths[i] <= depth_threshold
                else eval(self.env["depth_scale_for_over"])
            )

            # Calculate the center of the bounding box (on CPU, simple arithmetic)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Generate description for each detected object (on CPU)
            description.append(f"You detected {label} at coordinates ({center_x:.0f}, {center_y:.0f}) with a depth of {avg_depth:.2f} meters.")

        # Draw bounding boxes and labels on the frame if draw_on_frame is True (done on CPU using OpenCV)
        if draw_on_frame:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                org = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, f"{labels[i]}: {average_depths[i]:.1f}m", org, font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"{labels[i]}: {scores[i]:.1f}", org, font, 0.5, (255, 255, 255), 1)

        print(torch.cuda.is_available())
        # Return OWLResponse with the frame and the combined description
        return OWLResponse(frame, "\n".join(description))
    
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
