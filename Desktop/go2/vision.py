# vision.py
import math
from dataclasses import dataclass
import datetime
import warnings
import os

import cv2
import torch
from transformers import pipeline
from torchvision.ops import nms
from PIL import Image, ImageDraw
import numpy as np

import utils

# Conditional import of LangSAM based on env configuration
def import_langsam():
    from lang_sam import LangSAM
    return LangSAM

@dataclass
class VisionResponse:
    frame: "cv2.typing.MatLike"
    detected_objects: list
    distances: list
    description: list

class VisionModel:
    def __init__(self, env, depth_model_checkpoint="Intel/zoedepth-nyu-kitti"):
        """
        Args:
            depth_model_checkpoint
            - Absolute Depth: Use the checkpoint `"Intel/zoedepth-nyu-kitti"` for models that return depth in real-world units (e.g., meters).
            - Relative Depth: Use the checkpoint `"depth-anything/Depth-Anything-V2-base-hf"` for models that return normalized relative depth values.
        """
        self.env = env
        self.image_counter = 0

        # Initialize LangSAM if enabled in env
        if self.env.get("langsam", False):
            self.LangSAM = import_langsam()
        else:
            self.LangSAM = None

        # Define custom candidate labels for object detection
        #self.candidate_labels = ["apple"] # "tv", "potted plant", "coffee machine", "block", "table", "person", "chair", "plant", "bottle", "person"
        self.candidate_labels = self.env["target_in_test_dataset"]

        # Depth Estimation pipeline
        self.depth_pipe = pipeline("depth-estimation", model=depth_model_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

    def predict_langsam(self, image_pil):
        warnings.filterwarnings("ignore")

        # image_pil = Image.fromarray(np.uint8(frame)).convert("RGB")

        model = self.LangSAM()
        # caption = " ".join(self.candidate_labels)  # Join list into a single string
        boxes_tensor, logits_tensor, phrases = model.predict_dino(image_pil, self.candidate_labels, box_threshold=0.4, text_threshold=0.4)
        boxes = boxes_tensor.tolist()
        logits =logits_tensor.tolist()
        return phrases, boxes, logits

    def predict_owlv2(self, image_pil):
        torch.cuda.empty_cache()  # Clear the cache before model initialization

        # Initialize the OWL-V2 model for object detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection", device=0 if torch.cuda.is_available() else -1)
        
        # Perform object detection using the OWL-V2 model (handled by the pipeline)
        predictions = self.detector(image_pil, self.candidate_labels)
        
        boxes = []
        scores = []
        labels = []

        # Collect bounding boxes, scores, and labels
        for prediction in predictions:
            box = prediction["box"]
            score = prediction["score"]
            label = prediction["label"]

            # Add the box pixel index, score, and label
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
        keep_indices = nms(boxes_tensor, scores_tensor, self.env["iou_threshold_owlv2"])

        # Prepare lists to store the final output after NMS
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        # Process the kept indices and clamp the bounding boxes to image dimensions (done on CPU)
        image_array = utils.PIL2OpenCV(image_pil)
        height, width = image_array.shape[:2]
        for i in keep_indices:
            if scores[i] >= self.env["detect_confidence"]:
                filtered_labels.append(labels[i])
                filtered_scores.append(scores[i])  # No need to call .cpu().item() since scores[i] is already a float

                # Extract and clamp bounding box pixel index to stay within image bounds
                x1, y1, x2, y2 = boxes_tensor[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Ensure the box pixel index are within valid image dimensions
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                filtered_boxes.append([x1, y1, x2, y2])

        return filtered_labels, filtered_boxes, filtered_scores

    def detect_objects(self, image_pil):
        if self.env["detection_model"] == "langsam":
            if not self.env.get("langsam", False):
                print("LangSAM is not enabled in env.yml.")
                return [], [], []
            else: labels, boxes, scores = self.predict_langsam(image_pil)
        elif self.env["detection_model"] == "owlv2":
            labels, boxes, scores = self.predict_owlv2(image_pil)
        else:
            print("Detection model is not set in env.yml")
            return [], [], []  # Return empty values if the detection model is not set

        return labels, boxes, scores  # Ensure the method returns these values

    def depth_estimation(self, image_pil, boxes):
        image_array = utils.PIL2OpenCV(image_pil)

        # Get depth estimation predictions
        predictions = self.depth_pipe(image_pil)
        depth_values_gpu = predictions["predicted_depth"]

        # Get dimensions of depth map and frame
        depth_height, depth_width = depth_values_gpu.shape[-2:]
        frame_height, frame_width = image_array.shape[:2]

        # Calculate scaling factors
        scale_x = depth_width / frame_width
        scale_y = depth_height / frame_height

        center_depths = []

        # Calculate the depth for the center pixel of each bounding box
        for box in boxes:
            # Calculate center coordinates of the bounding box
            center_x = int(((box[0] + box[2]) / 2) * scale_x)
            center_y = int(((box[1] + box[3]) / 2) * scale_y)

            # Ensure the indices are within valid range
            center_x = max(0, min(center_x, depth_width - 1))
            center_y = max(0, min(center_y, depth_height - 1))

            # Get depth value at the center pixel
            center_depth = depth_values_gpu[0, center_y, center_x].item()
            center_depths.append(center_depth)

        return center_depths
        
        
        
        # image_array = utils.PIL2OpenCV(image_pil)

        # # Get depth estimation predictions (this returns a tensor on the GPU if available)
        # predictions = self.depth_pipe(image_pil)

        # # Extract depth map as a GPU tensor
        # depth_values_gpu = predictions["predicted_depth"]

        # # Get dimensions of depth map and frame
        # depth_height, depth_width = depth_values_gpu.shape[-2:]  # Assuming depth tensor is in [batch, height, width] format
        # frame_height, frame_width = image_array.shape[:2]

        # # Calculate scaling factors on the CPU
        # scale_x = depth_width / frame_width
        # scale_y = depth_height / frame_height

        # average_depths = []

        # # Calculate the average depth for each bounding box
        # for box in boxes:
        #     # Scale bounding box to match depth map size (done on CPU for simplicity)
        #     x1_depth = int(box[0] * scale_x)
        #     y1_depth = int(box[1] * scale_y)
        #     x2_depth = int(box[2] * scale_x)
        #     y2_depth = int(box[3] * scale_y)

        #     # Ensure the indices are within valid range (done on CPU)
        #     x1_depth = max(0, min(x1_depth, depth_width - 1))
        #     y1_depth = max(0, min(y1_depth, depth_height - 1))
        #     x2_depth = max(0, min(x2_depth, depth_width - 1))
        #     y2_depth = max(0, min(y2_depth, depth_height - 1))

        #     # Extract depth values within the bounding box (on GPU)
        #     box_depth_values = depth_values_gpu[0, y1_depth:y2_depth, x1_depth:x2_depth]

        #     # Calculate average depth on the GPU
        #     if box_depth_values.numel() > 0:
        #         average_depth = box_depth_values.mean().item()  # Compute mean on the GPU and retrieve the scalar value
        #     else:
        #         average_depth = 0
        #     average_depths.append(average_depth)

        # return average_depths

    def get_label(self, image_pil):
        labels, boxes, scores = self.detect_objects(image_pil)
        return labels[0]

    def describe_image(self, image_pil, draw_on_frame=True):
        image_array = utils.PIL2OpenCV(image_pil)

        # Get detected objects and their bounding boxes
        labels, boxes, scores = self.detect_objects(image_pil)
        
        # Get depth for each bounding box (already using GPU in depth_estimation)
        center_depths = self.depth_estimation(image_pil, boxes)

        detected_objects = []
        distances = []
        description = []
        depth_threshold = self.env["depth_threshold"]

        # Process each detected object on the CPU (text formatting, logic is more efficient on CPU)
        for i, label in enumerate(labels):
            x1, y1, x2, y2 = boxes[i]

            # Adjust depth using environment thresholds (on CPU)
            avg_depth = center_depths[i] * (
                eval(self.env["depth_scale_for_under"]) if center_depths[i] <= depth_threshold
                else eval(self.env["depth_scale_for_over"])
            )
            rounded_depth = round(avg_depth, 1)
            # Calculate the center of the bounding box (on CPU, simple arithmetic)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Generate description for each detected object (on CPU)
            detected_objects.append(f"{label}")
            distances.append(rounded_depth)
            description.append(f"You detected {label} at pixel index ({center_x:.0f}, {center_y:.0f}) with a distance of {rounded_depth} meters.")

        # Draw bounding boxes and labels on the frame if draw_on_frame is True (done on CPU using OpenCV)
        if draw_on_frame:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(image_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                org = (int(x1), int(y1) - 10 if y1 - 10 > 10 else int(y1) + 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, f"{labels[i]}: {average_depths[i]:.1f}m", org, font, 0.5, (0, 0, 255), 1)
                cv2.putText(image_array, f"{labels[i]}: {scores[i]:.1f}", org, font, 0.5, (0, 0, 255), 1)

        return VisionResponse(image_array, detected_objects, distances, description)
    
    # def store_image(self, cv2_image = None):
    #     if cv2_image is None:
    #         image = Image.new('RGB', (self.env["captured_width"], self.env["captured_height"]), 'black')
    #     else:
    #         image = utils.OpenCV2PIL(cv2_image)

    #     ## add 'assistant' as a parameter into the function
    #     # if (self.env["text_insertion"]):
    #     #     text = "\n".join([f"Current Position: {assistant.curr_position}", f"Target: {assistant.target}", f"Likelihood: {assistant.likelihood}", f"Action: {assistant.action}", f"New Position: {assistant.new_position}", f"Reason: {assistant.reason}"])
    #     #     image = utils.image_text_append(image, self.env["captured_width"], self.env["captured_height"], text)
        
    #     # Format the filename based on the number of images stored
    #     self.image_counter += 1
    #     filename = f"image{self.image_counter:02d}.jpg"  # Pad with zeros to two digits
    #     image_path = os.path.join(self.save_dir, filename)

    #     # Save the image
    #     image.save(image_path)
    #     # print(f"Image saved to {image_path}")
