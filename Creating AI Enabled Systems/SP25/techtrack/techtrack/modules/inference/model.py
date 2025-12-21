import cv2
import numpy as np
from typing import List, Tuple

class Detector:
    """
    A class that represents an object detection model using OpenCV's DNN module
    with a YOLO-based architecture.
    """

    def __init__(self, weights_path: str, config_path: str, class_path: str, score_threshold: float = 0.5) -> None:
        """
        Initializes the YOLO model by loading the pre-trained network and class labels.

        :param weights_path: Path to the pre-trained YOLO weights file.
        :param config_path: Path to the YOLO configuration file.
        :param class_path: Path to the file containing class labels.
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Load class labels
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.img_height: int = 0
        self.img_width: int = 0
        self.score_threshold = score_threshold

    def predict(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        if preprocessed_frame is None:
            print("[ERROR] Received None as input frame.")
            return []

        # Update image dimensions based on the original image
        self.img_height, self.img_width = preprocessed_frame.shape[:2]

        # Create blob from image for YOLO
        blob = cv2.dnn.blobFromImage(
            preprocessed_frame, 
            scalefactor=1/255.0, 
            size=(416, 416), 
            swapRB=True, 
            crop=False
        )
        self.net.setInput(blob)

        # Get the names of the output layers
        output_layers = self.net.getUnconnectedOutLayersNames()
        if not output_layers:
            print("[ERROR] No output layers found in YOLO model.")
            return []

        # Forward pass: get raw predictions from the network
        outputs = self.net.forward(output_layers)
        if outputs is None or len(outputs) == 0:
            print("[ERROR] No output received from YOLO forward pass.")
            return []
        
        # Concatenate outputs from all layers into one array
        predictions = np.concatenate(outputs, axis=0)
        # Debug: Uncomment to print prediction info
        # print("[DEBUG] Predictions shape:", predictions.shape)
        # print("[DEBUG] First 5 predictions:", predictions[:5])
        
        return [predictions]

    def post_process(self, predict_output: List[np.ndarray], score_threshold: float) -> Tuple[List[List[float]], List[int], List[float], List[np.ndarray]]:
        """
        Processes the raw YOLO model predictions and filters out low-confidence detections.

        :param predict_output: List of NumPy arrays from the YOLO forward pass.
        :param score_threshold: Minimum objectness score required to keep a detection.
        
        :return: A tuple containing:
            - bboxes: List of bounding boxes [x, y, width, height] in absolute coordinates.
            - class_ids: List of predicted class indices.
            - confidence_scores: List of objectness scores for each detection.
            - class_scores: List of all class-specific confidence scores.
        """
        # Concatenate all detection outputs into one array (shape: [N, 5+num_classes])
        predictions = np.concatenate(predict_output, axis=0)
        
        # Use column index 4 as objectness score (assumed to be already activated via sigmoid)
        objectness = predictions[:, 4]
        
        # Filter out detections with low objectness scores
        keep = objectness >= score_threshold
        predictions = predictions[keep]
        objectness = objectness[keep]
        
        if predictions.shape[0] == 0:
            return [], [], [], []
        
        # The remaining columns (index 5 onward) are class scores.
        class_probabilities = predictions[:, 5:]
        # Predicted class IDs (argmax over class probabilities)
        class_ids = np.argmax(class_probabilities, axis=1)
        
        # Extract bounding box coordinates in center-based format: [center_x, center_y, width, height]
        center_x = predictions[:, 0]
        center_y = predictions[:, 1]
        width = predictions[:, 2]
        height = predictions[:, 3]
        
        # Compute scaling factors based on the original image size and blob size (416x416)
        scale_x = self.img_width / 416.0
        scale_y = self.img_height / 416.0
        
        # Convert center-based coordinates to top-left corner format and scale them to the original image dimensions.
        x = (center_x - width / 2) * scale_x
        y = (center_y - height / 2) * scale_y
        w = width * scale_x
        h = height * scale_y
        bboxes = np.stack([x, y, w, h], axis=1)
        
        # Confidence scores are taken from the objectness values
        confidence_scores = objectness
        
        # Return all values as lists.
        return bboxes.tolist(), class_ids.tolist(), confidence_scores.tolist(), class_probabilities.tolist()



"""
EXAMPLE USAGE:
detector = Detector(weights_path, config_path, class_path, score_threshold=0.5)
frame = cv2.imread("path/to/image.jpg")
predictions = detector.predict(frame)
bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predictions, score_threshold=0.5)
"""
