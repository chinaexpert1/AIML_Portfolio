import cv2
import os
from modules.inference.nms import NMS
from modules.inference.model import Detector


class InferenceService:
    """
    Handles inference on images from a specified folder by applying object detection
    and saving detection results as .txt files.
    """
    
    def __init__(self, input_folder: str, detector: Detector, nms: NMS, output_folder: str) -> None:
        """
        Initializes the inference service for image folder processing.

        :param input_folder: Folder containing input images (.jpg).
        :param detector: An instance of the Detector class.
        :param nms: An instance of the NMS class.
        :param output_folder: Folder where detection results (.txt files) will be saved.
        """
        self.input_folder = input_folder
        self.detector = detector
        self.nms = nms
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        print("[INFO] Inference Service initialized.")

    def run(self) -> None:
        """
        Processes each image in the input folder, runs object detection,
        applies NMS filtering, and saves the detection results to a .txt file in the output folder.
        Each detection file will have lines formatted as:
            x y w h objectness cls_score_0 cls_score_1 ... cls_score_19
        """
        # List all .jpg files in the input folder
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith('.jpg')]
        
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"[WARN] Unable to read {image_path}. Skipping.")
                continue

            # Run detection on the image
            predictions = self.detector.predict(frame)
            bboxes, class_ids, confidence_scores, class_scores = self.detector.post_process(predictions, score_threshold=0.5)
            
            
            # Apply vectorized NMS:
            bboxes, class_ids, confidence_scores, class_scores = nms.vectorized_nms(bboxes, class_ids, confidence_scores, class_scores, 0.4)
            
            for bbox, conf in zip(bboxes, confidence_scores):
                print(f"Image {image_file}: Score: {conf:.6f}, BBox: {bbox}")
            
            # Save detections to a text file in the output folder
            detection_file = os.path.join(self.output_folder, os.path.splitext(image_file)[0] + ".txt")
            with open(detection_file, 'w') as f:
                # Write each detection with full details (without class_id)
                for bbox, conf, cls_score in zip(bboxes, confidence_scores, class_scores):
                    # Format: x y w h objectness cls_score_0 ... cls_score_19
                    line = f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} {conf:.6f} " + " ".join(f"{s:.6f}" for s in cls_score)
                    f.write(line + "\n")
            print(f"[INFO] Processed {image_file}; detections saved to {detection_file}")



# Runner for Inference Service
if __name__ == "__main__":
    print("[INFO] Starting Inference Service...")

    # Input folder containing .jpg images and corresponding ground truth .txt files
    INPUT_FOLDER = "storage/logistics"
    # Output folder to save the detection results
    OUTPUT_FOLDER = "detections "

    # Initialize YOLO Model
    WEIGHTS_PATH = "storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights"
    CONFIG_PATH = "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg"
    CLASS_NAMES_PATH = "storage/yolo_models/logistics.names"

    print("[INFO] Loading YOLO Model...")
    model = Detector(WEIGHTS_PATH, CONFIG_PATH, CLASS_NAMES_PATH)
    print("[INFO] Model loaded successfully.")

    # Initialize NMS
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    print("[INFO] Initializing Non-Maximum Suppression...")
    nms = NMS(SCORE_THRESHOLD, IOU_THRESHOLD)
    print("[INFO] NMS initialized.")

    # Create and run the inference service on the images in INPUT_FOLDER
    service = InferenceService(INPUT_FOLDER, model, nms, OUTPUT_FOLDER)
    service.run()

    print("[INFO] Inference Service terminated.")
