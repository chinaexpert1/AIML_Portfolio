import numpy as np
from typing import List, Tuple

class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to filter redundant bounding boxes 
    in object detection.

    This class takes bounding boxes, confidence scores, and class IDs and applies 
    NMS to retain only the most relevant bounding boxes based on confidence scores 
    and Intersection over Union (IoU) thresholding.
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        :param score_threshold: The minimum confidence score required to retain a bounding box.
        :param nms_iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[List[float]],
    ) -> Tuple[List[List[int]], List[int], List[float], List[List[float]]]:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        :param bboxes: A list of bounding boxes, where each box is represented as 
                       [x, y, width, height]. (x, y) is the top-left corner.
        :param class_ids: A list of class IDs corresponding to each bounding box.
        :param scores: A list of confidence scores for each bounding box.
        :param class_scores: A list of class-specific scores for each detection.
        
        :return: A tuple containing:
            - filtered_bboxes (List[List[int]]): The final bounding boxes after NMS.
            - filtered_class_ids (List[int]): The class IDs of retained bounding boxes.
            - filtered_scores (List[float]): The confidence scores of retained bounding boxes.
            - filtered_class_scores (List[List[float]]): The class-specific scores of retained boxes.
        """
        if len(bboxes) == 0:
            return [], [], [], []

        # Sort detections by descending confidence score.
        indices = np.argsort(scores)[::-1]
        filtered_bboxes = []
        filtered_class_ids = []
        filtered_scores = []
        filtered_class_scores = []

        while len(indices) > 0:
            best_index = indices[0]
            best_bbox = bboxes[best_index]

            filtered_bboxes.append(best_bbox)
            filtered_class_ids.append(class_ids[best_index])
            filtered_scores.append(scores[best_index])
            filtered_class_scores.append(class_scores[best_index])

            remaining_indices = []
            for i in indices[1:]:
                iou = self.compute_iou(best_bbox, bboxes[i])
                if iou < self.nms_iou_threshold:
                    remaining_indices.append(i)
            indices = np.array(remaining_indices)

        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores

    def compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Computes Intersection over Union (IoU) between two bounding boxes.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        intersection = inter_width * inter_height

        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0.0

    def vectorized_nms(self, bboxes: List[List[float]], class_ids: List[int], scores: List[float], class_scores: List[List[float]], iou_threshold: float) -> Tuple[List[List[float]], List[int], List[float], List[List[float]]]:
        """
        Applies Non-Maximum Suppression (NMS) per class in a vectorized manner.
        
        Parameters:
            bboxes: List of bounding boxes, each [x, y, w, h] (floats).
            class_ids: List of predicted class IDs (ints).
            scores: List of objectness scores (floats).
            class_scores: List of class probability vectors.
            iou_threshold: IoU threshold for suppression.
        
        Returns:
            filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores
        """
        if len(bboxes) == 0:
            return [], [], [], []
        
        boxes = np.array(bboxes)           # shape (N, 4)
        scores = np.array(scores)          # shape (N,)
        class_ids = np.array(class_ids)    # shape (N,)
        class_scores = np.array(class_scores)  # shape (N, num_classes)
        
        keep_indices = []
        # Process each class separately:
        for cls in np.unique(class_ids):
            # Get indices for this class
            cls_inds = np.where(class_ids == cls)[0]
            cls_boxes = boxes[cls_inds]
            cls_scores = scores[cls_inds]
            
            # Order detections by score (descending)
            order = cls_scores.argsort()[::-1]
            
            while order.size > 0:
                i = order[0]
                keep_indices.append(cls_inds[i])
                
                # Compute IoU between the box with highest score and the rest
                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 0] + cls_boxes[i, 2], cls_boxes[order[1:], 0] + cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 1] + cls_boxes[i, 3], cls_boxes[order[1:], 1] + cls_boxes[order[1:], 3])
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                
                area_i = cls_boxes[i, 2] * cls_boxes[i, 3]
                areas = cls_boxes[order[1:], 2] * cls_boxes[order[1:], 3]
                union = area_i + areas - inter
                iou = inter / union
                
                # Keep indices with IoU less than the threshold
                inds = np.where(iou < iou_threshold)[0]
                order = order[inds + 1]  # shift index because order[0] was current
        # End per-class loop
        
        keep_indices = np.array(keep_indices)
        
        filtered_bboxes = boxes[keep_indices].tolist()
        filtered_class_ids = class_ids[keep_indices].tolist()
        filtered_scores = scores[keep_indices].tolist()
        filtered_class_scores = class_scores[keep_indices].tolist()
        
        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores
