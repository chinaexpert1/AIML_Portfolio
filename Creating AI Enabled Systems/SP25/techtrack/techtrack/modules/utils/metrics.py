import numpy as np
from sklearn.preprocessing import label_binarize

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    The IoU is computed as the area of overlap divided by the union area.
    """
    # Unpack the bounding boxes
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    # Coordinates of the intersection rectangle
    x_left = max(xA, xB)
    y_top = max(yA, yB)
    x_right = min(xA + wA, xB + wB)
    y_bottom = min(yA + hA, yB + hB)

    # Compute width and height of the intersection rectangle
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    boxA_area = wA * hA
    boxB_area = wB * hB

    # Compute the union area
    union_area = boxA_area + boxB_area - inter_area

    # Return IoU (avoid division by zero)
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores"):
    """
    Evaluate detections by matching predicted bounding boxes with ground truth boxes
    and generate corresponding true labels and prediction scores for further evaluation.
    
    For each image, detections are first sorted in descending order by an evaluation score 
    (derived from objectness, classification, or their product). Each detection is then 
    matched to the ground truth box with the highest IoU. If the IoU exceeds the given threshold 
    and the predicted class equals the ground truth label—and that ground truth has not yet been matched—the 
    detection is considered a true positive (TP). Otherwise it is marked as a false positive (FP). 
    Finally, any ground truth that was not matched is added as a false negative (FN) with a dummy score of 0.
    
    The output is organized so that for each candidate (detection or missed gt) we produce:
      - a true label (the ground truth class for TPs and FNs, and -1 for FPs)
      - a prediction score vector of length num_classes with the evaluated score (if available) at the predicted class index.
    
    These outputs are then used to compute precision–recall curves.
    """
    y_true_result = []
    pred_scores_result = []
    
    # Determine number of classes from the first image's classification scores
    if cls_scores and isinstance(cls_scores[0], np.ndarray) and cls_scores[0].ndim > 1:
        num_classes = cls_scores[0].shape[1]
    else:
        num_classes = max([max(lst) for lst in gt_classes if lst]) + 1

    for i in range(len(boxes)):
        # Create a list to keep track of which ground truths in image i have been matched
        gt_matched = [False] * len(gt_boxes[i])
        
        # Helper function to compute the evaluation score for a given detection index j
        def get_eval_score(j):
            if eval_type == "objectness":
                return scores[i][j]
            elif eval_type == "class_scores":
                # Each cls_scores[i][j] is assumed to be a vector; take its maximum value.
                return float(max(cls_scores[i][j]))
            elif eval_type == "combined":
                return scores[i][j] * float(max(cls_scores[i][j]))
            else:
                return scores[i][j]
        
        # Process detections in descending order of evaluation score
        detection_indices = list(range(len(boxes[i])))
        sorted_indices = sorted(detection_indices, key=lambda j: get_eval_score(j), reverse=True)
        
        for j in sorted_indices:
            det_box = boxes[i][j]
            det_class = classes[i][j]
            eval_score = get_eval_score(j)
            
            # Find the best matching ground truth box (if any)
            best_iou = 0
            best_gt_idx = -1
            for k, gt_box in enumerate(gt_boxes[i]):
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = k
            
            if best_iou >= map_iou_threshold and best_gt_idx != -1 and (gt_classes[i][best_gt_idx] == det_class) and not gt_matched[best_gt_idx]:
                # True positive: detection correctly matches a ground truth
                gt_matched[best_gt_idx] = True
                y_true_result.append(det_class)
                score_vector = [0] * num_classes
                score_vector[det_class] = eval_score
                pred_scores_result.append(score_vector)
            else:
                # False positive: detection does not match any ground truth appropriately
                y_true_result.append(-1)  # Using -1 to denote an incorrect detection for its predicted class
                score_vector = [0] * num_classes
                score_vector[det_class] = eval_score
                pred_scores_result.append(score_vector)
        
        # For every ground truth that was not detected, add a false negative (FN)
        for k, matched in enumerate(gt_matched):
            if not matched:
                y_true_result.append(gt_classes[i][k])
                # Dummy score vector (all zeros) for a missed detection
                score_vector = [0] * num_classes
                pred_scores_result.append(score_vector)
    
    return y_true_result, pred_scores_result

def calculate_precision_recall_curve(y_true, pred_scores, num_classes=20):
    """
    Compute the precision-recall curve for each class in a multi-class classification task.
    
    For each class, the function binarizes the true labels (1 if the sample’s label matches the class, 0 otherwise)
    and extracts the corresponding predicted scores. It then iterates over unique thresholds (derived from the 
    sorted predicted scores) to compute the counts of true positives (TP), false positives (FP), and false negatives (FN),
    from which precision (TP/(TP+FP)) and recall (TP/(TP+FN)) are derived.
    """
    precision = {}
    recall = {}
    thresholds = {}

    y_true = np.array(y_true)
    pred_scores = np.array(pred_scores)

    for c in range(num_classes):
        # Create a binary vector: 1 if the sample is a positive instance for class c, 0 otherwise.
        binary_true = (y_true == c).astype(int)
        # Extract predicted scores for class c
        scores_c = pred_scores[:, c]
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores_c)[::-1]
        sorted_true = binary_true[sorted_indices]
        sorted_scores = scores_c[sorted_indices]
        
        # Total positive ground truths for this class
        total_positives = np.sum(binary_true)
        
        # Compute precision and recall at each unique threshold
        prec_vals = []
        rec_vals = []
        thresh_list = []
        unique_thresholds = np.unique(sorted_scores)[::-1]  # unique thresholds in descending order
        
        for thresh in unique_thresholds:
            # For detections with score >= threshold, count TP and FP
            idx = sorted_scores >= thresh
            TP = np.sum(sorted_true[idx])
            FP = np.sum(1 - sorted_true[idx])
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / total_positives if total_positives > 0 else 0
            prec_vals.append(prec)
            rec_vals.append(rec)
            thresh_list.append(thresh)
        
        # Append an extra threshold (0) to complete the curve if desired
        thresholds[c] = np.append(thresh_list, 0)
        precision[c] = prec_vals
        recall[c] = rec_vals

    return precision, recall, thresholds

def calculate_map_x_point_interpolated(precision_recall_points, num_classes, num_interpolated_points=11):
    """
    Calculate the Mean Average Precision (mAP) using x-point interpolation.
    
    For each class, the precision-recall points (a list of (recall, precision) tuples) are first sorted in descending order 
    by precision. Then, for each of the fixed recall thresholds (0.0, 0.1, ..., 1.0), the maximum precision for recall values 
    greater than or equal to the threshold is selected. The average precision for each class is the mean of these interpolated 
    precision values, and the mAP is the average over all classes.
    """
    mean_average_precisions = []

    for i in range(num_classes):
        # Retrieve and sort the precision-recall points for the current class.
        points = precision_recall_points[i]
        points = sorted(points, key=lambda x: x[1], reverse=True)
        
        interpolated_precisions = []
        # Generate equally spaced recall thresholds.
        for recall_threshold in [j * 0.1 for j in range(num_interpolated_points)]:
            possible_precisions = [p for r, p in points if r >= recall_threshold]
            if possible_precisions:
                interpolated_precisions.append(max(possible_precisions))
            else:
                interpolated_precisions.append(0)
        
        mean_average_precision = sum(interpolated_precisions) / len(interpolated_precisions)
        mean_average_precisions.append(mean_average_precision)
    
    overall_map = sum(mean_average_precisions) / num_classes
    
    return overall_map

if __name__ == "__main__":
    # -------------------------
    # Configuration Parameters
    # -------------------------
    
    num_classes = 3
    map_iou_threshold = 0.5

    # ---------------------------
    # Ground Truth Initialization
    # ---------------------------
    
    gt_boxes = [
        [[33, 117, 259, 396], [362, 161, 259, 362]],
        [[163, 29, 301, 553]]
    ]
    
    gt_classes = [
        [0, 0],
        [2]
    ]
    
    # -------------------------------
    # Detection (Prediction) Setup
    # -------------------------------
    
    boxes = [
        [[30, 187, 253, 276], [363, 194, 266, 291], [460, 371, 52, 23]],
        [[147, 26, 322, 578]]
    ]
    
    classes = [
        [0, 0, 1],
        [2]
    ]
    
    scores = [
        [0.95, 0.92, 0.30],
        [0.91]
    ]
    
    import numpy as np
    dummy_max_cls_scores = [
        [0.85, 0.75, 0.65],
        [0.80]
    ]
    cls_scores = [
        np.eye(num_classes)[np.array(class_list)] * np.array(score_list)
        for class_list, score_list in zip(classes, dummy_max_cls_scores)
    ]

    # ---------------------------
    # Evaluation of Detections
    # ---------------------------
    
    y_true, pred_scores = evaluate_detections(
        boxes, classes, scores, cls_scores,
        gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores"
    )
    
    print("True labels:", y_true)
    print("Prediction scores:", pred_scores)
    
    # ---------------------------
    # Precision-Recall Curve Calculation
    # ---------------------------
    
    precision, recall, thresholds = calculate_precision_recall_curve(
        y_true, pred_scores, num_classes=num_classes
    )
    
    for cls in range(num_classes):
        print(f"\nClass {cls}:")
        print("Precision:", precision[cls])
        print("Recall:", recall[cls])
        print("Thresholds:", thresholds[cls])
    
    # ---------------------------
    # Creating Precision-Recall Pairs
    # ---------------------------
    
    precision_recall_points = {
        class_index: list(zip(recall[class_index], precision[class_index]))
        for class_index in range(num_classes)
    }
    
    # ---------------------------
    # Compute Mean Average Precision (mAP)
    # ---------------------------
    
    map_value = calculate_map_x_point_interpolated(precision_recall_points, num_classes)
    print(f"Mean Average Precision (mAP): {map_value:.4f}")
