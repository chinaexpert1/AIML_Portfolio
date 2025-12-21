import numpy as np
from sklearn.preprocessing import label_binarize

def calculate_map_x_point_interpolated(precision_recall_points, num_classes, num_interpolated_points=11):
    """
    Calculate the mean average precision (mAP) using x-point interpolation for multi-class tasks.

    Args:
        precision_recall_points (dict): A dictionary where keys are class indices and values are lists of
                                        tuples representing (recall, precision) points in descending order.
        num_classes (int): Number of classes.

    Returns:
        float: The mAP value.
    """
    mean_average_precisions = []

    for i in range(num_classes):
        points = precision_recall_points[i]
        # the list is sorted by confidence in descending order
        points = sorted(points, key=lambda x: x[1], reverse=True)
        
        interpolated_precisions = []
        for recall_threshold in [j * 1./(num_interpolated_points-1) for j in range(num_interpolated_points)]:
            # Find all precisions with recall greater than or equal to the threshold
            possible_precisions = [p for r, p in points if r >= recall_threshold]
            
            # Interpolate precision: take the maximum precision to the right of the current recall level
            if possible_precisions:
                interpolated_precisions.append(max(possible_precisions))
            else:
                interpolated_precisions.append(0)
        
        # Calculate the mean of the interpolated precisions
        mean_average_precision = sum(interpolated_precisions) / len(interpolated_precisions)
        mean_average_precisions.append(mean_average_precision)
    
    # Calculate the overall mean average precision
    overall_map = sum(mean_average_precisions) / num_classes
    
    return overall_map


if __name__ == "__main__":
    """
    Example usage of `calculate_map_11_point_interpolated()` with sample precision-recall values 
    for a 3-class classification scenario.
    """

    num_classes = 3  # Define the number of classes

    # Sample precision values at different recall thresholds for each class
    precision = {
        0: [0.0, 0.0, 0.33, 0.25, 0.2, 0.16, 0.14, 0.125, 0.22, 0.2, 1.0],
        1: [0.0, 0.0, 0.0, 0.25, 0.4, 0.33, 0.28, 0.375, 0.33, 0.3, 1.0],
        2: [1.0, 1.0, 0.66, 0.5, 0.6, 0.5, 0.42, 0.375, 0.44, 0.5, 1.0]
    }

    # Sample recall values at different thresholds for each class
    recall = {
        0: [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0],
        1: [0.0, 0.0, 0.0, 0.33, 0.66, 0.66, 0.66, 1.0, 1.0, 1.0, 0.0],
        2: [0.2, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.8, 1.0, 0.0]
    }

    # Create precision-recall pairs for each class
    precision_recall_points = {
        class_index: list(zip(recall[class_index], precision[class_index]))
        for class_index in range(num_classes)
    }

    # Compute Mean Average Precision (mAP) using 11-point interpolation
    map_value = calculate_map_x_point_interpolated(precision_recall_points, num_classes)

    # Output the calculated mAP value with four decimal places
    print(f"Mean Average Precision (mAP): {map_value:.4f}")
