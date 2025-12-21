import cv2
import numpy as np
import random
import os
from pathlib import Path


## TASK 1: Complete the five augmenter class methods. 
#          - This class is used to transform data necessary for training TechTrack's models.
#          - Imagine that the output of `self.transform()` is fed directly to train the model.
#          - You will define your own keywords for "**kwargs".
#          --------------------------------------------------------------------------------
#          Create your own augmentation method. Use the same structure as the format used below.
#          For example,
#
#          def your_custom_transformation(**kwargs):
#              # your process
#              return ...
#
#          Name this method appropriately based on its capability. And add docstrings to 
#          describe its process.
#          --------------------------------------------------------------------------------
#          Provide a demonstration and visualizations of these methods in 
#          `techtrack/notebooks/augmentation.ipynb`.
    

class Augmenter:
    """
    A collection of dataset augmentation methods including transformations, 
    blurring, resizing, and brightness adjustments.
    """

    @staticmethod
    def horizontal_flip(image: np.ndarray) -> np.ndarray:
        """Horizontally flips the image."""
        return cv2.flip(image, 1)

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        Applies a stronger Gaussian blur to the image.
        
        :param image: Input image (numpy array).
        :param kernel_size: Size of the Gaussian blur kernel (default: 15, must be an odd number).
        :return: Blurred image.
        """
        # Ensure kernel size is odd for OpenCV
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def resize(image: np.ndarray, width: int = 128, height: int = 128) -> np.ndarray:
        """Resizes the image to the given dimensions."""
        return cv2.resize(image, (width, height))

    @staticmethod
    def change_brightness(image: np.ndarray, alpha: float = 1.2, beta: int = 30) -> np.ndarray:
        """Adjusts brightness and contrast of the image."""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def rotate(image: np.ndarray, angle: float = None) -> np.ndarray:
        """Rotates the image by a specified or random angle."""
        if angle is None:
            angle = random.uniform(-20, 20)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def transform(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies a random selection of transformations from the available methods.
        :param image: Input image.
        :param kwargs: Additional parameters for transformations.
        """
        augmentations = [
            lambda img: Augmenter.horizontal_flip(img),
            lambda img: Augmenter.gaussian_blur(img, kernel_size=kwargs.get("kernel_size", 5)),
            lambda img: Augmenter.resize(img, width=kwargs.get("width", 416), height=kwargs.get("height", 416)),
            lambda img: Augmenter.change_brightness(img, alpha=kwargs.get("alpha", 1.2), beta=kwargs.get("beta", 30)),
            lambda img: Augmenter.rotate(img, angle=kwargs.get("angle", None)),
        ]
        random.shuffle(augmentations)
        num_to_apply = random.randint(1, len(augmentations))

        for aug in augmentations[:num_to_apply]:
            image = aug(image)

        return image

import cv2
import numpy as np
from modules.rectification.augmentation import Augmenter

if __name__ == "__main__":
    def test_augmentation():
        """Quick test function to validate augmentation methods."""

        # Automatically find the project root (assumes running from the project root)
        project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from `modules/rectification/`
        image_path = project_root / "storage/sample_image.jpg"

        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"[INFO] Successfully loaded image from {image_path}")

        augmenter = Augmenter()

        # Define augmentation parameters
        kwargs = {
            "image": image,
            "kernel_size": 7,  # Stronger blur
            "width": 512, "height": 512,  # Resize to different dimensions
            "alpha": 1.5, "beta": 50,  # Increase brightness more
            "angle": 30,  # Fixed rotation angle
        }

        augmented_image = augmenter.transform(**kwargs)

        print("[INFO] Augmentation applied: Random transformations from Augmenter class")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Augmented Image")
        axes[1].axis("off")

        plt.show()

    test_augmentation()

