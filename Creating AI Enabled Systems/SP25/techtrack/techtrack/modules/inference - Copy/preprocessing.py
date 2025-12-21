import cv2
import numpy as np
from typing import Generator


class Preprocessing:
    """
    Handles video file reading and frame extraction for object detection inference.
    """

    def __init__(self, filename: str, drop_rate: int = 10) -> None:
        """
        Initializes the Preprocessing class.

        :param filename: Path to the video file.
        :param drop_rate: The interval at which frames are selected. For example, 
                          `drop_rate=10` means every 10th frame is retained.
        """
        self.filename = filename
        self.drop_rate = drop_rate

    def capture_video(self) -> Generator[np.ndarray, None, None]:
        """
        Captures frames from a video file and yields every nth frame.

        :return: A generator yielding frames as NumPy arrays.
        """
        cap = cv2.VideoCapture(self.filename)

        if not cap.isOpened():
            raise ValueError(f"Error: Unable to open video file '{self.filename}'.")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no more frames are available

            if frame_count % self.drop_rate == 0:
                yield frame
            frame_count += 1

        cap.release()
