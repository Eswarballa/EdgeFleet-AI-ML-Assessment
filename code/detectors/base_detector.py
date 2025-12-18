from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for a detection model.

    All detectors must inherit from this class and implement the `detect` method.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Performs object detection on a single frame.

        Args:
            frame: A numpy array representing the video frame (in BGR format).

        Returns:
            A list of detected bounding boxes. Each bounding box is a tuple of
            (x1, y1, x2, y2, confidence_score).
            - (x1, y1) is the top-left corner.
            - (x2, y2) is the bottom-right corner.
            - confidence_score is the model's confidence in the detection.
            Returns an empty list if no objects are detected.
        """
        pass

    def __init__(self, model_path: str = None, config: dict = None):
        """
        Initializes the detector.

        Args:
            model_path: Path to the model weights file (e.g., `.pt` file).
            config: A dictionary of configuration parameters for the model.
        """
        self.model_path = model_path
        self.config = config if config is not None else {}
