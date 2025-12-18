from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """
    A detector that uses a YOLO model for object detection.

    This class is a wrapper around the `ultralytics` library, making it
    compatible with the `BaseDetector` interface. It can be used with
    pre-trained models like YOLOv8 or custom-trained models.
    """

    def __init__(self, model_path: str, config: dict = None):
        super().__init__(model_path=model_path, config=config)
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}. Error: {e}")

        # COCO class index for 'sports ball' is 32.
        self.target_class_id = self.config.get("target_class_id", 32)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)

        # Frame cropping
        self.crop_width_ratio = self.config.get("crop_width_ratio", 1.0)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Performs object detection using the YOLO model.

        Args:
            frame: A numpy array representing the video frame (in BGR format).

        Returns:
            A list of detected bounding boxes for the target class that meet
            the confidence threshold.
        """
        # 0. Frame Cropping
        h, w, _ = frame.shape
        crop_start = 0
        if self.crop_width_ratio < 1.0:
            crop_width = int(w * self.crop_width_ratio)
            crop_start = (w - crop_width) // 2
            frame_cropped = frame[:, crop_start:crop_start + crop_width]
        else:
            frame_cropped = frame

        # The ultralytics library expects BGR images, which is the default for OpenCV.
        results = self.model.predict(frame_cropped, conf=self.confidence_threshold, verbose=False)

        detections = []
        if results:
            result = results[0]  # Get results for the first image
            for box in result.boxes:
                if box.cls == self.target_class_id:
                    x1_crop, y1, x2_crop, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Translate coordinates back to original frame
                    x1 = x1_crop + crop_start
                    x2 = x2_crop + crop_start
                    detections.append((x1, y1, x2, y2, confidence))

        return detections
