import cv2
import numpy as np
from typing import List, Tuple

from .base_detector import BaseDetector


class ClassicalDetector(BaseDetector):
    """
    A classical computer vision detector for finding a cricket ball.

    This detector uses background subtraction and contour analysis to
    identify the ball in a video with a static camera.
    """

    def __init__(self, config: dict = None):
        super().__init__(config=config)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get("history", 300),
            varThreshold=self.config.get("varThreshold", 25),
            detectShadows=self.config.get("detectShadows", False)
        )
        # Parameters for contour filtering
        self.min_area = self.config.get("min_area", 20)
        self.max_area = self.config.get("max_area", 300)
        self.min_circularity = self.config.get("min_circularity", 0.6)
        self.minRadius = self.config.get("minRadius", 3)
        self.maxRadius = self.config.get("maxRadius", 12)
        
        # Frame counter for background model warmup
        self.frame_count = 0
        self.warmup_frames = self.config.get("warmup_frames", 100)

        # Temporal gating
        self.prev_center = None
        self.max_jump_distance = self.config.get("max_jump_distance", 50)

        # Advanced filters
        self.hsv_lower = np.array(self.config.get("hsv_lower", [0, 0, 180]))
        self.hsv_upper = np.array(self.config.get("hsv_upper", [180, 40, 255]))
        self.hsv_lower2 = np.array(self.config.get("hsv_lower2", None))
        self.hsv_upper2 = np.array(self.config.get("hsv_upper2", None))
        self.min_color_ratio = self.config.get("min_color_ratio", 0.4)
        self.canny_thresh1 = self.config.get("canny_thresh1", 50)
        self.canny_thresh2 = self.config.get("canny_thresh2", 150)
        self.min_edge_density = self.config.get("min_edge_density", 0.1)

        # Frame cropping
        self.crop_width_ratio = self.config.get("crop_width_ratio", 1.0)


    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects a ball in the frame using classical CV techniques.

        Args:
            frame: A numpy array representing the video frame (in BGR format).

        Returns:
            A list containing a single bounding box tuple if a ball is found,
            otherwise an empty list.
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

        # 1. Background Subtraction
        learning_rate = -1 # Default learning rate
        if self.frame_count > self.warmup_frames:
            learning_rate = 0 # Freeze background model
        fg_mask = self.bg_subtractor.apply(frame_cropped, learningRate=learning_rate)
        self.frame_count += 1

        # 2. Image Preprocessing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour Detection and Filtering
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
            
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if not (self.minRadius < r < self.maxRadius):
                continue
            
            # Candidate passed shape filters, add to list
            candidates.append(((x, y), r, area))

        # 4. Temporal Gating
        if self.prev_center is not None:
            # Filter candidates by distance to previous detection
            valid_candidates = []
            for (x,y), r, area in candidates:
                dist = np.linalg.norm(np.array([x, y]) - self.prev_center)
                if dist < self.max_jump_distance:
                    valid_candidates.append(((x, y), r, area))
            candidates = valid_candidates

        best_candidate = None
        if candidates:
            # Select the best candidate (e.g., largest radius)
            candidates.sort(key=lambda c: c[1], reverse=True)
            best_candidate = candidates[0]

        if best_candidate:
            (x, y), r, area = best_candidate
            
            # 5. Advanced Sanity Checks on the best candidate
            x1_crop, y1, x2_crop, y2 = int(x - r), int(y - r), int(x + r), int(y + r)
            
            # HSV Color Check
            h_crop, w_crop, _ = frame_cropped.shape
            roi = frame_cropped[max(0, y1):min(h_crop, y2), max(0, x1_crop):min(w_crop, x2_crop)]
            if roi.size == 0:
                self.prev_center = None
                return []
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            if self.hsv_lower2.any() and self.hsv_upper2.any():
                mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
                mask = cv2.bitwise_or(mask, mask2)

            color_ratio = np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

            if color_ratio < self.min_color_ratio:
                self.prev_center = None
                return []

            # Edge Strength Check
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, self.canny_thresh1, self.canny_thresh2)
            edge_density = np.count_nonzero(edges) / (roi.shape[0] * roi.shape[1])

            if edge_density < self.min_edge_density:
                self.prev_center = None
                return []

            # Candidate passed all checks
            self.prev_center = np.array([x, y])
            confidence = 0.9  # Fixed confidence

            # Translate coordinates back to original frame
            x1 = x1_crop + crop_start
            x2 = x2_crop + crop_start
            return [(x1, y1, x2, y2, confidence)]
        else:
            # No valid detection, reset temporal gate
            self.prev_center = None
            return []
