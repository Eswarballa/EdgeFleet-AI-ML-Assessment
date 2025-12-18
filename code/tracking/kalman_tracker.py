import cv2
import numpy as np


class KalmanTracker:
    """
    A Kalman Filter-based tracker for a single object.

    This class uses a Kalman Filter to predict and track the position of an
    object (like a cricket ball) over time, smoothing the trajectory and
    providing fallback predictions when detections are missed.
    """

    def __init__(self, tracking_config):
        """
        Initializes the Kalman Tracker.

        Args:
            tracking_config: A Pydantic model containing tracking parameters.
        """
        dt = 1.0  # Time step, assuming constant frame rate
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # Use configurable noise covariance matrices
        # Process noise covariance (Q)
        q = tracking_config.process_noise_cov
        self.kf.processNoiseCov = np.array([
            [q[0], 0, 0, 0],
            [0, q[1], 0, 0],
            [0, 0, q[2], 0],
            [0, 0, 0, q[3]]
        ], dtype=np.float32)
        
        # Measurement noise covariance (R)
        r = tracking_config.measurement_noise_cov
        self.kf.measurementNoiseCov = np.array([
            [r[0], 0],
            [0, r[1]]
        ], dtype=np.float32)
        
        # Error covariance post (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1

        self.is_initialized = False
        self.age = 0
        self.max_age = tracking_config.max_age
        self.last_observation = None

    def _bbox_to_centroid(self, bbox):
        """Converts a bounding box (x1, y1, x2, y2) to a centroid (x, y)."""
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        return np.array([x, y], dtype=np.float32)

    def update(self, detection_bbox):
        """
        Updates the tracker with a new detection.

        Args:
            detection_bbox: A tuple (x1, y1, x2, y2) for the detected object's
                            bounding box, or None if no detection was made.
        """
        # Predict the next state
        if self.is_initialized:
            self.kf.predict()

        if detection_bbox is not None:
            # A detection was made
            self.last_observation = self._bbox_to_centroid(detection_bbox)
            
            if not self.is_initialized:
                # First detection, initialize the filter state
                self.kf.statePost = np.array([self.last_observation[0], self.last_observation[1], 0, 0], dtype=np.float32)
                self.is_initialized = True
            else:
                # Correct the state with the new measurement
                self.kf.correct(self.last_observation)
            
            self.age = 0
        else:
            # No detection, rely on prediction
            if self.is_initialized:
                self.age += 1
                if self.age > self.max_age:
                    # Mark as lost
                    self.is_initialized = False
    
    def get_state(self):
        """
        Returns the current estimated state of the object.

        Returns:
            A tuple (x, y) representing the estimated centroid position,
            or None if the tracker is not active.
        """
        if self.is_initialized:
            state = self.kf.statePost
            return int(state[0]), int(state[1])
        return None
