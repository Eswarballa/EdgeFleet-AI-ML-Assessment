import cv2
import numpy as np
from typing import List, Tuple


def draw_centroid(frame: np.ndarray, position: Tuple[int, int], visible: bool, radius: int = 5):
    """
    Draws a circle at the tracked position of the ball.

    The color of the circle indicates the visibility status.
    - Green: The ball was detected in the current frame.
    - Red: The ball was not detected; this is a predicted position.

    Args:
        frame: The video frame to draw on.
        position: A tuple (x, y) for the centroid's position.
        visible: A boolean indicating if the ball was detected in this frame.
        radius: The radius of the circle to draw.
    """
    color = (0, 255, 0) if visible else (0, 0, 255)  # Green for visible, Red for predicted
    cv2.circle(frame, position, radius, color, -1)


def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple[int, int]], color=(255, 255, 0), thickness: int = 2):
    """
    Draws the trajectory of the ball on the frame as a simple line.
    (This function will be replaced by draw_perspective_trajectory)
    """
    if len(trajectory) > 1:
        pts = np.array(trajectory, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)

def draw_perspective_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    base_width: int = 50,  # Width at the bottom of the screen (max y)
    min_width: int = 5,    # Width at the top of the screen (min y)
    color=(255, 255, 255), # White color for the spotlight
    alpha: float = 0.3,    # Transparency level
    smoothing_window: int = 5 # Size of moving average window
):
    """
    Draws a smoothed, perspective-aware, semi-transparent trajectory (spotlight effect) on the frame.

    Args:
        frame: The video frame to draw on.
        trajectory: A list of (x, y) points representing the path.
        base_width: The maximum width of the trajectory at the bottom of the frame (max y).
        min_width: The minimum width of the trajectory at the top of the frame (min y).
        color: The color of the trajectory polygon.
        alpha: The transparency level (0.0 to 1.0).
        smoothing_window: The window size for moving average smoothing.
    """
    if len(trajectory) < 2:
        return

    # Apply smoothing
    smoothed_trajectory = []
    if len(trajectory) >= smoothing_window:
        trajectory_np = np.array(trajectory)
        # Simple moving average for x and y separately
        for i in range(len(trajectory)):
            start_idx = max(0, i - smoothing_window + 1)
            end_idx = i + 1
            window = trajectory_np[start_idx:end_idx]
            avg_point = np.mean(window, axis=0)
            smoothed_trajectory.append((int(avg_point[0]), int(avg_point[1])))
    else:
        smoothed_trajectory = trajectory[:]

    overlay = frame.copy()
    
    # Calculate the perspective polygon based on smoothed trajectory
    polygon_points = []
    left_points = []
    right_points = []
    frame_height = frame.shape[0]

    for x, y in smoothed_trajectory:
        # Calculate width based on y-coordinate (linear interpolation)
        current_width = min_width + (base_width - min_width) * (y / frame_height)
        current_width = max(0, current_width)

        x_left = int(x - current_width / 2)
        x_right = int(x + current_width / 2)
        
        left_points.append((x_left, y))
        right_points.append((x_right, y))
    
    # Combine points
    polygon_points.extend(left_points)
    right_points.reverse()
    polygon_points.extend(right_points)
    
    polygon_points = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [polygon_points], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_detection_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float):
    """
    Draws the bounding box of a detection on the frame.

    Args:
        frame: The video frame to draw on.
        bbox: A tuple (x1, y1, x2, y2) for the bounding box.
        confidence: The confidence score of the detection.
    """
    x1, y1, x2, y2 = bbox
    color = (255, 0, 0) # Blue for detection box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Put confidence score
    label = f"{confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
