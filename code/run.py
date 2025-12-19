import argparse
import json
import os
import sys
import time
from datetime import datetime
import logging
from pydantic import ValidationError
import glob
import copy

import cv2
import csv
import numpy as np
import pandas as pd

from detectors.base_detector import BaseDetector
from detectors.classical_detector import ClassicalDetector
from detectors.yolo_detector import YOLODetector
from tracking.kalman_tracker import KalmanTracker
from utils.visualization import draw_centroid, draw_perspective_trajectory, draw_detection_box
from utils.logger import setup_logger, add_file_handler
from utils.evaluation import parse_xml_ground_truth, compute_metrics, load_system_predictions
from utils.training import train_model
from utils.config_models import PipelineConfig

logger = setup_logger()

def get_detector(detector_config) -> BaseDetector:
    """Factory function to get a detector instance based on the config."""
    logger.info(f"Initializing detector of type: {detector_config.type}")
    
    # Make model path absolute if it's relative
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = detector_config.model_path
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if detector_config.type == "classical":
        return ClassicalDetector(config=detector_config.hyperparameters.model_dump(exclude_none=True))
    elif detector_config.type == "yolo":
        if not model_path:
            raise ValueError("model_path is required for YOLO detector.")
        return YOLODetector(
            model_path=model_path,
            config=detector_config.hyperparameters.model_dump(exclude_none=True)
        )

class Pipeline:
    """Encapsulates the entire detection and tracking pipeline."""

    def __init__(self, config: PipelineConfig, ground_truth_path=None):
        self.config = config
        self.mode = config.mode
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Create a unique results directory name based on video name, detector type, and timestamp
        video_name = os.path.splitext(os.path.basename(str(self.config.io_config.input_video_path)))[0]
        detector_type = config.detector_config.type
        self.results_dir = os.path.join(project_root, 'results', detector_type, f"{video_name}_{run_id}")
        
        self.annotations_path = os.path.join(project_root, 'annotations', detector_type, f"{video_name}_{run_id}.csv")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)
        
        add_file_handler(os.path.join(self.results_dir, "pipeline.log"))
        
        logger.info(f"Pipeline run starting for {video_name} with ID: {run_id}")
        logger.info(f"Mode: {self.mode}")
        
        self.detector = get_detector(config.detector_config)
        self.tracker = KalmanTracker(config.tracking_config)
        # Ground truth is now loaded dynamically in _perform_evaluation
        self.ground_truth_df = None

        self.annotations = []
        self.trajectory_points = []
        
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            f.write(self.config.model_dump_json(indent=4))
        logger.info(f"Configuration saved to {self.results_dir}")



    def run(self):
        """Runs the pipeline based on the configured mode."""
        if self.mode in ["TRACK", "EVALUATE"]:
            self._run_track_evaluate_mode()
        elif self.mode == "TRAIN":
            # This is handled logically in main(), but good to have a placeholder or error if called directly
            logger.warning("TRAIN mode should be handled via utils.training, not distinct pipeline run method yet.")
            pass

    def _run_track_evaluate_mode(self):
        """Handles logic for TRACK and EVALUATE modes."""
        cap = cv2.VideoCapture(str(self.config.io_config.input_video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file {self.config.io_config.input_video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video_filename = os.path.basename(str(self.config.io_config.input_video_path)).split('.')[0] + "_output.mp4"
        out_video_path = os.path.join(self.results_dir, output_video_filename)
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        logger.info(f"Output video will be saved to {out_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            logger.debug(f"Processing frame {frame_count}/{total_frames}")

            detections = self.detector.detect(frame)
            best_detection = sorted(detections, key=lambda x: x[4], reverse=True)[0] if detections else None
            
            self.tracker.update(best_detection[:4] if best_detection else None)
            tracked_state = self.tracker.get_state()
            
            # Edge Filtering check
            if tracked_state:
                tx, ty = tracked_state
                margin = self.config.tracking_config.edge_margin
                if (tx < margin) or (tx > width - margin) or (ty < margin) or (ty > height - margin):
                    # Track has drifted to edge/corner, likely false. Kill it.
                    logger.debug(f"Track killed due to edge proximity: ({tx}, {ty})")
                    self.tracker.is_initialized = False # Force reset
                    tracked_state = None
            
            visibility = 1 if best_detection else 0
            
            if tracked_state:
                self.annotations.append([frame_count, tracked_state[0], tracked_state[1], visibility])
                self.trajectory_points.append(tracked_state)
            else:
                self.annotations.append([frame_count, "", "", 0])

            if tracked_state: draw_centroid(frame, tracked_state, visibility == 1)
            # Use the new perspective trajectory drawing function
            if len(self.trajectory_points) > 1:
                # Fading Tail: Truncate history to the last N frames
                history_len = self.config.visualization_config.history_length
                recent_trajectory = self.trajectory_points[-history_len:]
                
                draw_perspective_trajectory(
                    frame,
                    recent_trajectory,
                    base_width=self.config.visualization_config.base_width,
                    min_width=self.config.visualization_config.min_width,
                    alpha=self.config.visualization_config.alpha,
                    smoothing_window=self.config.visualization_config.smoothing_window
                )
            if best_detection: draw_detection_box(frame, best_detection[:4], best_detection[4])
            video_writer.write(frame)

            if self.config.save_sample_frames and frame_count % 50 == 0:
                sample_path = os.path.join(self.results_dir, f"sample_{frame_count}.jpg")
                cv2.imwrite(sample_path, frame)
                logger.info(f"Saved sample frame to {sample_path}")

        cap.release()
        video_writer.release()
        self._save_annotations()
        logger.info(f"Output video saved to {out_video_path}")

        if self.mode == "EVALUATE":
            self._perform_evaluation()

    def _save_annotations(self):
        if not self.annotations: return
        
        header = ["frame_index", "x_centroid", "y_centroid", "visibility_flag"]
        if self.mode == 'INFER':
            header = ["frame_index", "x1", "y1", "x2", "y2", "confidence"]
            
        with open(self.annotations_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.annotations)
        logger.info(f"Annotations saved to {self.annotations_path}")

    def _perform_evaluation(self):
        logger.info("--- Performing Evaluation ---")
        
        # 1. Determine local ground truth path based on video name
        # Assuming video name is "1" from "1.mp4", then GT is "1_annotations.xml"
        video_basename = os.path.splitext(os.path.basename(str(self.config.io_config.input_video_path)))[0]
        gt_filename = f"{video_basename}_annotations.xml"
        
        # Resolve absolute path for GT folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        gt_folder = self.config.evaluation_config.ground_truth_folder
        if not os.path.isabs(gt_folder):
            gt_folder = os.path.join(project_root, gt_folder)
            
        gt_path = os.path.join(gt_folder, gt_filename)
        
        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth file not found: {gt_path}. Skipping evaluation.")
            return

        # 2. Parse Ground Truth
        gt_df = parse_xml_ground_truth(gt_path)

        # 3. Load Predictions (from the just-saved CSV)
        pred_df = load_system_predictions(self.annotations_path)

        # 4. Compute Metrics
        metrics = compute_metrics(
            gt_df=gt_df,
            pred_df=pred_df,
            distance_threshold=self.config.evaluation_config.distance_threshold
        )
        
        if not metrics:
            logger.warning("Evaluation failed or returned no metrics.")
            return

        # 5. Log and Save Results
        formatted_metrics = {
            "total_ground_truth_visible_balls": metrics['total_visible_gt'],
            "detected_balls_matched": metrics['true_positives'],
            "recall": f"{metrics['recall']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "mean_centroid_drift_pixels": f"{metrics['avg_drift']:.2f}" if pd.notna(metrics['avg_drift']) else "N/A"
        }
        
        logger.info(f"Evaluation Results: {json.dumps(formatted_metrics, indent=4)}")

        metrics_path = os.path.join(self.results_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(formatted_metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Cricket Ball Detection and Tracking Pipeline")
    parser.add_argument("--config", type=str, default="../config/track_config.json", help="Path to the pipeline configuration file.")
    parser.add_argument("--ground_truth", type=str, help="Path to the ground truth CSV file for EVALUATE mode.")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Resolve config path relative to project root
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        base_config = PipelineConfig.model_validate(config_dict)

        # Resolve ground truth path if provided
        ground_truth_path = args.ground_truth
        if ground_truth_path and not os.path.isabs(ground_truth_path):
            ground_truth_path = os.path.join(project_root, ground_truth_path)

        if base_config.mode == 'TRAIN':
            logger.info("--- Running in TRAIN mode ---")
            
            # Resolve model path relative to project root
            model_path = base_config.detector_config.model_path
            if model_path and not os.path.isabs(model_path):
                model_path = os.path.join(project_root, str(model_path))

            # Resolve data path relative to project root
            data_path = base_config.train_config.data_path
            if data_path and not os.path.isabs(data_path):
                data_path = os.path.join(project_root, data_path)
            
            # Check if files exist
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path for training not found: {model_path}")
            if not data_path or not os.path.exists(data_path):
                raise FileNotFoundError(f"Data YAML path for training not found: {data_path}")

            train_model(
                model_path=model_path,
                data_path=data_path,
                epochs=base_config.train_config.epochs
            )
            logger.info("--- Finished training ---")
            return

        input_path_str = str(base_config.io_config.input_video_path)
        if not os.path.isabs(input_path_str):
            input_path_str = os.path.join(project_root, input_path_str)

        video_files = []
        if os.path.isdir(input_path_str):
            logger.info(f"Input path is a directory. Searching for .mp4 and .mov files in {input_path_str}")
            mp4_files = glob.glob(os.path.join(input_path_str, '*.mp4'))
            mov_files = glob.glob(os.path.join(input_path_str, '*.mov'))
            video_files.extend(mp4_files)
            video_files.extend(mov_files)
            if not video_files:
                logger.warning("No .mp4 or .mov files found in the specified directory.")
                return
        elif os.path.isfile(input_path_str):
            video_files.append(input_path_str)
        else:
            raise FileNotFoundError(f"Input path {input_path_str} is not a valid file or directory.")

        logger.info(f"Found {len(video_files)} video(s) to process.")

        for video_path in video_files:
            logger.info(f"--- Starting pipeline for: {os.path.basename(video_path)} ---")
            try:
                # Create a deep copy of the config for each run to ensure isolation
                video_config = copy.deepcopy(base_config)
                video_config.io_config.input_video_path = video_path
                
                pipeline = Pipeline(video_config, ground_truth_path)
                pipeline.run()
                logger.info(f"--- Finished pipeline for: {os.path.basename(video_path)} ---")

            except Exception as e:
                logger.error(f"Failed to process {video_path}. Error: {e}", exc_info=True)
                # Continue to the next video
                continue

    except ValidationError as e:
        logger.error(f"Configuration error in '{os.path.basename(config_path)}': {e}")
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
