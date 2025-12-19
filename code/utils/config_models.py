from pydantic import BaseModel, Field, FilePath
from typing import Literal, Optional, Dict, Any, List

# Using Field to add descriptions that can be used for documentation
class ModelHyperparameters(BaseModel):
    """Hyperparameters for a detection model."""
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence score for a detection to be considered valid.")
    target_class_id: Optional[int] = Field(None, description="For YOLO models, the specific class ID to target (e.g., 32 for 'sports ball').")

    # Allow extra parameters for different models (like classical detector)
    class Config:
        extra = 'allow'

class IOConfig(BaseModel):
    """Input/Output configuration."""
    input_video_path: str = Field(..., description="Path to the input video file or directory.")
    output_dir: str = Field("output", description="Base directory for saving output files (deprecated in favor of root-level folders).")

class TrackingConfig(BaseModel):
    """Configuration for the Kalman Tracker."""
    max_age: int = Field(30, gt=0, description="Maximum number of frames to predict without a detection before the track is lost.")
    process_noise_cov: Optional[List[float]] = Field([1e-2, 1e-2, 5e-3, 5e-3], description="Process noise covariance (Q) diagonal.")
    measurement_noise_cov: Optional[List[float]] = Field([1e-1, 1e-1], description="Measurement noise covariance (R) diagonal.")
    edge_margin: int = Field(10, ge=0, description="Margin in pixels from the frame edge. Tracks entering this margin are terminated.")

class VisualizationConfig(BaseModel):
    """Configuration for trajectory visualization."""
    base_width: int = Field(50, gt=0, description="Maximum width of the trajectory at the bottom of the frame (max y).")
    min_width: int = Field(5, ge=0, description="Minimum width of the trajectory at the top of the frame (min y).")
    alpha: float = Field(0.3, ge=0.0, le=1.0, description="Transparency level of the trajectory (0.0 to 1.0).")
    smoothing_window: int = Field(5, ge=1, description="Window size for moving average smoothing.")
    history_length: int = Field(30, ge=1, description="Number of past frames to visualize (tail length).")

class EvaluationConfig(BaseModel):
    """Configuration for evaluation mode."""
    distance_threshold: int = Field(10, gt=0, description="Maximum pixel distance for a predicted centroid to match a ground truth centroid.")
    ground_truth_folder: str = Field("data/TEST/Annotations", description="Path to the directory containing ground truth XML files.")

class ModelConfig(BaseModel):
    """Configuration for the detection model."""
    type: Literal['classical', 'yolo'] = Field(..., description="The type of detector to use.")
    model_path: Optional[FilePath] = Field(None, description="Path to the model weights file (e.g., .pt for YOLO).")
    hyperparameters: ModelHyperparameters = Field(default_factory=ModelHyperparameters, description="Model-specific hyperparameters.")


class TrainConfig(BaseModel):
    """Configuration for training mode."""
    epochs: int = Field(100, gt=0, description="Number of training epochs.")
    data_path: str = Field("data/TRAIN/data.yaml", description="Path to the data configuration file.")


class PipelineConfig(BaseModel):
    """Top-level configuration for the entire pipeline."""
    mode: Literal['TRACK', 'EVALUATE', 'TRAIN']
    detector_config: ModelConfig
    io_config: IOConfig
    tracking_config: TrackingConfig = Field(default_factory=TrackingConfig)
    visualization_config: VisualizationConfig = Field(default_factory=VisualizationConfig)
    evaluation_config: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Configuration for evaluation mode.")
    train_config: Optional[TrainConfig] = Field(default_factory=TrainConfig, description="Configuration for training mode.")
    save_sample_frames: bool = False


