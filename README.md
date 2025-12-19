# Cricket Ball Detection and Tracking Pipeline

This project implements a robust pipeline for detecting and tracking cricket balls in video footage. It supports both **Classical Computer Vision** (background subtraction, contour filtering) and **Deep Learning** (YOLOv8) approaches.

## Features

*   **Hybrid Tracking**: Kalman Filter-based tracking for smoothing trajectories and handling occlusions.
*   **Dual Detectors**: 
    *   **Classical**: Tuned for static camera setups using background subtraction.
    *   **YOLO**: Custom trained YOLOv8 model for robust detection in diverse conditions.
*   **Advanced Visualization**: Spotlight-style trajectory visualization with smoothing and fading tails.
*   **Integrated Evaluation**: Automated evaluation against ground truth XML annotations.

## Prerequisites

*   Python 3.8+
*   pip

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/EswarBalla/EdgeFleet-AI-ML-Assessment.git
    cd EdgeFleet-AI-ML-Assessment
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `ultralytics`, `opencv-python`, `pandas`, `numpy`, and `pydantic` are installed)*

## Execution Modes

The project is designed to run in 4 distinct modes using the unified `run.py` script.

### 1. Classical Tracking (Test Videos)
Run the pipeline using the classical computer vision detector. Best for static backgrounds.

```bash
python3 code/run.py --config config/tuned_classical_config.json
```
*   **Input**: Defined in `config/tuned_classical_config.json` (default: `data/TEST`)
*   **Output**: Saved to `results/classical/<video_name>_<timestamp>/`

### 2. YOLO Model Training
Train a custom YOLOv8 model on your dataset.

```bash
python3 code/run.py --config config/train_yolo_config.json
```
*   **Config**: `config/train_yolo_config.json` defines epochs and data path.
*   **Data**: Expects YOLO-format dataset at `data/TRAIN/data.yaml`.
*   **Output**: Trained models saved to `runs/detect/train/`.

### 3. YOLO Tracking
Run the pipeline using your trained YOLO model for detection.

```bash
python3 code/run.py --config config/track_yolo_config.json
```
*   **Model**: Uses the model path specified in the config (e.g., `runs/detect/train/weights/best.pt`).
*   **Output**: Saved to `results/yolo/<video_name>_<timestamp>/`

### 4. Evaluation
Evaluate the pipeline's performance against ground truth annotations (CVAT XML format).

```bash
python3 code/run.py --config config/evaluate_config.json
```
*   **Logic**: 
    1.  Runs tracking on the input video.
    2.  Automatically finds the corresponding Ground Truth XML in `data/TEST/Annotations`.
    3.  Computes metrics (Recall, Precision, Drift).
*   **Output**: `evaluation_metrics.json` saved in the results directory.

## Project Structure

```
.
├── code/
│   ├── run.py                  # Main entry point
│   ├── detectors/              # Detector implementations (YOLO, Classical)
│   ├── tracking/               # Kalman Filter implementation
│   └── utils/                  # Utility modules (Training, Evaluation, Visualization, Config)
├── config/                     # JSON Configuration files for each mode
├── data/                       # Dataset (TRAIN, TEST, Annotations)
└── results/                    # Output videos and metrics
```
