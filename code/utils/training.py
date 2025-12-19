import os
from ultralytics import YOLO

def train_model(model_path, data_path, epochs):
    """
    Train a YOLO model.

    Args:
        model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
        data_path (str): Path to the data configuration file (e.g., 'data.yaml').
        epochs (int): Number of training epochs.
    """
    # Load a model
    model = YOLO(model_path)

    # Train the model
    # imgsz=640 is standard for YOLOv8
    model.train(data=data_path, epochs=epochs, imgsz=640)
