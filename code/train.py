import os
from ultralytics import YOLO
import torch

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
    model.train(data=data_path, epochs=epochs, imgsz=640)

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the paths to the model and data files
    model_path = os.path.join(script_dir, '..', 'models', 'yolov8n.pt')
    data_path = os.path.join(script_dir, '..', 'data', 'TRAIN', 'data.yaml')
    
    # Define the number of epochs
    epochs = 100

    # Train the model
    train_model(model_path, data_path, epochs)
