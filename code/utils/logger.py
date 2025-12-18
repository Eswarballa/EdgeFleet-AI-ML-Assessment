import logging
import sys

def setup_logger():
    """Sets up a shared logger for the project."""
    logger = logging.getLogger("CricketBallTracker")
    logger.setLevel(logging.DEBUG) # Set the lowest level to capture all messages

    # Prevent adding handlers multiple times if this function is called more than once
    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO) # Only show INFO and above on console
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()

def add_file_handler(log_path):
    """Dynamically adds a file handler to the existing logger."""
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG) # Log everything to the file
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the file handler
    logging.getLogger("CricketBallTracker").addHandler(handler)
