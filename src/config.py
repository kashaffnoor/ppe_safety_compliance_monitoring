"""
Configuration file for PPE Detection System
"""

# Model Configuration
MODEL_CONFIG = {
    'model_type': 'yolov8n',  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    'confidence_threshold': 0.5,  # Minimum confidence for detection
    'iou_threshold': 0.45,  # IOU threshold for NMS
    'max_detections': 50,  # Maximum number of detections per frame (reduced for speed)
}

# Required PPE Classes (based on Hard Hat Workers dataset)
# Class mapping: 0=head, 1=helmet, 2=person
REQUIRED_CLASSES = {
    'helmet': {'required': True, 'color': (0, 255, 0)},  # Green when present
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'target_fps': 30,  # Target frames per second
    'display_width': 1280,  # Display width
    'display_height': 720,  # Display height
    'buffer_size': 1,  # Frame buffer size (reduced for speed)
}

# Warning System Configuration
WARNING_CONFIG = {
    'warning_duration': 3,  # How long to show warnings (seconds)
    'alert_sound': True,  # Enable sound alerts for violations
    'log_violations': True,  # Log violations to file
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'image_size': 640,
    'learning_rate': 0.001,
    'patience': 10,  # Early stopping patience
    'device': '0',  # GPU device, 'cpu' for CPU (use GPU for training)
}

# Dataset Configuration
DATASET_CONFIG = {
    'data_yaml': 'data/data.yaml',
    'train_path': 'data/train',
    'valid_path': 'data/valid',
    'test_path': 'data/test',
}

# UI Configuration
UI_CONFIG = {
    'title': 'Safety Compliance Monitoring',
    'refresh_rate': 10,  # UI refresh rate in ms
    'show_fps': True,
    'show_confidence': True,
}
