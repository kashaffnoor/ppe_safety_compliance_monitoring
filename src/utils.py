"""
Utility functions for PPE Detection System
"""

import cv2
import numpy as np
from pathlib import Path
import json

def load_image(image_path):
    """Load an image from file"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image

def save_image(image, output_path):
    """Save an image to file"""
    cv2.imwrite(str(output_path), image)

def draw_bounding_box(image, box, label, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on an image"""
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # Draw label
    if label:
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def resize_image(image, target_size=(640, 640)):
    """Resize an image to target size"""
    return cv2.resize(image, target_size)

def convert_bgr_to_rgb(image):
    """Convert BGR image to RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)
