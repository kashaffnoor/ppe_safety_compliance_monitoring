"""
Real-time PPE Detection Pipeline using OpenCV and YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import sys

class PPEDetector:
    def __init__(self, model_path, config):
        """
        Initialize PPE detector
        
        Args:
            model_path: Path to trained YOLO model
            config: Configuration dictionary
        """
        self.model = YOLO(model_path)
        self.config = config
        self.model_config = config['MODEL_CONFIG']
        self.required_classes = config['REQUIRED_CLASSES']
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'violations': 0,
            'compliant_detections': 0
        }
        
        # Class name mapping (COCO dataset classes for pre-trained model)
        # When using custom trained model, this should match data.yaml
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        
        # For custom PPE model, override with PPE classes
        # Check if this is a custom trained model by looking at model classes
        if hasattr(self.model, 'names') and len(self.model.names) <= 6:
            self.class_names = self.model.names
            # Required class IDs for compliance (PPE classes)
            # Hard Hat Workers dataset: 0=head, 1=helmet, 2=person
            self.required_class_ids = [1]  # helmet is required
        else:
            # Using pre-trained COCO model - can't do PPE compliance checking
            self.required_class_ids = []
        
    def process_frame(self, frame):
        """
        Process a single frame for PPE detection - Optimized for speed
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            annotated_frame: Frame with bounding boxes
            frame_stats: Statistics for this frame
        """
        # Run inference with optimizations
        results = self.model(
            frame,
            conf=self.model_config['confidence_threshold'],
            iou=self.model_config['iou_threshold'],
            max_det=self.model_config['max_detections'],
            verbose=False,
            imgsz=640,  # Fixed input size for faster inference
            half=False,  # Disable half precision for CPU compatibility
            device='cpu'  # Explicitly use CPU
        )
        
        # Initialize frame statistics
        frame_stats = {
            'people_detected': 0,
            'violations': 0,
            'compliant': 0,
            'missing_gear': defaultdict(list)
        }
        
        # Process detections
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Track statistics
                    self.stats['total_detections'] += 1
                    
                    # Determine if this is a person/worker
                    if class_id in [0, 3]:  # person or worker
                        frame_stats['people_detected'] += 1
                        
                        # Check for required gear in this frame
                        # We need to check if required classes are detected
                        # For simplicity, we'll mark violations based on overall frame analysis
                        
                    # Draw bounding box
                    color = self._get_box_color(class_id, confidence)
                    label = f"{self.class_names.get(class_id, f'class_{class_id}')}: {confidence:.2f}"
                    
                    cv2.rectangle(annotated_frame, 
                                 (int(x1), int(y1)), 
                                 (int(x2), int(y2)), 
                                 color, 2)
                    
                    cv2.putText(annotated_frame, label,
                               (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Analyze compliance for this frame
        frame_stats = self._analyze_compliance(results, frame_stats)
        
        # Update global statistics
        self.stats['total_frames'] += 1
        self.stats['violations'] += frame_stats['violations']
        self.stats['compliant_detections'] += frame_stats['compliant']
        
        return annotated_frame, frame_stats
    
    def _get_box_color(self, class_id, confidence):
        """Get color for bounding box based on class and confidence"""
        # Green for required PPE, Blue for people, Red for violations
        if self.required_class_ids and class_id in self.required_class_ids:
            return (0, 255, 0)  # Green for required gear
        elif class_id == 0:  # person
            return (255, 255, 0)  # Yellow for people
        else:
            return (0, 255, 255)  # Cyan for other objects
    
    def _analyze_compliance(self, results, frame_stats):
        """Analyze compliance based on detections"""
        detected_classes = set()
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    detected_classes.add(class_id)
        
        # Only do compliance checking if we have required classes defined (custom PPE model)
        if self.required_class_ids:
            # For Hard Hat Workers dataset:
            # Class 0: head, Class 1: helmet, Class 2: person
            # Violation: we detect people/heads but no helmets
            # Compliant: we detect helmets
            
            has_helmet = 1 in detected_classes
            has_person_or_head = 2 in detected_classes or 0 in detected_classes
            
            # Count heads without helmets
            heads_without_helmets = 0
            if 0 in detected_classes and not has_helmet:
                heads_without_helmets = 1
            
            if has_person_or_head and not has_helmet:
                frame_stats['violations'] = 1
                frame_stats['missing_gear'] = ['helmet']
            elif has_helmet:
                frame_stats['compliant'] = 1
        else:
            # Using pre-trained COCO model - just mark as compliant if we have detections
            if detected_classes:
                frame_stats['compliant'] = 1
        
        return frame_stats
    
    def get_statistics(self):
        """Get overall detection statistics"""
        return self.stats
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'violations': 0,
            'compliant_detections': 0
        }

class VideoProcessor:
    def __init__(self, detector, config):
        """Initialize video processor"""
        self.detector = detector
        self.config = config
        self.video_config = config['VIDEO_CONFIG']
    
    def process_video(self, video_path):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_config['display_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_config['display_height'])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, frame_stats = self.detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('PPE Detection', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
