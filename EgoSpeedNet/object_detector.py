
import torch
import torchvision 
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image, UnidentifiedImageError
import time
import cv2
import numpy as np
from loguru import logger
import yaml
from typing import Dict, List, Tuple

class ObjectDetector:
    def __init__(
        self,
        coco_labels_file: str = "../configs/coco_labels.yaml",
        confidence_threshold: float = 0.5
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Mask R-CNN
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights, progress=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold

        # Load category definitions
        self.categories = {
            'car': ['car', 'bus', 'truck'],
            'pedestrian': ['person'], 
            'traffic': ['traffic light', 'stop sign']
        }

        # Max objects per category (from paper)
        self.max_objects = {
            'car': 20,
            'pedestrian': 10,
            'traffic': 10
        }
        
        # Load COCO labels
        with open(coco_labels_file, 'r') as f:
            self.coco_labels = yaml.safe_load(f)

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to tensor and normalize"""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        # Convert to PIL
        frame = Image.fromarray(frame)
        
        # Apply transforms
        frame = self.transform(frame).unsqueeze(0)
        return frame.to(self.device)

    def detect(self, frame: np.ndarray) -> Dict:
        """Detect objects in frame"""
        # Preprocess
        image = self.preprocess_frame(frame)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image)[0]
            
        # Filter by confidence
        keep = predictions['scores'] > self.confidence_threshold
        boxes = predictions['boxes'][keep]
        labels = predictions['labels'][keep] 
        scores = predictions['scores'][keep]

        # Organize by category
        detections = {
            'car': {'boxes': [], 'scores': []},
            'pedestrian': {'boxes': [], 'scores': []},
            'traffic': {'boxes': [], 'scores': []}
        }

        # Sort by confidence and assign to categories
        sorted_idx = torch.argsort(scores, descending=True)
        for idx in sorted_idx:
            label = self.coco_labels[labels[idx].item()]
            
            # Find category
            for category, valid_labels in self.categories.items():
                if label in valid_labels:
                    if len(detections[category]['boxes']) < self.max_objects[category]:
                        detections[category]['boxes'].append(boxes[idx])
                        detections[category]['scores'].append(scores[idx])
                        
        # Convert to tensors
        for category in detections:
            if detections[category]['boxes']:
                detections[category]['boxes'] = torch.stack(detections[category]['boxes'])
                detections[category]['scores'] = torch.stack(detections[category]['scores'])
            else:
                detections[category]['boxes'] = torch.empty((0, 4)).to(self.device)
                detections[category]['scores'] = torch.empty(0).to(self.device)
                
        return detections