```python
import torch
import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import Image
import torchvision.transforms as T
import json
import os

class DataAugmentation:
    """Data augmentation for training"""
    def __init__(self, config: Dict):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(config['augmentation']['random_rotation']),
            T.ColorJitter(
                brightness=config['augmentation']['color_jitter']['brightness'],
                contrast=config['augmentation']['color_jitter']['contrast'],
                saturation=config['augmentation']['color_jitter']['saturation'],
                hue=config['augmentation']['color_jitter']['hue']
            ),
            T.Resize(config['augmentation']['resize_size'])
        ])
        
    def __call__(self, image: Image.Image) -> Image.Image:
        return self.transform(image)

def load_and_preprocess_frame(frame_path: str,
                            transform: T.Compose = None) -> torch.Tensor:
    """Load and preprocess a single frame"""
    # Load image
    image = Image.open(frame_path).convert('RGB')
    
    # Apply augmentation if provided
    if transform is not None:
        image = transform(image)
    else:
        image = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])(image)
        
    return image

def create_sequences(data: List[Dict],
                    sequence_length: int,
                    stride: int = 1) -> List[Dict]:
    """Create sequences from video data"""
    sequences = []
    
    for video in data:
        # Get timestamps and velocities
        timestamps = video['sensor']['timestamp']
        velocities = [float(v) for v in video['sensor']['velocity']]
        
        # Create frame paths
        frame_paths = []
        for t in timestamps:
            frame_id = f"frame_{int(float(t)*30):06d}.jpg"
            frame_paths.append(frame_id)
            
        # Create sequences
        for i in range(0, len(frame_paths) - sequence_length + 1, stride):
            sequence = {
                'frames': frame_paths[i:i + sequence_length],
                'velocity': velocities[i:i + sequence_length],
                'video_id': video['video_id'],
                'start_time': video['start time'],
                'timestamps': timestamps[i:i + sequence_length]
            }
            sequences.append(sequence)
            
    return sequences

def save_predictions(predictions: List[float],
                    metadata: Dict,
                    output_path: str):
    """Save predictions with metadata"""
    output = {
        'predictions': predictions,
        'metadata': metadata
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
        
def load_json_data(json_path: str) -> Dict:
    """Load and parse JSON data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def normalize_velocity(velocity: float,
                      min_vel: float = 0.0,
                      max_vel: float = 100.0) -> float:
    """Normalize velocity to [0, 1] range"""
    return (velocity - min_vel) / (max_vel - min_vel)

def denormalize_velocity(normalized: float,
                        min_vel: float = 0.0,
                        max_vel: float = 100.0) -> float:
    """Convert normalized velocity back to original range"""
    return normalized * (max_vel - min_vel) + min_vel
```
