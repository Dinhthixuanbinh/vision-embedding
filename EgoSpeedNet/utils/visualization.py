```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import torch

def plot_velocity_predictions(true_velocity: List[float], 
                            pred_velocity: List[float],
                            save_path: str = None):
    """Plot true vs predicted velocities"""
    plt.figure(figsize=(12, 6))
    plt.plot(true_velocity, label='True Velocity', alpha=0.7)
    plt.plot(pred_velocity, label='Predicted Velocity', alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Velocity (m/s)')
    plt.title('True vs Predicted Velocity')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def draw_object_detections(frame: np.ndarray,
                         detections: Dict[str, Dict[str, torch.Tensor]],
                         colors: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
    """Draw detected objects with their categories"""
    if colors is None:
        colors = {
            'car': (0, 255, 0),      # Green
            'pedestrian': (0, 0, 255),  # Red
            'traffic': (255, 0, 0)    # Blue
        }
        
    frame = frame.copy()
    
    for category, objects in detections.items():
        boxes = objects['boxes']
        scores = objects['scores']
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[category], 2)
            
            # Draw label
            label = f"{category}: {score:.2f}"
            cv2.putText(frame,
                       label,
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       colors[category],
                       2)
            
    return frame

def visualize_spatial_features(features: Dict[str, torch.Tensor],
                             save_path: str = None):
    """Visualize spatial features from each category"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (category, feats) in enumerate(features.items()):
        # Convert to numpy
        feats = feats.cpu().numpy()
        
        # Plot heatmap
        sns.heatmap(feats.T, 
                   ax=axes[i],
                   cmap='YlOrRd',
                   xticklabels=False)
        axes[i].set_title(f'{category} Features')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        save_path: str = None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
```
