```python
import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(true_velocity: List[float],
                     pred_velocity: List[float]) -> Dict[str, float]:
    """Calculate regression metrics"""
    metrics = {
        'mse': mean_squared_error(true_velocity, pred_velocity),
        'rmse': np.sqrt(mean_squared_error(true_velocity, pred_velocity)),
        'mae': mean_absolute_error(true_velocity, pred_velocity),
        'r2': r2_score(true_velocity, pred_velocity)
    }
    return metrics

def evaluate_model(model: torch.nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[List[float], List[float], Dict[str, float]]:
    """Evaluate model on test set"""
    model.eval()
    true_velocities = []
    pred_velocities = []
    
    with torch.no_grad():
        for batch in test_loader:
            frames = batch['frames'].to(device)
            velocity = batch['velocity'].cpu().numpy()
            
            predictions = model(frames).cpu().numpy()
            
            true_velocities.extend(velocity.flatten())
            pred_velocities.extend(predictions.flatten())
            
    # Calculate metrics
    metrics = calculate_metrics(true_velocities, pred_velocities)
    
    return true_velocities, pred_velocities, metrics

def analyze_predictions(true_velocity: List[float],
                       pred_velocity: List[float]) -> Dict:
    """Detailed analysis of predictions"""
    analysis = {
        'error_stats': {
            'mean_error': np.mean(np.array(true_velocity) - np.array(pred_velocity)),
            'std_error': np.std(np.array(true_velocity) - np.array(pred_velocity)),
            'max_error': np.max(np.abs(np.array(true_velocity) - np.array(pred_velocity))),
        },
        'velocity_stats': {
            'mean_true': np.mean(true_velocity),
            'mean_pred': np.mean(pred_velocity),
            'std_true': np.std(true_velocity),
            'std_pred': np.std(pred_velocity)
        },
        'metrics': calculate_metrics(true_velocity, pred_velocity)
    }
    return analysis
```
