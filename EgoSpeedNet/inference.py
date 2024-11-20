```python
import torch
import cv2
import numpy as np
from typing import List, Dict, Union
import json
import argparse
from pathlib import Path
from loguru import logger
from speed_modules import VelocityPredictor

class VelocityInference:
    def __init__(self,
                 model_path: str,
                 sequence_length: int = 10,
                 device: str = None):
        """
        Initialize velocity inference
        
        Args:
            model_path: Path to trained model checkpoint
            sequence_length: Number of frames to use for prediction
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.sequence_length = sequence_length
        
        # Load model
        self.model = VelocityPredictor().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (224, 224))
        
        # Normalize and convert to tensor
        frame = frame / 255.0
        frame = torch.FloatTensor(frame.transpose(2, 0, 1))
        
        return frame
        
    def predict_sequence(self, frames: List[np.ndarray]) -> float:
        """Predict velocity for a sequence of frames"""
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            processed = self.preprocess_frame(frame)
            processed_frames.append(processed)
            
        # Stack frames
        frames_tensor = torch.stack(processed_frames).unsqueeze(0)
        frames_tensor = frames_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            velocity = self.model(frames_tensor)
            
        return velocity.item()
        
    def process_video(self, 
                     video_path: str,
                     output_path: str = None,
                     save_predictions: bool = True) -> List[float]:
        """
        Process video file and predict velocities
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            save_predictions: Whether to save predictions to JSON
            
        Returns:
            List of predicted velocities
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if needed
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
        # Process frames
        frame_buffer = []
        predictions = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                frame_buffer.append(frame)
                
                # Once we have enough frames
                if len(frame_buffer) >= self.sequence_length:
                    # Get prediction
                    velocity = self.predict_sequence(frame_buffer)
                    predictions.append(velocity)
                    
                    # Add prediction to frame
                    frame_with_text = frame_buffer[-1].copy()
                    cv2.putText(
                        frame_with_text,
                        f"Velocity: {velocity:.2f} m/s",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Save frame
                    if output_path:
                        writer.write(frame_with_text)
                        
                    # Remove oldest frame
                    frame_buffer.pop(0)
                    
                # Log progress
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
                    
        finally:
            cap.release()
            if output_path:
                writer.release()
                
        # Save predictions
        if save_predictions:
            prediction_path = Path(output_path).with_suffix('.json') if output_path else Path(video_path).with_suffix('.json')
            with open(prediction_path, 'w') as f:
                json.dump({
                    'video_path': video_path,
                    'predictions': predictions
                }, f, indent=2)
                
        return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str,
                       help='Path to save output video')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = VelocityInference(
        model_path=args.model,
        device=args.device
    )
    
    # Process video
    predictor.process_video(
        video_path=args.video,
        output_path=args.output
    )

if __name__ == '__main__':
    main()
```
