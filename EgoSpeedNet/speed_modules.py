
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import os
from loguru import logger

from object_detector import ObjectDetector
from object_relation_graph import ObjectRelationDetector, MultiViewGCN
from object_spatial_feature_extractor import ParallelSpatialTemporalLSTM

class VelocityPredictor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
    ):
        super().__init__()
        
        # Object detection
        self.object_detector = ObjectDetector()
        
        # Object relation graphs
        self.relation_detector = ObjectRelationDetector()
        self.multi_view_gcn = MultiViewGCN(
            in_feats=4,  # bounding box coordinates
            hidden_size=hidden_size
        )
        
        # Temporal features
        self.temporal_extractor = ParallelSpatialTemporalLSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers
        )
        
        # Velocity prediction
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size//2, 1)  # Single velocity value
        )
        
    def process_frame(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single frame"""
        # Detect objects
        detections = self.object_detector.detect(frame)
        
        # Build graphs
        graphs, features = self.relation_detector.build_graphs(detections)
        
        # Extract spatial features
        spatial_features = self.multi_view_gcn(graphs, features)
        
        # Pool features
        pooled_features = self.multi_view_gcn.pool_features(spatial_features)
        
        return pooled_features
        
    def forward(self, frames: torch.Tensor, h0_c0: Optional[Dict] = None) -> torch.Tensor:
        """
        Process sequence of frames to predict velocity
        
        Args:
            frames: Tensor of shape (batch_size, seq_len, channels, height, width)
            h0_c0: Optional initial LSTM states
            
        Returns:
            Predicted velocities of shape (batch_size, seq_len)
        """
        batch_size = frames.size(0)
        seq_len = frames.size(1)
        
        # Process each frame
        spatial_features = []
        for t in range(seq_len):
            frame_features = []
            for b in range(batch_size):
                frame = frames[b,t]
                features = self.process_frame(frame)
                frame_features.append(features)
                
            spatial_features.append(torch.stack(frame_features))
            
        # Convert to sequences
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Extract temporal features
        temporal_features = self.temporal_extractor(spatial_features, h0_c0)
        
        # Predict velocity
        velocity = self.regressor(temporal_features)
        
        return velocity.squeeze(-1)  # Remove last dimension
        
    def train_step(self, batch: Dict[str, torch.Tensor], criterion: nn.Module, 
                  optimizer: torch.optim.Optimizer) -> float:
        """Single training step"""
        frames = batch['frames'] 
        target_velocity = batch['velocity']
        
        # Forward pass
        predicted_velocity = self(frames)
        loss = criterion(predicted_velocity, target_velocity)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
