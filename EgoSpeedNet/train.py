
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from typing import Dict, List
import json
import os
import cv2
import numpy as np
from loguru import logger
from pathlib import Path


class DrivingDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 10, mode: str = 'train'):
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.mode = mode
        
        # Load data
        json_path = os.path.join(data_dir, f'{mode}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.videos = data['videos']
            
        # Create sequences
        self.sequences = []
        for video in self.videos:
            # Get timestamps and velocities
            timestamps = video['sensor']['timestamp']
            velocities = [float(v) for v in video['sensor']['velocity']]
            
            # Get frame paths
            frame_paths = []
            for t in timestamps:
                if self.mode == 'train':
                    frame_id = f"frame_{int(float(t)*30):06d}.jpg"
                    frame_path = os.path.join(data_dir, 'images', frame_id)
                else:
                    frame_id = f"test{int(float(t)*30):06d}.jpg"
                    frame_path = os.path.join(data_dir, 'kaggle/working/images', frame_id)
                    
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
                    
            # Create sequences with overlap
            for i in range(len(frame_paths) - self.seq_len + 1):
                self.sequences.append({
                    'frames': frame_paths[i:i + self.seq_len],
                    'velocity': velocities[i:i + self.seq_len]
                })
                
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Load and preprocess frames
        frames = []
        for frame_path in sequence['frames']:
            # Read frame
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            frame = cv2.resize(frame, (224, 224))
            frame = frame.transpose(2, 0, 1)  # HWC to CHW
            frame = frame / 255.0  # Normalize
            frames.append(frame)
            
        # Convert to tensors
        frames = torch.FloatTensor(np.array(frames))
        velocity = torch.FloatTensor(sequence['velocity'])
        
        return {
            'frames': frames,
            'velocity': velocity
        }

def train_model(config: Dict):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DrivingDataset(
        data_dir=config['data_dir'],
        sequence_length=config['sequence_length'],
        mode='train'
    )
    
    val_dataset = DrivingDataset(
        data_dir=config['data_dir'],
        sequence_length=config['sequence_length'],
        mode='eval'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = VelocityPredictor(
        hidden_size=config['hidden_size'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_count = 0
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # Move to device
            frames = batch['frames'].to(device)
            velocity = batch['velocity'].to(device)
            
            # Forward pass
            pred_velocity = model(frames)
            loss = criterion(pred_velocity, velocity)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                velocity = batch['velocity'].to(device)
                
                pred_velocity = model(frames)
                loss = criterion(pred_velocity, velocity)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_count = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config['checkpoint_dir'], 'best_model.pt'))
        else:
            early_stopping_count += 1
            
        if early_stopping_count >= config['early_stopping_patience']:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
            
        # Log progress
        logger.info(f'Epoch {epoch + 1}/{config["num_epochs"]} - '
                   f'Train Loss: {train_loss:.6f} - '
                   f'Val Loss: {val_loss:.6f}')
        
if __name__ == '__main__':
    # Load config
    config = {
        # Data
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'sequence_length': 10,
        'batch_size': 32,
        'num_workers': 4,
        
        # Model
        'hidden_size': 32,
        'lstm_hidden': 64,
        'lstm_layers': 2,
        
        # Training
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'max_grad_norm': 1.0,
        'num_epochs': 100,
        'early_stopping_patience': 10,
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train model
    train_model(config)
