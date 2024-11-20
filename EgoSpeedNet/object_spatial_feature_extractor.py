
import torch
import torch.nn as nn
from typing import Dict, Optional

class ParallelSpatialTemporalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        
        # Separate LSTM for each category
        self.lstms = nn.ModuleDict({
            category: nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            ) for category in ['car', 'pedestrian', 'traffic']
        })
        
    def forward(self, features_seq: Dict[str, torch.Tensor], 
                h0_c0: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Args:
            features_seq: Dictionary of sequences {category: tensor of shape (batch, seq_len, input_size)}
            h0_c0: Optional initial hidden states
            
        Returns:
            Concatenated features from all LSTMs
        """
        outputs = []
        
        for category in ['car', 'pedestrian', 'traffic']:
            # Get feature sequence
            seq = features_seq[category]
            
            # Get initial states if provided
            if h0_c0 is not None:
                h0, c0 = h0_c0[category]
                out, _ = self.lstms[category](seq, (h0, c0))
            else:
                out, _ = self.lstms[category](seq)
                
            # Take final hidden state
            outputs.append(out[:, -1, :])  # shape: (batch, hidden_size)
            
        # Concatenate all features
        return torch.cat(outputs, dim=1)  # shape: (batch, 3*hidden_size)
