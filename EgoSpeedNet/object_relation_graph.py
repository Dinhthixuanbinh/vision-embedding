
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLHeteroGraph
from typing import Dict, List, Tuple
from loguru import logger

class ObjectRelationDetector:
    def __init__(self):
        self.categories = ['car', 'pedestrian', 'traffic']
        
    def build_graphs(self, detections: Dict) -> Tuple[List[DGLHeteroGraph], Dict[str, torch.Tensor]]:
        """Build separate graphs for each category"""
        graphs = []
        features = {}
        
        for category in self.categories:
            boxes = detections[category]['boxes']
            
            if len(boxes) > 0:
                # Build fully connected graph
                num_nodes = boxes.size(0)
                src, dst = [], []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            src.append(i)
                            dst.append(j)
                            
                graph = dgl.graph((src, dst), num_nodes=num_nodes)
                graphs.append(graph)
                features[category] = boxes
            else:
                # Empty graph
                graph = dgl.graph(([],[]), num_nodes=0)
                graphs.append(graph)
                features[category] = torch.empty((0, 4)).to(boxes.device)
                
        return graphs, features

class GCNLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        
    def forward(self, graph: DGLHeteroGraph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            # Normalize adjacency matrix
            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(features.device)
            
            # Set features
            graph.ndata['h'] = features * norm.view(-1, 1)
            
            # Message passing
            graph.update_all(
                fn.copy_u('h', 'm'),
                fn.sum('m', 'h')
            )
            
            # Apply normalization and linear transform
            h = graph.ndata['h'] * norm.view(-1, 1)
            return self.linear(h)

class MultiViewGCN(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int):
        super().__init__()
        
        # Separate GCN for each view
        self.gcns = nn.ModuleDict({
            category: nn.ModuleList([
                GCNLayer(in_feats, hidden_size),
                GCNLayer(hidden_size, hidden_size)
            ]) for category in ['car', 'pedestrian', 'traffic']
        })
        
    def forward(self, graphs: List[DGLHeteroGraph], features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process each view through GCN layers"""
        outputs = {}
        
        for i, category in enumerate(['car', 'pedestrian', 'traffic']):
            h = features[category]
            if len(h) > 0:
                # Apply GCN layers
                for layer in self.gcns[category]:
                    h = F.relu(layer(graphs[i], h))
                outputs[category] = h
            else:
                outputs[category] = h
                
        return outputs

    def pool_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Max pool features from each view"""
        pooled_features = []
        
        for category in features:
            if len(features[category]) > 0:
                pooled = torch.max(features[category], dim=0)[0]
            else:
                pooled = torch.zeros(features[category].size(1)).to(features[category].device)
            pooled_features.append(pooled)
            
        return torch.cat(pooled_features)
