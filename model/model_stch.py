"""
A simple FC Pareto Set model.
"""

import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_node = 1024 
        
        self.fc1 = nn.Linear(self.n_obj, self.n_node)
        self.fc2 = nn.Linear(self.n_node, self.n_node)
        self.fc3 = nn.Linear(self.n_node, self.n_dim)
       
    def forward(self, pref):

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = torch.sigmoid(x / 1) 
        
        return x.to(torch.float64)
