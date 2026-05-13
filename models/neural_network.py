import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class NeurNet(nn.Module):
    def __init__(self, input_dim):
        super(NeurNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)