
import torch.nn as nn
import numpy as np
import torch
from model.config import Config

class FatLeNet5(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 120, kernel_size=53),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, config.NUM_CLASS),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
        