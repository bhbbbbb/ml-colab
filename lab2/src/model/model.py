
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
    
    # TODO
    def summary(self) -> str:
        def prod(arr: list) -> int:
            res = 1
            for a in arr:
                res *= a
            return res
        
        sum = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(prod(param.size()))
            sum += prod(param.size())
        

class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 x 112

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28 x 28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14 x 14

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=14, stride=14),

            nn.Flatten(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, config.NUM_CLASS),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
        