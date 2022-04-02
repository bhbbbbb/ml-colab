import re
import torch.nn as nn
import torch
from .config import Config


class _BaseNN(nn.Module):

    seq: nn.Sequential

    def __init__(self):
        super().__init__()
        return
    
    # Sequential.append is not available util torch 1.11
    def append(self, module: nn.Module):
        PATTERN = r"([^\(].+?)\("
        match = re.match(PATTERN, str(module))
        assert match
        name = match.group(1) + "_" + str(len(self.seq))
        self.seq.add_module(name, module)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
    
    def summary(self) -> str:
        def prod(arr: list) -> int:
            res = 1
            for a in arr:
                res *= a
            return res
        
        sum_params = 0
        layers_name = "Layer's name"
        print(f"{layers_name:25}\t{'Size':<30}\t{'Num of params':>12}")
        for name, param in self.named_parameters():
            print(f"{name :25}\t{str(param.size()):<30}\t{prod(param.size()):>12}")
            sum_params += prod(param.size())
        print("--------------------------------------------------------")

        print(f"total params: {sum_params}", end="")
        if int(sum_params / 1e9):
            print(f" = {sum_params / 1e9:.2}G")
        elif int(sum_params / 1e6):
            print(f" = {sum_params / 1e6:.2}M")
        elif int(sum_params / 1e3):
            print(f" = {sum_params / 1e3:.2}K")
        print("")
        

class FatLeNet5(_BaseNN):
    def __init__(self, config: Config, batch_norm: bool = True, dropout: bool = True):
        super().__init__()
        self.seq = nn.Sequential()

        self.append(nn.Conv2d(3, 6, kernel_size=5, stride=1))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU(inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(6))
        self.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.append(nn.Conv2d(6, 16, kernel_size=5, stride=1))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU(inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(16))
        self.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.append(nn.Conv2d(16, 120, kernel_size=53))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU(inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(120))

        self.append(nn.Flatten())

        self.append(nn.Linear(120, 84))
        if dropout:
            self.append(nn.Dropout(inplace=True))
        self.append(nn.ReLU(inplace=True))

        self.append(nn.Linear(84, config.NUM_CLASS))
        self.append(nn.Softmax(dim=1))


class FakeVGG16(_BaseNN):
    def __init__(self, config: Config, batch_norm: bool = True, dropout: bool = True):
        super().__init__()
        
        self.seq = nn.Sequential()

        self.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding="same"))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(64))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 112 x 112

        self.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(128))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 56 x 56

        self.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(256))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 28 x 28

        self.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding="same"))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(512))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 14 x 14

        self.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"))
        if dropout:
            self.append(nn.Dropout2d(inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(512))
        self.append(nn.MaxPool2d(kernel_size=14, stride=14))

        self.append(nn.Flatten())

        self.append(nn.Linear(512, 512))
        if dropout:
            self.append(nn.Dropout(inplace=True))
        self.append(nn.ReLU())

        self.append(nn.Linear(512, 128))
        if dropout:
            self.append(nn.Dropout(inplace=True))
        self.append(nn.ReLU())

        self.append(nn.Linear(128, config.NUM_CLASS))
        self.append(nn.Softmax(dim=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
        
