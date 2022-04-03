import re
from io import StringIO
import torch.nn as nn
import torch
from .config import ModelConfig


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
    
    def summary(self, file) -> str:
        def prod(arr: list) -> int:
            res = 1
            for a in arr:
                res *= a
            return res
        
        sum_params = 0
        layers_name = "Layer's name"
        sio = StringIO()
        sio.write(f"{layers_name:25}\t{'Size':<30}\t{'Num of params':>12}\n")
        for name, param in self.named_parameters():
            sio.write(f"{name :25}\t{str(param.size()):<30}\t{prod(param.size()):>12}\n")
            sum_params += prod(param.size())
        sio.write("--------------------------------------------------------\n")

        sio.write(f"total params: {sum_params}")
        if int(sum_params / 1e9):
            sio.write(f" = {sum_params / 1e9:.2}G")
        elif int(sum_params / 1e6):
            sio.write(f" = {sum_params / 1e6:.2}M")
        elif int(sum_params / 1e3):
            sio.write(f" = {sum_params / 1e3:.2}K")
        sio.write("\n\n")
        print(sio.getvalue(), file=file)
        sio.close()
        return

class FatLeNet5(_BaseNN):
    def __init__(self, config: ModelConfig,
                batch_norm: bool = None, dropout_rate: float = None):
        super().__init__()
        
        batch_norm = batch_norm or config.batch_norm
        dropout_rate = dropout_rate or config.dropout_rate

        self.seq = nn.Sequential()

        self.append(nn.Conv2d(3, 6, kernel_size=5, stride=1))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU(inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(6))
        self.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.append(nn.Conv2d(6, 16, kernel_size=5, stride=1))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU(inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(16))
        self.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.append(nn.Conv2d(16, 120, kernel_size=53))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU(inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(120))

        self.append(nn.Flatten())

        self.append(nn.Linear(120, 84))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU(inplace=True))

        self.append(nn.Linear(84, config.num_class))
        self.append(nn.Softmax(dim=1))


class FakeVGG16(_BaseNN):
    def __init__(self, model_config: ModelConfig,
                batch_norm: bool = None, dropout_rate: float = None):
        super().__init__()

        batch_norm = batch_norm or model_config.batch_norm
        dropout_rate = dropout_rate or model_config.dropout_rate
        
        self.seq = nn.Sequential()

        self.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding="same"))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(64))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 112 x 112

        self.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(128))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 56 x 56

        self.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(256))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 28 x 28

        self.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding="same"))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(512))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 14 x 14

        self.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"))
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())
        if batch_norm:
            self.append(nn.BatchNorm2d(512))
        self.append(nn.MaxPool2d(kernel_size=14, stride=14))

        self.append(nn.Flatten())

        self.append(nn.Linear(512, 512))
        if dropout_rate:
            self.append(nn.Dropout(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())

        self.append(nn.Linear(512, 128))
        if dropout_rate:
            self.append(nn.Dropout(p=dropout_rate, inplace=True))
        self.append(nn.ReLU())

        self.append(nn.Linear(128, model_config.num_class))
        self.append(nn.Softmax(dim=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
        