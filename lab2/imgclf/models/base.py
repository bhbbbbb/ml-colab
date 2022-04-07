import re
from io import StringIO
import torch.nn as nn
import torch



class _BaseNN(nn.Module):

    seq: nn.Sequential

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential()
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
    
    def summary(self, file = None) -> str:
        def prod(arr: list) -> int:
            res = 1
            for a in arr:
                res *= a
            return res
        
        sum_params = 0
        layers_name = "Layer's name"
        sio = StringIO()
        sio.write(f"{layers_name:45}\t{'Size':<30}\t{'Num of params':>12}\n")
        for name, param in self.named_parameters():
            sio.write(f"{name :45}\t{str(param.size()):<30}\t{prod(param.size()):>12}\n")
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

class BnDoConv2d(_BaseNN):
    """Batch-norm Dropout Conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        batch_norm: bool = True,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.append(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                                    padding=padding, stride=stride, **kwargs)
        )
        if dropout_rate:
            self.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        if batch_norm:
            self.append(nn.BatchNorm2d(out_channels))
        return

    
    
