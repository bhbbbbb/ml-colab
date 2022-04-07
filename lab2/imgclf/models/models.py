import torch.nn as nn
from .config import ModelConfig
from .base import _BaseNN, BnDoConv2d

class FatLeNet5(_BaseNN):
    def __init__(self, config: ModelConfig, conv_dropout_rate: float = None,
                batch_norm: bool = None, dropout_rate: float = None):
        super().__init__()
        
        batch_norm = batch_norm if batch_norm is not None else config.batch_norm
        conv_dropout_rate = conv_dropout_rate\
                        if conv_dropout_rate is not None else config.conv_dropout_rate
        dropout_rate = dropout_rate\
                        if dropout_rate is not None else config.dropout_rate

        self.append(BnDoConv2d(3, 6, kernel_size=5, stride=1,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.append(BnDoConv2d(6, 16, kernel_size=5, stride=1,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.append(BnDoConv2d(16, 120, kernel_size=53,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))

        self.append(nn.Flatten())

        if dropout_rate:
            self.append(nn.Dropout(p=dropout_rate, inplace=True))
        
        self.append(nn.Linear(120, 84))
        self.append(nn.ReLU(inplace=True))

        self.append(nn.Linear(84, config.num_class))


class FakeVGG16(_BaseNN):
    def __init__(self, model_config: ModelConfig, conv_dropout_rate: float = None,
                batch_norm: bool = None, dropout_rate: float = None):
        super().__init__()

        batch_norm = batch_norm if batch_norm is not None else model_config.batch_norm
        conv_dropout_rate = conv_dropout_rate\
                        if conv_dropout_rate is not None else model_config.conv_dropout_rate
        dropout_rate = dropout_rate\
                        if dropout_rate is not None else model_config.dropout_rate
        
        self.append(BnDoConv2d(3, 64, kernel_size=3, stride=1, padding="same",
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 112 x 112

        self.append(BnDoConv2d(64, 128, kernel_size=3, stride=1, padding="same",
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 56 x 56

        self.append(BnDoConv2d(128, 256, kernel_size=3, stride=1, padding="same",
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 28 x 28

        self.append(BnDoConv2d(256, 512, kernel_size=3, stride=1, padding="same",
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 14 x 14

        self.append(BnDoConv2d(512, 512, kernel_size=3, stride=1, padding="same",
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=5, stride=3)) # 4 x 4

        self.append(nn.Flatten())
        if dropout_rate:
            self.append(nn.Dropout(p=dropout_rate, inplace=True))
        
        self.append(nn.Linear(4 * 4 * 512, 4096))
        self.append(nn.ReLU(inplace=True))
        if dropout_rate:
            self.append(nn.Dropout(p=dropout_rate, inplace=True))

        self.append(nn.Linear(4096, 1024))
        self.append(nn.ReLU(inplace=True))
        if dropout_rate:
            self.append(nn.Dropout(p=dropout_rate, inplace=True))

        self.append(nn.Linear(1024, model_config.num_class))
        return
        

class AlexNet(_BaseNN):

    def __init__(self, config: ModelConfig, conv_dropout_rate: float = None,
                    dropout_rate: float = None, batch_norm: bool = None):
        
        super().__init__()
        batch_norm = batch_norm if batch_norm is not None else config.batch_norm
        dropout_rate = dropout_rate\
                        if dropout_rate is not None else config.dropout_rate
        conv_dropout_rate = conv_dropout_rate\
                        if conv_dropout_rate is not None else config.conv_dropout_rate
        
        # 224 x 224
        self.append(BnDoConv2d(3, 64, kernel_size=11, padding=2, stride=4,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        # 55 x 55
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        # 27 x 27
        self.append(BnDoConv2d(64, 192, kernel_size=5, padding=2, stride=1,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        # 27 x 27
        self.append(nn.ReLU(inplace=True))
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        # 13 x 13
        self.append(BnDoConv2d(192, 384, kernel_size=3, stride=1, padding=1,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        # 13 x 13
        self.append(BnDoConv2d(384, 256, kernel_size=3, stride=1, padding=1,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        # 13 x 13
        self.append(BnDoConv2d(256, 256, kernel_size=3, stride=1, padding=1,
                                batch_norm=batch_norm, dropout_rate=conv_dropout_rate))
        self.append(nn.ReLU(inplace=True))
        # 13 x 13
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        # 6 x 6

        self.append(nn.Flatten())
        self.append(nn.Dropout(p=dropout_rate))

        self.append(nn.Linear(6 * 6 * 256, 4096))
        self.append(nn.Dropout(p=dropout_rate))
        self.append(nn.ReLU(inplace=True))

        self.append(nn.Linear(4096, 1024))
        self.append(nn.ReLU(inplace=True))
 
        self.append(nn.Linear(1024, config.num_class))
        return
