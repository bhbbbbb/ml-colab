from typing import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from nfnets import NFNet # pylint: disable=import-error

class MyNfnet(NFNet):
    def __init__(self, **kwargs):
        num_classes = kwargs["num_classes"] 
        kwargs["num_classes"] = 1000
        super().__init__(**kwargs)
        self.fc = nn.Linear(1000, num_classes)
        return

    def forward(self, x):
        output = super().forward(x)
        return self.fc(output)

    @staticmethod
    def fix_output_layer(model_state_dict: OrderedDict, num_class):

        fc_weight = np.random.normal(scale=0.01, size=[num_class, 1000])
        fc_bias = np.random.normal(scale=0.1, size=[num_class])

        model_state_dict["fc.weight"] = torch.tensor(fc_weight, dtype=torch.float32)
        model_state_dict["fc.bias"] = torch.tensor(fc_bias, dtype=torch.float32)
        model_state_dict.move_to_end("fc.weight", last=True)
        model_state_dict.move_to_end("fc.bias", last=True)
        return model_state_dict
