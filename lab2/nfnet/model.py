import torch.nn as nn
import torch
# from imgclf.nfnets import SGD_AGC, pretrained_nfnet, NFNet # pylint: disable=no-name-in-module
from nfnets import SGD_AGC, pretrained_nfnet, NFNet # pylint: disable=no-name-in-module

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
    def fix_output_layer(model_state_dict, num_class):
        fc_weight = torch.randn([num_class, 1000])
        fc_bias = torch.randn([num_class])
        model_state_dict["fc.weight"] = fc_weight
        model_state_dict["fc.bias"] = fc_bias
        model_state_dict.move_to_end("fc.weight", last=True)
        model_state_dict.move_to_end("fc.bias", last=True)
        return model_state_dict

