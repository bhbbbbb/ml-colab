from ..base.config import BaseConfig, UNIMPLEMENTED

class ModelConfig(BaseConfig):
    """provides defalut configulartion for the models below """

    batch_norm: bool = True

    dropout_rate: float = 0.5
    """probability of an elements to be zeor-ed in dropout layers"""

    conv_dropout_rate: float = dropout_rate
    """probability of an elements to be zeor-ed in conv dropout2d layers"""

    num_class: int = UNIMPLEMENTED
    """num of classes"""
