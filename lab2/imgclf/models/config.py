from ..base.config import BaseConfig, UNIMPLEMENTED

class ModelConfig(BaseConfig):
    """provides defalut configulartion for the models below """

    batch_norm: bool = True

    # probability of an elements to be zeor-ed in dropout layers
    dropout_rate: float = 0.5

    # probability of an elements to be zeor-ed in dropout layers
    conv_dropout_rate: float = dropout_rate

    num_class: int = UNIMPLEMENTED
