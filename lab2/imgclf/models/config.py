from ..base.config import BaseConfig, UNIMPLEMENTED

class ModelConfig(BaseConfig):
    """provides defalut configulartion for the models below """

    batch_norm: bool = True

    # probability of an elements to be zeor-ed in dropout layers
    dropout_rate: float = 0.5

    num_class: int = UNIMPLEMENTED
