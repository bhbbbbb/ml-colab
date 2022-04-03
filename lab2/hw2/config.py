import os
from imgclf.config import Config, ModelConfig

class Hw2Config(Config):

    NUM_CLASS = 10

    EPOCHS_PER_CHECKPOINT: int = 10

    # dir for saving checkpoints
    LOG_DIR: str = os.path.abspath(os.path.join(__file__, "..", "..", "log"))

    EARLY_STOPPING: bool = True

    # only matter when EARLY_STOPPING is set to True
    EARLY_STOPPING_THRESHOLD: int = 20

    MODEL_CONFIG = ModelConfig(num_class = 10)
