import os
from imgclf.config import Config

class Hw2Config(Config):

    num_class = 10

    epochs_per_checkpoint: int = 10

    # dir for saving checkpoints
    log_dir: str = os.path.abspath(os.path.join(__file__, "..", "..", "log"))

    early_stopping: bool = True

    # only matter when EARLY_STOPPING is set to True
    early_stopping_threshold: int = 20
