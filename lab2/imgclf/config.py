import os
import torch
from .base import BaseConfig
from .models import ModelConfig

class Config(BaseConfig):

    # device
    DEVICE = torch.device("cuda:0")

    # config for torch's DataLoader
    NUM_WORKERS = 4
    PERSISTENT_WORKERS = os.name == "nt" and NUM_WORKERS

    # batch sizes
    BATCH_SIZE = {
        "train": 128,
        "eval": 256,
    }

    # IMAGE_SHAPE = (224, 224)

    LEARNING_RATE = 0.001


    # num of epochs per checkpoints
    # e.g. 1 stand for save model every epoch
    #      0 for not save until finish
    EPOCHS_PER_CHECKPOINT: int = 0

    # dir for saving checkpoints and log files
    LOG_DIR: str = "log"

    EARLY_STOPPING: bool = False

    # only matter when EARLY_STOPPING is set to True
    EARLY_STOPPING_THRESHOLD: int = 10

    MODEL_CONFIG: ModelConfig = ModelConfig()


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
