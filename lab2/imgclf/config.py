import os
import torch
from torch.optim import Adam
from .models.config import ModelConfig
from .dataset.config import DatasetConfig
from .base.model_utils.config import ModelUtilsConfig

@staticmethod
def adam_lambda(params, config: ModelUtilsConfig):
    return Adam(params, lr=config.learning_rate)

class Config(ModelConfig, DatasetConfig, ModelUtilsConfig):

    ## ------------- ModelUtilsConfig ---------------------------

    device = torch.device("cuda:0")

    learning_rate = 0.001

    optimizer = adam_lambda

    _optimizer_name = "adam"

    # num of epochs per checkpoints
    # e.g. 1 stand for save model every epoch
    #      0 for not save until finish
    epochs_per_checkpoint: int = 0

    # dir for saving checkpoints and log files
    log_dir: str = "log"

    early_stopping: bool = False

    # only matter when EARLY_STOPPING is set to True
    early_stopping_threshold: int = 10

    ## ------------- DatasetConfig ---------------------------

    # config for torch's DataLoader
    num_workers = 4
    persistent_workers = os.name == "nt" and bool(num_workers)

    # batch sizes
    batch_size = {
        "train": 128,
        "eval": 256,
    }
