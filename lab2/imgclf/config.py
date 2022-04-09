import os
import torch
from .models.config import ModelConfig
from .dataset.config import DatasetConfig
from .base.model_utils.config import ModelUtilsConfig


class Config(ModelConfig, DatasetConfig, ModelUtilsConfig):

    ### ModelConfig
    batch_norm: bool = True

    dropout_rate: float = 0.5
    """probability of an elements to be zeor-ed in dropout layers"""

    conv_dropout_rate: float = dropout_rate
    """probability of an elements to be zeor-ed in conv dropout2d layers"""


    ## ------------- DatasetConfig ---------------------------

    num_workers: int = 4
    """config for torch's DataLoader"""

    persistent_workers: bool = os.name == "nt" and bool(num_workers)
    """config for torch's DataLoader"""

    pin_memory: bool = True
    """config for torch's DataLoader"""

    batch_size = {
        "train": 128,
        "eval": 256,
    }


    ## ------------ ModelUtilsConfig ----------------------

    device = torch.device("cuda:0")
    """Device to use, cpu or gpu"""

    learning_rate: float = 0.001

    epochs_per_checkpoint: int = 0
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    save_best: bool = False
    """set True to save every time when the model reach highest val_acc"""

    log_dir: str = "log"
    """dir for saving checkpoints and log files"""

    early_stopping: bool = False
    """whether enable early stopping"""

    early_stopping_threshold: int = 10
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""
