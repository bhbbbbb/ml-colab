from model_utils.base.config import UNIMPLEMENTED, register_checking_hook
from .dataset import DatasetConfig
from .model_utils import ModelUtilsConfig
from .model import ModelConfig
from .mode import Mode as M

class Config(ModelUtilsConfig, DatasetConfig, ModelConfig):

    device = "cuda:0"
    """Device to use, cpu or gpu"""

    learning_rate: float = UNIMPLEMENTED

    scale_learning_rate: bool = True

    learning_rate_before_scale: float

    @register_checking_hook
    def do_scale(self):
        if self.scale_learning_rate:
            self.learning_rate_before_scale = self.learning_rate
            self.learning_rate *= self.batch_size[M.TRAIN] / 256

    epochs_per_checkpoint: int = 10
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    save_best: bool = False
    """set True to save every time when the model reach highest val_acc"""

    log_dir: str = "log"
    """dir for saving checkpoints and log files"""

    logging: bool = True
    """whether log to log.log. It's useful to turn this off when inference"""

    early_stopping: bool = False
    """whether enable early stopping"""

    early_stopping_threshold: int = 25
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    # early_stopping_by_acc: bool = False
    # """
    # Early stopping with valid_acc as criterion, cannot be true if enable_accuracy is False.
    # Turn off to use valid_loss as criterion.
    # """

    # enable_accuracy: bool = False
    # """Whether enable logging accuracy in history. Turn off to use loss only."""

    @register_checking_hook
    def fix_input_shape(self):
        if self.preprocessing:
            self.input_shape = self.input_shape[0], self.n_clusters
        return
