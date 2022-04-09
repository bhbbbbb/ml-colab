from ..base.config import BaseConfig, UNIMPLEMENTED

class DatasetConfig(BaseConfig):

    num_workers: int = UNIMPLEMENTED
    """config for torch's DataLoader"""

    persistent_workers: bool = UNIMPLEMENTED
    """config for torch's DataLoader"""

    pin_memory: bool = UNIMPLEMENTED
    """config for torch's DataLoader"""


    batch_size = {
        "train": UNIMPLEMENTED,
        "eval": UNIMPLEMENTED,
    }
