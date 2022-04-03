from ..base.config import BaseConfig, UNIMPLEMENTED

class DatasetConfig(BaseConfig):

    # config for torch's DataLoader
    num_workers: int = UNIMPLEMENTED
    persistent_workers: bool = UNIMPLEMENTED

    # batch sizes
    batch_size = {
        "train": UNIMPLEMENTED,
        "eval": UNIMPLEMENTED,
    }
