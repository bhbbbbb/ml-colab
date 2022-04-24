import os
from torchvision import transforms as T
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torch import Generator
from model_utils.base.config import BaseConfig
from .mode import Mode as M

SEED = 0xAAAAAAA

class DatasetConfig(BaseConfig):

    batch_size = {
        M.TRAIN: 128,
        M.EVAL: 128,
    }

    pin_memory = True
    num_workers = 4 if os.name == "nt" else 2
    
    @property
    def persistent_workers(self):
        return self.num_workers > 0 and os.name == "nt"

IMG_TRANSFORM = T.Compose([
    T.ToTensor()
])

class Dataset(MNIST):
    mode: M
    config: DatasetConfig

    def __init__(self, mode: M, config: DatasetConfig):
        super().__init__(
            root="./data/MNIST",
            download=True,
            train=(mode is M.TRAIN),
            transform=IMG_TRANSFORM,
        )
        self.config = config
        self.mode = mode
        return

    @property
    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.config.batch_size[self.mode],
            shuffle=True,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            generator=Generator().manual_seed(SEED),
        )

class DatasetCIFAR10(CIFAR10):
    mode: M
    config: DatasetConfig

    def __init__(self, mode: M, config: DatasetConfig):
        super().__init__(
            root="./data/cifar10",
            download=True,
            train=(mode is M.TRAIN),
            transform=IMG_TRANSFORM,
        )
        self.config = config
        self.mode = mode
    
        return
    
    @property
    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.config.batch_size[self.mode],
            shuffle=True,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            generator=Generator().manual_seed(SEED),
        )
