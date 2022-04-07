# required before pythonV3.10
from __future__ import annotations
from typing import Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from .config import DatasetConfig


class Dataset(TorchDataset):
    """Dataset for image classification task"""

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224), InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    
    def __init__(self, df: pd.DataFrame, config: DatasetConfig,
        mode: Literal["train", "eval", "inference"] = "train",
        transform: transforms.Compose = None):
        """
        Args:
            df (pd.DataFrame): for mode `train` and `eval` df should be in the form that
            |   img (str)      |  label (np.integer) |
            |:----------------:|:-----------------:|
            | <path to image1> | <label of image1> |
            | <path to image2> | <label of image2> |
            | &vellip; | &vellip; |

            for mode `inference`, df can only contain `img` column

            mode: Defaults to "train".
        """
        assert mode in ["train", "eval", "inference"], f"unknown type of mode: {mode}"

        assert "img" in df.columns

        if mode != "inference":
            assert np.issubdtype(df.dtypes["label"], np.integer), (
                f"the dtype of 'img' column must to be np.integer or its subdtype,"
                f"got {df.dtypes['label']}"
            )
            df["label"] = df["label"].astype(np.int64)
        
        else:
            assert len(df.columns) == 1, f"Except only a column in df, got {df.columns}"

        self.mode = mode
        self.df = df
        if transform is None:
            self.transform = self.TRAIN_TRANSFORM if mode == "train" else self.EVAL_TRANSFORM
        else:
            self.transform = transform
        
        self.config = config
        return
    
    @classmethod
    def split(
            cls,
            df: pd.DataFrame,
            split_ratio: Tuple[float, float],
            config: DatasetConfig,
            transforms_f: Tuple[transforms.Compose, transforms.Compose] = None,
        ):
        """get dataset by split then with given ratio

        Args:
            df (pd.DataFrame): same as __init__
            split_ratio (tuple): split ratio of dataset for train and validation
                e.g. [0.7, 0.15]
                the sumation of the split_ratio must < 1, and if sum != 1, the rest part
                would be split for test.
        """
        sumation = sum(split_ratio)
        assert sumation <= 1.0,\
            f"the sumation of split_ratio is expected to be <= 1, got {sumation}"

        tem_dataset = cls(df, config=config)
        train_set_size = int(len(tem_dataset) * split_ratio[0])
        valid_set_size = int(len(tem_dataset) * split_ratio[1])
        # test_set_size = len(tem_dataset) - train_set_size - valid_set_size
        seed = 0xAAAAAAAA

        train_df, eval_df = train_test_split(tem_dataset.df, train_size=train_set_size,
                            shuffle=True, random_state=seed)

        valid_df, test_df = train_test_split(eval_df, train_size=valid_set_size,
                            shuffle=True, random_state=seed)
        
        if transforms_f is None: # use default transform
            transforms_f = (None, None)
        return (
            cls(train_df, config=config, mode="train", transform=transforms_f[0]),
            cls(valid_df, config=config, mode="eval", transform=transforms_f[1]),
            cls(test_df, config=config, mode="eval", transform=transforms_f[1]),
        )

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        if self.mode != "inference":
            imgpath, label = self.df.iloc[index]
        else:
            imgpath, = self.df.iloc[index]
        
        img = Image.open(imgpath).convert("RGB")
        img = self.transform(img)

        if self.mode != "inference":
            return img, label

        return img, index
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return self.df.shape[0]

    @property
    def data_loader(self):
        if self.mode == "inference":
            mode = "eval"
        else:
            mode = self.mode
        return DataLoader(
            self,
            batch_size = self.config.batch_size[mode],
            shuffle = self.mode == "train",
            num_workers = self.config.num_workers,
            persistent_workers= self.config.persistent_workers,
        )
