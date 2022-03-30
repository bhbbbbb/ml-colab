# required before pythonV3.10
from __future__ import annotations
from typing import Literal
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from .config import Config


class Dataset(TorchDataset):
    """Dataset for image classification task"""

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    def __init__(self, df: pd.DataFrame, config: Config,
        mode: Literal["train", "test"] = "train", transform: transforms.Compose = TRAIN_TRANSFORM):
        """
        Args:
            df (pd.DataFrame): in the form that
            |   img (str)      |  label (np.integer) |
            |:----------------:|:-----------------:|
            | <path to image1> | <label of image1> |
            | <path to image2> | <label of image2> |
            | &vellip; | &vellip; |

            mode (Literal[&quot;train&quot;, &quot;eval&quot;], optional): Defaults to "train".
        """
        assert mode in ["train", "eval"], f"unknown type of mode: {mode}"
        # assert df.dtypes["img"] == str, f"the dtype of 'img' column must to be str, got {df.dtypes['img']}"
        assert np.issubdtype(df.dtypes["label"], np.integer), f"the dtype of 'img' column must to be np.integer or its subdtype, got {df.dtypes['label']}"
        df["label"] = df["label"].astype(np.int64)

        self.mode = mode
        self.df = df
        self.transform = transform
        self.config = config
        return
    
    @classmethod
    def split(cls, df: pd.DataFrame, split_ratio: Tuple[float, float],
            transforms: Tuple[transforms.Compose, transforms.Compose], config: Config):
        """get dataset by split then with given ratio

        Args:
            df (pd.DataFrame): same as __init__
            split_ratio (tuple): split ratio of dataset for train and validation
                e.g. [0.7, 0.15]
                the sumation of the split_ratio must < 1, and if sum != 1, the rest part
                would be split for test.
        """
        sumation = sum(split_ratio)
        assert sumation <= 1.0, f"the sumation of split_ratio is expected to be <= 1, got {sumation}"

        tem_dataset = cls(df, config=config)
        train_set_size = int(len(tem_dataset) * split_ratio[0])
        valid_set_size = int(len(tem_dataset) * split_ratio[1])
        # test_set_size = len(tem_dataset) - train_set_size - valid_set_size
        seed = 0xAAAAAAAA

        train_df, eval_df = train_test_split(tem_dataset.df, train_size=train_set_size,
                            shuffle=True, random_state=seed)

        valid_df, test_df = train_test_split(eval_df, train_size=valid_set_size,
                            shuffle=True, random_state=seed)
        
        return (
            cls(train_df, config=config, mode="train", transform=transforms[0]),
            cls(valid_df, config=config, mode="eval", transform=transforms[1]),
            cls(test_df, config=config, mode="eval", transform=transforms[1]),
        )

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath, label = self.df.iloc[index]
        img = Image.open(imgpath).convert("RGB")
        img = self.transform(img)

        return img, label
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return self.df.shape[0]

    @property
    def data_loader(self):
        return DataLoader(
            self,
            batch_size = self.config.BATCH_SIZE[self.mode],
            shuffle = self.mode == "train",
            num_workers = self.config.NUM_WORKERS,
            persistent_workers= self.config.PERSISTENT_WORKERS,
        )
