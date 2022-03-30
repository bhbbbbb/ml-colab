import sys
import os
sys.path.append(os.path.join(__file__, "..", ".."))
from utils import DatasetUtils
from model.model_utils import ModelUtils
from model.dataset import Dataset
from model.model import FatLeNet5
from config import Hw2Config

def train(config, datasets):
    utils = ModelUtils(model=FatLeNet5(config), datasets=datasets, epochs=2, config=config)
    utils.train()

def retrain(config, datasets):
    mu = ModelUtils.load_checkpoint(model=FatLeNet5(config), datasets=datasets, epochs=4,
        config=config, checkpoint_path="D:\\Documents\\1102\\AI\\ml-practices\\lab2\\src\\hw2\\log\\20220330T16-31-29\\20220330T16-32-24_epoch_2")
    
    mu.train()
    return



if __name__ == "__main__":
    df = DatasetUtils.load_csv(
            csv_path = DatasetUtils.TRAIN_CSV,
            images_root = DatasetUtils.TRAIN_DATASET,
        )
    # config = Hw2Config(BATCH_SIZE = { "train": 8, "eval": 64, })
    config = Hw2Config()
    config.display()
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config,
                            transforms=[Dataset.TRAIN_TRANSFORM, Dataset.EVAL_TRANSFORM])
    
    # train(config, datasets)
    retrain(config, datasets)
