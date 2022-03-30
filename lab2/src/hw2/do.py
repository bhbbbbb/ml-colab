import sys
import os
sys.path.append(os.path.join(__file__, "..", ".."))
from utils import DatasetUtils
from model.model_utils import ModelUtils
from model.dataset import Dataset
from model.model import FatLeNet5
from config import Hw2Config
DATASET_ROOT  = os.path.abspath(os.path.join(__file__, "..", "..", "..", "homework2_Dataset"))
TRAIN_DATASET = os.path.join(DATASET_ROOT, "train_dataset")
TEST_DATASET  = os.path.join(DATASET_ROOT, "test_dataset")
TRAIN_CSV     = os.path.join(DATASET_ROOT, "train.csv")
TEST_CSV      = os.path.join(DATASET_ROOT, "test.csv")

def train(config):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils(model=FatLeNet5(config), config=config)
    utils.train(datasets, epochs=50)

def retrain(config):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils.load_last_checkpoint(model=FatLeNet5(config), config=config)
    utils.train(datasets, epochs=101)
    return

def inference(config, categories: list):
    df = DatasetUtils.load_test_csv(
            csv_path = TEST_CSV,
            images_root = TEST_DATASET,
        )
    dataset = Dataset(df, config=config, mode="inference")
    utils = ModelUtils.load_last_checkpoint(model=FatLeNet5(config), config=config)
    df = utils.inference(dataset, categories, confidence=True)
    return df

if __name__ == "__main__":
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    
    config = Hw2Config()
    config.display()

    # inference(config, cat)


    
    # train(config, datasets)
    # retrain(config)
