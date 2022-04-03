import os
import torch
from imgclf.model_utils import ModelUtils
from imgclf.dataset import Dataset
from imgclf.models import FatLeNet5, FakeVGG16
from hw2.utils import DatasetUtils
from hw2.config import Hw2Config

DATASET_ROOT  = os.path.abspath(os.path.join(__file__, "..", "homework2_Dataset"))
TRAIN_DATASET = os.path.join(DATASET_ROOT, "train_dataset")
TEST_DATASET  = os.path.join(DATASET_ROOT, "test_dataset")
TRAIN_CSV     = os.path.join(DATASET_ROOT, "train.csv")
TEST_CSV      = os.path.join(DATASET_ROOT, "test.csv")

def train(config, model, epochs):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils.start_new_training(model=model, config=config)
    utils.train(epochs, *datasets)
    utils.plot_history()

def train_from(config, model, epochs, checkpoint_path):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils.load_checkpoint(model=model, config=config, checkpoint_path=checkpoint_path)
    utils.train(epochs, *datasets)
    utils.plot_history()
    return

def retrain(config, model, epochs):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils.load_last_checkpoint(model=model, config=config)
    utils.train(epochs, *datasets)
    utils.plot_history()
    return

def inference(config, categories: list, model):
    df = DatasetUtils.load_test_csv(
            csv_path = TEST_CSV,
            images_root = TEST_DATASET,
        )
    dataset = Dataset(df, config=config, mode="inference")
    utils = ModelUtils.load_last_checkpoint(model=model, config=config)
    df = utils.inference(dataset, categories, confidence=True)
    return df


# from imgclf.nfnets import pretrained_nfnet
# from imgclf.nfnets import NFNet
# from imgclf.nfnets import SGD_AGC


# NFNET_F1_PATH = ""
# def train_nfnet(config, epochs):
# 

def main():
    assert torch.cuda.is_available()

    config = Hw2Config()
    config.batch_size["train"] = 64
    config.batch_size["eval"] = 32

    config.early_stopping_threshold = 20
    config.learning_rate *= 0.5 
    config.epochs_per_checkpoint = 20
    config.dropout_rate = 0.3
    config.display()
    model = FatLeNet5(config)
    # train(config, model, 50)
    retrain(config, model, 100)
    # pretrained_nfnet()
    

if __name__ == "__main__":
    main()
