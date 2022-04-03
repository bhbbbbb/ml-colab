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
    utils = ModelUtils(model=model, config=config)
    utils.train(datasets, epochs=epochs)
    utils.plot_history()

def train_from(config, model, epochs, checkpoint_path):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils.load_checkpoint(model=model, config=config, checkpoint_path=checkpoint_path)
    utils.train(datasets, epochs=epochs)
    utils.plot_history()
    return

def retrain(config, model, epochs):
    df, cat = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    utils = ModelUtils.load_last_checkpoint(model=model, config=config)
    utils.train(datasets, epochs=epochs)
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
    config.BATCH_SIZE["train"] = 32
    config.BATCH_SIZE["eval"] = 16

    config.EARLY_STOPPING_THRESHOLD = 30
    config.LEARNING_RATE *= 10
    config.EPOCHS_PER_CHECKPOINT = 20
    config.MODEL_CONFIG.dropout_rate = 0.3
    config.display()
    model = FakeVGG16(config.MODEL_CONFIG)
    # train(config, model, 50)
    utils = ModelUtils.load_last_checkpoint(model, config)
    utils.plot_history()
    # retrain(config, FatLeNet5(config), 105)
    # pretrained_nfnet()
    

if __name__ == "__main__":
    main()
