import os
import torch
from imgclf.model_utils import ModelUtils
from imgclf.dataset import Dataset
from imgclf.models import FatLeNet5, FakeVGG16
from hw2.utils import DatasetUtils
from hw2.config import Hw2Config
from imgclf.models.models import AlexNet

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

from nfnet.config import NfnetConfig
from nfnet.nfnet_model_utils import NfnetModelUtils

def train_nfnet(epochs):
    config = NfnetConfig()
    train_batch_size = 4
    config.batch_size["train"] = train_batch_size
    config.batch_size["eval"] = 8
    config.learning_rate = 0.1 * train_batch_size / 256
    config.num_workers = 4
    config.num_class = 10
    config.optimizer = None
    config.epochs_per_checkpoint = 1  ########
    # config.display()

    # pretrained_path = "D:\\Documents\\1102\\AI\\ml-practices\\lab2\\imgclf\\pretrained_weight\\f1_haiku.npz"
    # utils = NfnetModelUtils.start_new_training_from_pretrained(pretrained_path, config)
    df, _ = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    model = NfnetModelUtils.init_model(config)
    # utils = NfnetModelUtils.load_last_checkpoint(model, config)
    folder = "D:\\Documents\\1102\\AI\\ml-practices\\lab2\\log\\20220406T17-08-42"
    utils = NfnetModelUtils.load_last_checkpoint_from_dir(model, dir_path=folder, config=config)
    utils.plot_history()
    return

def fine_tune(utils: NfnetModelUtils, config: NfnetConfig):
    config.epochs_per_checkpoint = 0  ########
    config.display()
    df, _ = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    # datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    df_ = df.loc[[0, 3000, 5000, 10000], ]
    train_set = Dataset(df_, config, mode="train")
    valid_set = Dataset(df_, config, mode="eval")
    utils.train(50, train_set, valid_set, valid_set)

def main():
    assert torch.cuda.is_available()

    config = Hw2Config()
    config.batch_size["train"] = 128
    config.batch_size["eval"] = 128

    config.early_stopping_threshold = 25
    config.learning_rate = 0.0001 * config.batch_size["train"] / 32
    config.epochs_per_checkpoint = 10
    config.conv_dropout_rate = 0
    config.dropout_rate = 0.5
    # config.display()
    model = AlexNet(config)
    print(model)
    return
    # model = FatLeNet5(config)
    # utils = ModelUtils.start_new_training(model, config)
    utils = ModelUtils.load_last_checkpoint(model, config)
    # utils = ModelUtils.load_last_checkpoint_from_dir(model, config, dir_path="D:\\Documents\\1102\\AI\\ml-practices\\lab2\\log\\20220407T12-54-31")
    # utils.plot_history()
    df, _ = DatasetUtils.load_csv(TRAIN_CSV, TRAIN_DATASET)
    # datasets = Dataset.split(df, split_ratio=[0.7, 0.15], config=config)
    # indexer = torch.randint(0, df.shape[0], size=[config.batch_size["train"]])
    # df_ = df.loc[indexer, ]
    # train_set = Dataset(df_, config, mode="train")
    # valid_set = Dataset(df_, config, mode="eval")
    datasets = Dataset.split(df, [.7, .15], config)
    model.summary(file=utils.logger)
    utils.train(100, *datasets)
    utils.plot_history()
    # utils.train(200, train_set, valid_set, valid_set)
    # train_nfnet(100)
    

if __name__ == "__main__":
    main()
