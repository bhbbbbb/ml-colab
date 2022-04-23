"""
start new training:
    python donf.py new --epochs=10 --batch-size=8
    or equivlently
    python donf.py new -e 10 -b 8

restart from last training:
    python donf.py last --epochs=10 --batch-size=8

restart from specific checkpoint:
    python donf.py last --epochs=10 --batch-size=8 --weights=/path/to/checkpoints
"""
import os
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dotenv import load_dotenv
from imgclf.dataset import Dataset
from nfnet.config import NfnetConfig
from nfnet.nfnet_model_utils import NfnetModelUtils

load_dotenv(".env")
SET_PATH = os.path.abspath("set.csv")
VAR = os.environ["VARIANT"]
DATASET_ROOT = os.path.abspath(os.environ["DATASET_ROOT"])
PRETRAINED_PATH = os.path.abspath(os.path.join(os.environ["PRETRAINED_ROOT"], f"{VAR}_haiku.npz"))
IMG_SIZE = {
    "F1": (224, 224),
    "F2": (256, 256),
    "F3": (320, 320),
    "F4": (384, 384),
}
CATS = ["banana", "bareland", "carrot", "corn", "dragonfruit", "garlic", "guava", "inundated", "peanut", "pineapple", "pumpkin", "rice", "soybean", "sugarcane", "tomato"]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE[VAR], InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE[VAR], InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

def main():
    assert os.path.isdir(DATASET_ROOT), f"cannot find dir {DATASET_ROOT}, pls checkout .env"
    assert os.path.isfile(PRETRAINED_PATH), "cannot find pretrained weights' path, pls checkout .env"

    mode, batch_size, epochs, weights = parse()
    train(mode, batch_size, epochs, weights)
    return


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str, help="'new', 'last'", metavar="<train>")
    parser.add_argument("-b", "--batch-size", required=True, type=int, metavar="<batch size>")
    parser.add_argument("-e", "--epochs", required=True, type=int, metavar="<num of epochs>")
    parser.add_argument("-w", "--weights", required=False, metavar="<path to checkpoint>")
    args = parser.parse_args()

    assert args.train in ["new", "last"]
    if args.weights is not None:
        assert os.path.isfile(args.weights), f"{args.weights} is not a valid path."
        assert args.train == "last", "do not specify weights in 'new' mode"
    
    return args.train, args.batch_size, args.epochs, args.weights

def get_config(batch_size):
    config = NfnetConfig(variant=VAR, log_dir=os.environ["LOG_ROOT"])
    config.batch_size["train"] = batch_size
    config.batch_size["eval"] = batch_size
    config.learning_rate = 0.1 * batch_size / 256
    config.display()
    return config

def get_df():
    if not os.path.isfile("set.csv"):
        def get_samples_list():
            img_list = np.array([], dtype=str)
            label_list = np.array([], dtype=int)
            for idx, cat in enumerate(CATS):
                sub_root = os.path.join(DATASET_ROOT, cat)
                sub_img_list = np.array(os.listdir(sub_root))
                sub_root += "/"
                sub_img_list = np.char.add(sub_root, sub_img_list)
                img_list = np.concatenate((img_list, sub_img_list))
                label_list = np.concatenate((label_list, np.full([len(sub_img_list)], idx)))

            df = pd.DataFrame({"img": img_list, "label": label_list})
            df.to_csv("set.csv")
            return
        get_samples_list()
    return pd.read_csv("set.csv", usecols=["img", "label"], dtype={"img": str, "label": np.int64})

def train(mode: str, batch_size, epochs, weight_path: str):
    config = get_config(batch_size)
    df = get_df()
    train_set, valid_set = Dataset.train_test_split(df, train_ratio=0.8, config=config, 
                        transforms_f=[TRAIN_TRANSFORM, EVAL_TRANSFORM])

    if mode == "new":
        utils = NfnetModelUtils.start_new_training_from_pretrained(PRETRAINED_PATH, config)
    elif weight_path is not None:
        model = NfnetModelUtils.init_model(config)
        utils = NfnetModelUtils.load_checkpoint(model, weight_path, config)
    else:
        model = NfnetModelUtils.init_model(config)
        utils = NfnetModelUtils.load_last_checkpoint(model, config)
    
    utils.train(epochs, train_set, valid_set)
    return
    
if __name__ == "__main__":
    main()


    