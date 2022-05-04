"""
start new training:
    python donf.py new --epochs=10 --batch-size=8
    or equivlently
    python donf.py new -e 10 -b 8

restart from last training:
    python donf.py last --epochs=10 --batch-size=8

restart from specific checkpoint:
    python donf.py last --epochs=10 --batch-size=8 --weights=/path/to/checkpoints

inference for testing:
    python donf.py inference --batch-size=? --weights=/path/to/checkpoints --confidence --full-path

        --confidence: export csv with 'confidence' column
        --full-path: export csv with the 'image_filename' column having full-path
                        (otherwise, only filename)
    output file: test_inf(_conf).csv

inference for submission:
    python donf.py inference --batch-size=? --weights=/path/to/checkpoints --test-dir=/path/to/testdir
    output file: submission.csv
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
# CATS = ["banana", "bareland", "carrot", "corn", "dragonfruit", "garlic", "guava", "inundated", "peanut", "pineapple", "pumpkin", "rice", "soybean", "sugarcane", "tomato"]
CATS = ["banana", "bareland", "carrot", "corn", "dragonfruit", "garlic", "guava", "bareland", "peanut", "pineapple", "pumpkin", "rice", "soybean", "sugarcane", "tomato"]

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
    # assert os.path.isfile(PRETRAINED_PATH), "cannot find pretrained weights' path, pls checkout .env"

    mode, batch_size, epochs, weights, confidence, test_dir, full_path = parse()
    if mode != "inference":
        train(mode, batch_size, epochs, weights)
    else:
        inference(batch_size, weights, confidence, test_dir, full_path)
    return


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="'new', 'last', 'inference'", metavar="command")
    parser.add_argument("-b", "--batch-size", required=True, type=int, metavar="<batch size>")
    parser.add_argument("-e", "--epochs", required=False, type=int, metavar="<num of epochs>")
    parser.add_argument("-w", "--weights", required=False, metavar="<path to checkpoint>")
    parser.add_argument("-c", "--confidence", action="store_true")
    parser.add_argument("--full-path", action="store_true")
    parser.add_argument("--test-dir", required=False, type=str, metavar="<path to test dir>")
    args = parser.parse_args()

    assert args.command in ["new", "last", "inference"]
    if args.weights is not None:
        assert os.path.isfile(args.weights), f"{args.weights} is not a valid path."
        assert args.command != "new", "do not specify weights in 'new' mode"
    
    if args.confidence:
        assert args.command == "inference"
    
    return (
        args.command, args.batch_size, args.epochs,
        args.weights, args.confidence, args.test_dir, args.full_path
    )

def get_config(batch_size):
    config = NfnetConfig(variant=VAR, log_dir=os.environ["LOG_ROOT"])
    config.batch_size["train"] = batch_size
    config.batch_size["eval"] = batch_size
    config.num_class = 10
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

def get_infset(test_dir: str, config: NfnetConfig):
    """supposed the images to inference is placed in `test_dir`"""
    assert os.path.isdir(test_dir)
    images = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]
    df = pd.DataFrame({"img": images})
    return Dataset(df, config, mode="inference", transform=EVAL_TRANSFORM)


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

def inference(batch_size, weight_path: str, confidence: bool,
                test_dir: str = None, full_path: bool = None):
    config = get_config(batch_size)
    
    if test_dir is None:
        df = get_df()
        _, inf_set = Dataset.train_test_split(df, train_ratio=0.99, config=config, 
                            transforms_f=[TRAIN_TRANSFORM, EVAL_TRANSFORM])
        inf_set = Dataset(inf_set.df, config, mode="inference", transform=EVAL_TRANSFORM)
    else:
        inf_set  = get_infset(test_dir, config)

    if weight_path is not None and weight_path != "last":
        model = NfnetModelUtils.init_model(config)
        utils = NfnetModelUtils.load_checkpoint(model, weight_path, config)
    else:
        model = NfnetModelUtils.init_model(config)
        utils = NfnetModelUtils.load_last_checkpoint(model, config)
    
    res_df = utils.inference(inf_set, CATS, confidence)
    name = "test_inf" if test_dir is None else "submission"
    if confidence:
        name += "_conf"
    name += ".csv"
    img_list = inf_set.df["img"].to_list()
    if not full_path:
        img_list = [os.path.basename(path) for path in img_list]
    if confidence:
        res_df = pd.DataFrame(
            {
                "image_filename": img_list,
                "label": res_df["label"],
                "confidence": res_df["confidence"],
            }
        )
    else:
        res_df = pd.DataFrame(
            {
                "image_filename": img_list,
                "label": res_df["label"],
            }
        )
    print(res_df)
    res_df.to_csv(name, index=False)
    return


    
if __name__ == "__main__":
    main()

    