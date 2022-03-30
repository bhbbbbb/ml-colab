from model.dataset import Dataset
import pandas as pd
import os

class DatasetUtils:
    DATASET_ROOT  = os.path.abspath(os.path.join(__file__, "..", "..", "..", "homework2_Dataset"))
    TRAIN_DATASET = os.path.join(DATASET_ROOT, "train_dataset")
    TEST_DATASET  = os.path.join(DATASET_ROOT, "test_dataset")
    TRAIN_CSV     = os.path.join(DATASET_ROOT, "train.csv")
    TEST_CSV      = os.path.join(DATASET_ROOT, "test.csv")

    @staticmethod
    def load_csv(csv_path: str, images_root: str):
        """load the required csv file and process for `model.dataset.Dataset`
        Args:
            csv_path (str): csv file that contain the image_file
            images_root (str): root of where images located
        """
        df = pd.read_csv(csv_path, dtype={"img": str, "label": "category"})

        assert os.path.isdir(images_root)
        if images_root[-1] not in ["\\", "/"]:
            images_root += "/"
        df["img"] = (images_root + df['img'])
        df["img"] = df["img"].astype(str)
        categories = df["label"].cat.categories
        df["label"] = df['label'].cat.codes
        return df
