import os
import pandas as pd

class DatasetUtils:

    @staticmethod
    def load_csv(csv_path: str, images_root: str):
        """load the required csv file and process for `imgclf.dataset.Dataset`
        Args:
            csv_path (str): csv file that contain the image_file
            images_root (str): root of where images located
        
        Returns:
            df, categories
        """
        df = pd.read_csv(csv_path, dtype={"img": str, "label": "category"})

        assert os.path.isdir(images_root)
        if images_root[-1] not in ["\\", "/"]:
            images_root += "/"
        df["img"] = (images_root + df["img"])
        categories = df["label"].cat.categories
        df["label"] = df["label"].cat.codes
        return df, categories
    
    @staticmethod
    def load_test_csv(csv_path: str, images_root: str):
        """
        Returns:
            df
        """
        df, _ = DatasetUtils.load_csv(csv_path, images_root)

        return df.drop(columns=["label"])
