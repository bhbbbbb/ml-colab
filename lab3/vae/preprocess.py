import os
from typing import Tuple
import torch
from torch import Tensor
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
from model_utils.base.config import BaseConfig, UNIMPLEMENTED

class PreprocessingConfig(BaseConfig):
    n_clusters: int = 128
    kmeans_model_path: str = UNIMPLEMENTED
    

class Preprocessor:

    model: MiniBatchKMeans

    def __init__(self, config: PreprocessingConfig):
        
        assert os.path.isfile(config.kmeans_model_path), "have to train the kmeans model first"

        self.model = joblib.load(config.kmeans_model_path)
        
        self.config = config
        self.transform = _Transform()

    def forward(self, img: Tensor):
        """From RGB pixel to one hot encoding
            img (Tensor): b x c x w x h
        """
        img = self.transform.to_2d(img)
        # (b*w*h) x c
        prediction: Tensor = self.model.predict(img)
        # prediction: (b*w*h)
        def to_one_hot(index: int):
            # arr = torch.zeros(self.config.n_clusters)
            arr = np.zeros(self.config.n_clusters)
            arr[index] = 1.0
            return arr

        p_img = np.array([to_one_hot(index) for index in prediction])
        p_img = Tensor(p_img)
        # (b*w*h) x k
        b, _, w, h = self.transform.shape
        p_img = p_img.reshape(b, w * h, self.config.n_clusters)
        # b x (w*h) x k
        p_img = p_img.transpose(1, 2)
        # b x k x (w*h)
        p_img = p_img.reshape(b, self.config.n_clusters, w, h)
        # b x k x w x h
        return p_img
    
    def inference(self, img: Tensor):
        b, _, w, h = img.shape
        # b x k x w x h
        _, indices = img.max(dim=1)
        # indices: b x w x h

        mapping = self.model.cluster_centers_
        c = len(mapping[0])
        indices = indices.reshape(b * w * h)
        img_p = np.array([mapping[idx] for idx in indices])
        img_p = Tensor(img_p)
        # (b * w * h) x c
        img_p = img_p.reshape(b, w*h, c)
        # b x (w * h) x c
        img_p = img_p.transpose(1, 2)
        # b x c x (w * h)
        img_p = img_p.reshape(b, c, w, h)
        return img_p
    
    def print_percentage(self):
        percent = []
        labels = self.model.labels_.tolist()
        for i in range(len(self.model.cluster_centers_)):
            j = labels.count(i)
            j = j / (len(labels))
            percent.append(j)
        percent.sort()
        print(percent)



class _Transform:
    shape: Tuple[int, int, int, int] # b x c x w x h

    def __init__(self, shape = None):
        """_summary_

        Args:
            c (int): num of channels
            shape (_type_, optional): _description_. Defaults to None.
        """
        self.shape = shape

    def to_2d(self, img: Tensor):
        self.shape = img.shape
        b, c, w, h = self.shape
        img = img.view(b, c, w * h)
        img.transpose_(1, 2)
        img = img.reshape(b * w * h, c)
        return img

    def de_trans(self, flat: Tensor):
        b, c, w, h = self.shape
        flat = flat.reshape(b, w * h, c)
        flat = flat.transpose(1, 2)
        flat = flat.reshape(b, c, w, h)
        return flat
