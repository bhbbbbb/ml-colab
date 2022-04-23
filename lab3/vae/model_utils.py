import os
from tqdm import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
from sklearn.cluster import MiniBatchKMeans
import joblib
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from model_utils import BaseModelUtils
from model_utils.config import ModelUtilsConfig as BaseModelUtilsConfig
from .dataset import Dataset
from .model import VariationalAutoencoder
from .preprocess import _Transform, PreprocessingConfig, Preprocessor

class ModelUtilsConfig(BaseModelUtilsConfig, PreprocessingConfig):
    preprocessing: bool = True



class ModelUtils(BaseModelUtils):

    config: ModelUtilsConfig
    model: VariationalAutoencoder
    preprocessor: Preprocessor
    
    @staticmethod
    def _get_optimizer(model: nn.Module, config: ModelUtilsConfig):
        return Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    
    def train(self, epochs: int, trainset: Dataset,
                validset: Dataset = None, testset: Dataset = None) -> str:
        if self.config.preprocessing:
            self.preprocessor = Preprocessor(self.config)
        return super().train(epochs, trainset, validset, testset)

    def _train_epoch(self, train_dataset: Dataset):

        self.model.train()

        train_loss = 0.0
        for img, _ in tqdm(train_dataset.dataloader):
            
            img: Tensor
            if self.config.preprocessing:
                img = self.preprocessor.forward(img)
            img = img.to(self.config.device)

            # vae reconstruction
            outputs: Tensor = self.model(img)
            # print(image_batch_recon.shape)
            # reconstruction error
            loss= self.model.criterion(img, *outputs)
            
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            self.optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataset)
        return train_loss
    
    def _eval_epoch(self, eval_dataset: Dataset):
        self.model.eval()

        eval_loss = 0.0
        for img, _ in eval_dataset.dataloader:
            
            img: Tensor

            if self.config.preprocessing:
                img = self.preprocessor.forward(img)
            
            img = img.to(self.config.device)

            # vae reconstruction
            outputs: Tensor = self.model(img)
            # print(image_batch_recon.shape)
            # reconstruction error
            loss: Tensor = self.model.criterion(img, *outputs)
            
            eval_loss += loss.item()
        
        eval_loss /= len(eval_dataset)
        return eval_loss

    def visualise_output(self, dataset: Dataset):
        self.model.eval()
        dataloader = dataset.dataloader
        images, _ = next(iter(dataloader))
        # First visualise the original images
        images = images[:50]
        def show(imgs, title: str):
            imgs_grid = torchvision.utils.make_grid(imgs, 10, 5)
            imgs_grid = to_img(imgs_grid)
            npimg = imgs_grid.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            path = os.path.join(self.root, title)
            plt.savefig(path)
            plt.show()
        show(images, "original")
        if self.config.preprocessing:
            self.preprocessor = Preprocessor(self.config) 
            images = self.preprocessor.forward(images)
            show(self.preprocessor.inference(images), "preprocessed")
        with torch.no_grad():
            images: Tensor = images.to(self.config.device)
            images, _, _ = self.model(images)
            images = images.cpu()
            images = to_img(images)
            if self.config.preprocessing:
                show(self.preprocessor.inference(images), "reconstructional")
            else:
                show(images, "reconstructional")
            

    def visualise_2d_lantent_space(self):
        # load a network that was trained with a 2d latent space
        assert self.model.config.latent_dims == 2,\
            "Please change the parameters to two latent dimensions."
        
        SIZE = 20
        with torch.no_grad():
            # create a sample grid in 2d latent space
            latent_x = np.linspace(-1.5,1.5,SIZE)
            latent_y = np.linspace(-1.5,1.5,SIZE)
            latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
            for i, lx in enumerate(latent_x):
                for j, ly in enumerate(latent_y):
                    latents[j, i, 0] = lx
                    latents[j, i, 1] = ly
            latents = latents.view(-1, 2) # flatten grid into a batch

            # reconstruct images from the latent vectors
            latents = latents.to(self.config.device)
            image_recon: Tensor = self.model.decoder(latents)
            image_recon = image_recon.cpu()
            fig, ax = plt.subplots(figsize=(10, 10))
            images = image_recon.data[:(SIZE ** 2)]
            if self.config.preprocessing:
                self.preprocessor = Preprocessor(self.config)
                images = self.preprocessor.inference(images)

            show_image(torchvision.utils.make_grid(images, 20, 5))
            path = os.path.join(self.root, "2d_lantent_space")
            plt.savefig(path)
            plt.show()


# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x: Tensor):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def train_kmeans(train_set: Dataset, config: ModelUtilsConfig, show: bool = True):
    idx = 0
    kmeans = MiniBatchKMeans(n_clusters=config.n_clusters)
    transform = _Transform()
    # B = train_set.config.batch_size[M.TRAIN]
    for img, _ in tqdm(train_set.dataloader):
        img: Tensor
        # if img.size(0) != B:
        #     break
        if idx == 0:
            first_batch = img
        img = transform.to_2d(img)
        kmeans.partial_fit(img)
        idx += 1

    def predict_and_recover(kmeans: MiniBatchKMeans, img: Tensor):
        img = transform.to_2d(img)
        p_img: Tensor = kmeans.predict(img)
        p_img = np.array([kmeans.cluster_centers_[pixel] for pixel in p_img])
        p_img = Tensor(p_img)
        p_img = transform.de_trans(p_img)
        return p_img

    if show:
        def _show(images: Tensor, path: str = "ori"):
            np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 2).numpy()
            np_imagegrid = np.transpose(np_imagegrid, (1, 2, 0))
            plt.imshow(np_imagegrid)
            plt.savefig(path)
            plt.show()
        _show(first_batch)
        _show(predict_and_recover(kmeans, first_batch), "pro")

    joblib.dump(kmeans, config.kmeans_model_path)
    return kmeans
