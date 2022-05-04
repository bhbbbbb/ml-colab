import os
from tqdm import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam
from sklearn.cluster import MiniBatchKMeans
import joblib
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from model_utils import BaseModelUtils
from model_utils.base.criteria import Loss, Criteria
from model_utils.config import ModelUtilsConfig as BaseModelUtilsConfig
from .dataset import DatasetMNIST
from .model import VariationalAutoencoder
from .preprocess import _Transform, PreprocessingConfig, Preprocessor

class ModelUtilsConfig(BaseModelUtilsConfig, PreprocessingConfig):
    preprocessing: bool = True


@Criteria.register_criterion
class OverallLoss(Loss):

    short_name: str = "overall_loss"
    full_name: str = "Overall Loss"
    
    plot: bool = True
    """Whether plot this criterion"""

    primary: bool = True

@Criteria.register_criterion
class ReconstructLoss(Loss):

    short_name: str = "recon_loss"
    full_name: str = "Reconstruct Loss"
    
    plot: bool = False
    """Whether plot this criterion"""

    primary: bool = False
@Criteria.register_criterion
class KLDLoss(Loss):

    short_name: str = "kl_loss"
    full_name: str = "KLD Loss"
    
    plot: bool = False
    """Whether plot this criterion"""

    primary: bool = False

class ModelUtils(BaseModelUtils):

    config: ModelUtilsConfig
    model: VariationalAutoencoder
    preprocessor: Preprocessor
    
    @staticmethod
    def _get_optimizer(model: nn.Module, config: ModelUtilsConfig):
        return Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    
    def train(self, epochs: int, trainset: DatasetMNIST,
                validset: DatasetMNIST = None, testset: DatasetMNIST = None) -> str:
        if self.config.preprocessing:
            self.preprocessor = Preprocessor(self.config)
        return super().train(epochs, trainset, validset, testset)

    def _train_epoch(self, train_dataset: DatasetMNIST):

        self.model.train()

        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        for img, _ in tqdm(train_dataset.dataloader):
            
            img: Tensor
            if self.config.preprocessing:
                img = self.preprocessor.forward(img)
            img = img.to(self.config.device)

            # vae reconstruction
            outputs: Tensor = self.model(img)
            # print(image_batch_recon.shape)
            # reconstruction error
            overall_loss, recon_loss, kl_loss = self.model.criterion(img, *outputs)

            # print(recon_loss)
            # self.logger.file.log(recon_loss)
            # self.logger.file.log(str(img[0][0]))
            # self.logger.file.log(str(outputs[0][0]))
            
            # backpropagation
            self.optimizer.zero_grad()
            overall_loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            self.optimizer.step()
            
            train_loss += overall_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            
        
        train_loss /= len(train_dataset)
        train_recon_loss /= len(train_dataset)
        train_kl_loss /= len(train_dataset)
        return Criteria(
            OverallLoss(train_loss),
            ReconstructLoss(train_recon_loss),
            KLDLoss(train_kl_loss),
        )
    
    def _eval_epoch(self, eval_dataset: DatasetMNIST):
        self.model.eval()

        eval_loss = 0.0
        eval_recon_loss = 0.0
        eval_kl_loss = 0.0
        
        for img, _ in eval_dataset.dataloader:
            
            img: Tensor

            if self.config.preprocessing:
                img = self.preprocessor.forward(img)
            
            img = img.to(self.config.device)

            # vae reconstruction
            outputs: Tensor = self.model(img)
            # if eval_loss == 0:
            #     sample = outputs[0][0]
            #     ans = img[0]
            #     print(sample)
            #     print(ans)
            #     print(sample.shape)
            # print(image_batch_recon.shape)
            # reconstruction error
            overall_loss, recon_loss, kl_loss = self.model.criterion(img, *outputs)
            
            tem = overall_loss.item()
            # if eval_loss == 0:
            #     print(f"first loss: {tem}")
            eval_loss += tem
            eval_recon_loss += recon_loss.item()
            eval_kl_loss += kl_loss.item()
        
        eval_loss /= len(eval_dataset)
        eval_recon_loss /= len(eval_dataset)
        eval_kl_loss /= len(eval_dataset)
        return Criteria(
            OverallLoss(eval_loss),
            ReconstructLoss(eval_recon_loss),
            KLDLoss(eval_kl_loss),
        )

    def visualise_output(self, dataset: DatasetMNIST):
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
            

    def visualise_2d_lantent_space(self, width: float = 1.5):
        def do(latents: Tensor, name_suffixes: str = ""):
            with torch.inference_mode():
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
                path = os.path.join(self.root, f"2d_lantent_space{name_suffixes}")
                plt.savefig(path)
                plt.show()
            return
        
        # load a network that was trained with a 2d latent space
        assert self.model.config.latent_dims >= 2,\
            "latent_dims have to be greater than 2"
        
        SIZE = 20
        # create a sample grid in 2d latent space
        latent_x = np.linspace(-1 * width, width, SIZE)
        latent_y = np.linspace(-1 * width, width, SIZE)
        latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, 2) # flatten grid into a batch
        
        remain_dim = self.model.config.latent_dims - 2
        if remain_dim:
            for l_pad in range(remain_dim + 1):
                r_pad = remain_dim - l_pad
                padding = (l_pad, r_pad)
                latents_ = F.pad(latents, padding)
                name = str(padding).replace(", ", "_")
                do(latents_, name)
        else:
            do(latents)
            


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

def train_kmeans(train_set: DatasetMNIST, config: ModelUtilsConfig, show: bool = True):
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

    model_dir = os.path.dirname(config.kmeans_model_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, config.kmeans_model_path)
    return kmeans
