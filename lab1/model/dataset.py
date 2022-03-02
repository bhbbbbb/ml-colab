import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

class Dataset:
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    ROOT: str
    def __init__(self):
        DATASET_ROOT = {
            "local": os.path.abspath(os.path.join(__file__, "..", "data")),
            "colab": "/content/drive/MyDrive/Colab Notebooks/data/MNIST"
        }
        self.ROOT = DATASET_ROOT["local"] if os.name == "nt" else DATASET_ROOT["colab"]
        return

    def load(self) -> Tuple[DataLoader, DataLoader]:
        """load dataset from wether internet or local

        Returns:
            Tuple[DataLoader, DataLoader]: trainloader, testloader
        """
        transform = transforms.Compose([transforms.ToTensor(),])##
        # /content/drive/MyDrive/Colab Notebooks/data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
        trainset = torchvision.datasets.MNIST(root=self.ROOT, train=True,download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=32,shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=self.ROOT, train=False,download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=4,shuffle=True, num_workers=2)

        return trainloader, testloader
