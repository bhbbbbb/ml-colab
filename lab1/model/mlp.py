from typing import Tuple
import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
from math import floor
from torch.utils.data import DataLoader
import os

import torch.optim as optim

from typing import TypedDict

LOG_DIR = os.path.abspath(os.path.join(__file__, "..", "log"))



class ModelStates(TypedDict): # interface
    model_state_dict: dict
    optimizer_state_dict: dict
    loss: float
    val_acc: float

class Model(TypedDict): # interface
    start_epoch: int
    epochs: int
    neurons: int
    layers: int
    activation: str
    learning_rate: float
    model_states: ModelStates
class MLP(nn.Module):
    INPUT_DIM = 28 * 28
    OUTPUT_DIM = 10
    def __init__(self, layers: int, neurons: int, epochs: int, learning_rate: float = 0.001,
                    activation: str = "relu", start_epoch: int = 0, model_states=None):
        """_summary_

        Args:
            layers (int)
            neurons (int)
            epochs (int)
            start_epoch (int): can be omit if not start from checkpoint.
                If specified, training will start from the start_epoch
                notice that epoch 0 is the first epoch, (start_epoch == epochs) indicates
                that training is finished.
            learning_rate (float): = 0.001
            activation (str): whether activation function to use only 'sigmoid' or 'relu'
        """
        assert activation == "relu" or activation == "sigmoid"

        torch.manual_seed(880)
        np.random.seed(880)
        super().__init__()
        self.layers = layers
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation = activation
        self.start_epoch = start_epoch

        self._build(neurons, layers)

        ## Use GPU
        assert torch.cuda.is_available()
        self.device = 'cuda:0'
        self.to(self.device)


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # load saved weights if start from checkpoint
        if start_epoch > 0 and start_epoch <= epochs:
            self.load_state_dict(model_states["model_state_dict"])
            self.optimizer.load_state_dict(model_states["optimizer_state_dict"])

        return
    
    def _build(self, neurons: int, layers: int) -> None:
        # build sequential model
        self.seq = nn.Sequential()
        if layers > 1:
            d = floor((2 * neurons) / (layers - 1))
        else: d = 0

        in_units = self.INPUT_DIM
        out_units = neurons * 3

        for i in range(1, layers + 1):
            self.seq.add_module(f"Lin{i}", nn.Linear(in_units, out_units))
            act = nn.ReLU() if self.activation == "relu" else nn.Sigmoid()
            self.seq.add_module(f"{self.activation}{i}", act)
            in_units = out_units
            out_units = in_units - d

        self.seq.add_module("fc", nn.Linear(in_units, self.OUTPUT_DIM))
        self.seq.add_module("softmax", nn.Softmax(dim=1))


    def _print_layers(self):
        print(self.seq)
        print("number of layer: {}  number of neurons: {} eta: {}".format(self.layers, self.neurons, self.learning_rate))

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 28*28)
        x = self.seq(x)
        return x
    
    def start_train(self, train_loader: DataLoader, test_loader: DataLoader,
            checkpoints: list = None) -> None:
        """start train

        Args:
            train_loader (DataLoader)
            test_loader (DataLoader)
            checkpoints (list, optional): list of epoch. specify which epoch should be save
                Defaults to None, and would save all epoch.
        """
        self._print_layers()
        self.train() # set to train mode

        # e.g. LOG_DIR/relu/l16_n1204_eta0.001/
        log_sub_dir = os.path.join(LOG_DIR, self.activation,
                f"l{self.layers}_n{self.neurons}_eta{self.learning_rate}")
        if not os.path.isdir(log_sub_dir): os.makedirs(log_sub_dir)

        if self.start_epoch:
            print(f"restart at epoch: {self.start_epoch + 1}")
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            overall_loss = running_loss / len(train_loader)
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, overall_loss))

            if not checkpoints or (epoch + 1) in checkpoints:
                val_acc, val_loss = self.start_eval(test_loader)
                self.train() # reset to train mode
                torch.save(Model(
                    start_epoch = epoch + 1,
                    epochs=self.epochs,
                    neurons=self.neurons,
                    layers=self.layers,
                    activation=self.activation,
                    learning_rate=self.learning_rate,
                    model_states=ModelStates(
                        model_state_dict=self.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        loss=overall_loss,
                        val_acc=val_acc,
                    ),
                ), os.path.join(log_sub_dir, f"epoch_{epoch+1}"))
            
            if epoch + 2 <= self.epochs:
                print(f"\nEpoch: {epoch+2}/{self.epochs}")

        print('Finished Training')

    def start_eval(self, data_loader: DataLoader) -> Tuple[float, float]:
        """start evaluation

        Args:
            data_loader (DataLoader)

        Returns:
            Tuple[float, float]: accuracy, loss
        """

        tem = f"epoch: {self.start_epoch}\n" if self.start_epoch > 0 else ""
        print(f"start evaluation:\nlayers: {self.layers}\nneurons: {self.neurons}\n{tem}")
        self.eval()
        correct = 0
        total = 0
        overall_loss = 0.0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                overall_loss += loss.item()

        accuracy = 100 * correct / total
        loss = overall_loss / len(data_loader)
        print('Accuracy of the network on the 10000 test images: %f %%' % accuracy)
        print(f"Loss: {loss: .6f}")
        print("-------------------------------------------------")

        return accuracy, loss
    
