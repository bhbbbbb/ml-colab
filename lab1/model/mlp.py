import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
from math import floor
from torch.utils.data import DataLoader
import os


LOG_DIR = os.path.abspath(os.path.join(__file__, "..", "log"))

import torch.optim as optim
class MLP(nn.Module):
    INPUT_DIM = 28 * 28
    OUTPUT_DIM = 10
    def __init__(self, layers: int, neurons: int, epochs: int, learning_rate: float = 0.001,
                    start_epoch: int = 0, model_state_dict=None, optimizer_state_dict=None):
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
        """
        torch.manual_seed(880)
        np.random.seed(880)
        super().__init__()
        self.layers = layers
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate

        self._build(neurons, layers)

        ## Use GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = "cpu"

        ## Don't use cpu
        assert self.device == "cuda:0"

        self.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # load saved weights if start from checkpoint
        if start_epoch > 0 and start_epoch <= epochs:
            self.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        return
    
    def _build(self, neurons: int, layers: int) -> None:
        # build sequential model
        self.seq = nn.Sequential()
        if layers > 1:
            d = floor((2 * neurons / 3) / (layers - 1))
            pass
        else: d = 0

        in_units = self.INPUT_DIM
        out_units = neurons

        for i in range(1, layers + 1):
            self.seq.add_module(f"Lin{i}", nn.Linear(in_units, out_units))
            if i > 1:
                self.seq.add_module(f"sigmoid{i}", nn.Sigmoid()) ##
            in_units = out_units
            out_units = in_units - d

        self.seq.add_module("fc", nn.Linear(in_units, self.OUTPUT_DIM))
        self.seq.add_module("softmax", nn.Softmax(dim=1))

        self._print_layers()


    def _print_layers(self):
        print(self.seq)
        print("number of layer: {}  number of neurons: {}".format(self.layers, self.neurons))

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 28*28)
        x = self.seq(x)
        return x
    
    def start_train(self, data_loader: DataLoader) -> None:
        self.train() # set to train mode

        log_sub_dir = os.path.join(LOG_DIR, f"l{self.layers}_n{self.neurons}")
        if not os.path.isdir(log_sub_dir): os.makedirs(log_sub_dir)

        for epoch in tqdm(range(self.epochs)):
            running_loss = 0.0
            for i, data in enumerate(data_loader):
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

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 1000 mini-batches 
                    print('[%d, %5d] loss: %.6f' %
                        (epoch + 1, i + 1, running_loss / 999))
                    running_loss = 0.0

            torch.save({
                "start_epoch": epoch + 1,
                "epochs": self.epochs,
                "neurons": self.neurons,
                "layers": self.layers,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, os.path.join(log_sub_dir, f"l{self.layers}_n{self.neurons}_e{epoch+1}_of_{self.epochs}"))

        print('Finished Training')

    def start_eval(self, data_loader: DataLoader) -> None:
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %f %%' % (
            100 * correct / total))
    
