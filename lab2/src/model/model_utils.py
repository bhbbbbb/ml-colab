from .config import Config
from .dataset import Dataset
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, TypedDict
from argparse import Namespace
from tqdm import tqdm
import torch
import os
import json

class Stat:
    train_loss: float
    valid_loss: float
    train_acc: float
    valid_acc: float
    test_loss: float
    test_acc: float

    def __init__(self, train_loss: float, valid_loss: float, train_acc: float, valid_acc: float):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_acc = train_acc
        self.valid_acc = valid_acc
        return
    
    def display(self):
        if hasattr(self, "test_loss") and hasattr(self, "test_acc"):
            print(f"test_loss: {self.valid_loss: .6f}")
            print(f"test_acc: {self.train_acc * 100: .2f} %")
        else:
            print(f"train_loss: {self.train_loss: .6f}")
            print(f"train_acc: {self.train_acc * 100: .2f} %")
            print(f"valid_loss: {self.valid_loss: .6f}")
            print(f"valid_acc: {self.train_acc * 100: .2f} %")
        return


class ModelStates(Namespace):
    # epoch to start from (0 is the first)
    start_epoch: int

    # epoch to train until
    epochs: int

    # config: Config

    ########## torch built-in model states ############
    model_state_dict: dict
    optimizer_state_dict: dict

    # statistic of the last epoch
    stat: Stat

class History(TypedDict):

    # root of log dir
    root: str

    history: List[dict] # List of Stat in dict format

    checkpoints: dict


class ModelUtils:

    history: History

    def __init__(self, model: nn.Module, datasets: Tuple[Dataset, Dataset, Dataset],
        epochs: int, config: Config):

        self.model = model
        self.model.to(config.DEVICE)
        self.train_loader = datasets[0].data_loader
        self.valid_loader = datasets[1].data_loader
        self.test_loader  = datasets[2].data_loader
        self.epochs = epochs
        self.config = config

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.start_epoch = 0

        self.root = config.LOG_DIR
        self.history = History(root="", history=[], checkpoints={})
        return

    @classmethod
    def load_checkpoint(cls, model: nn.Module, datasets: Tuple[Dataset, Dataset, Dataset], 
            checkpoint_path: str, config: Config, epochs: int = None):
        """init ModelUtils class with the saved model (or checkpoint)

        Args:
            model (nn.Module): model architecture
            datasets (Tuple[Dataset, Dataset, Dataset]): dataset
            checkpoint_path (str): path of saved model (or checkpoint)
            config (Config): config
            epochs (int): defalut to None, if None. train to the epochs store in checkpoint.
            Specify to change the target epochs

        """

        assert os.path.isfile(checkpoint_path)

        tem = torch.load(checkpoint_path)
        checkpoint = ModelStates(**tem)
        
        if epochs is None:
            epochs = checkpoint.epochs

        new = ModelUtils(model, datasets, epochs, config)

        new.model.load_state_dict(checkpoint.model_state_dict)
        new.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        new.start_epoch = checkpoint.start_epoch
        
        new.root = os.path.dirname(checkpoint_path)
        log_path = os.path.join(new.root, f"{os.path.basename(new.root)}_history.json")

        ### TEM
        assert os.path.isfile(log_path)
        ### TEM

        if os.path.isfile(log_path):
            with open(log_path, "r") as fin:
                new.history = json.load(fin)
            
        return new

    def _save(self, cur_epoch: int, stat: Stat) -> str:
        tem = vars(ModelStates(
            start_epoch = cur_epoch + 1,
            epochs = self.epochs,
            model_state_dict = self.model.state_dict(),
            optimizer_state_dict = self.optimizer.state_dict(),
            stat = vars(stat),
        ))
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch + 1}"
        path = os.path.join(self.root, name)
        if not os.path.isdir(self.root): os.makedirs(self.root)
        torch.save(tem, path)
        self.history["checkpoints"][cur_epoch + 1] = name
        return name
    
    def _log(self, stat: Stat) -> str:

        self.history["history"].append(vars(stat))
        self.history["root"] = self.root
        name = f"{os.path.basename(self.root)}_history.json"
        path = os.path.join(self.root, name)
        if not os.path.isdir(self.root): os.makedirs(self.root)
        with open(path, "w") as fout:
            json.dump(self.history, fout, indent=4)
        return path
    


    def _train_epoch(self) -> Tuple[float, float]:
        """train a single epoch

        Returns:
            Tuple[float, float]: train_loss, train_acc
        """
        self.model.train()

        train_loss = 0.0
        train_correct = 0
        
        for data, label in tqdm(self.train_loader):

            data: Tensor
            label: Tensor

            data, label = data.to(self.config.DEVICE), label.to(self.config.DEVICE)
            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output: Tensor = self.model(data)
            # calculate the batch loss
            loss = self.criterion.forward(output, label)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            self.optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)
            # update training Accuracy
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == label).sum().item()
        
        train_loss = train_loss / len(self.train_loader.dataset)
        train_acc = train_correct / len(self.train_loader.dataset)

        return train_loss, train_acc
    
    def _eval_epoch(self) -> Tuple[float, float]:
        """evaluate single epoch

        Returns:
            Tuple[float, float]: valid_loss, valid_acc
        """
        self.model.eval()
        eval_loss = 0.0
        correct = 0

        for data, target in self.test_loader:
            data: Tensor
            target: Tensor

            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)

            output: Tensor = self.model(data)

            loss = self.criterion.forward(output, target)

            eval_loss += loss.item() * data.size(0)
            
            _, predicted = output.max(dim=1)
            correct += (predicted == target).sum().item()
        
        eval_loss = eval_loss / len(self.test_loader.dataset)
        eval_acc = correct / len(self.test_loader.dataset)
        return eval_loss, eval_acc
    
    def train(self) -> str:
        """start training

        Returns:
            str: json path as the history
        """

        # counting for early stopping
        min_valid_loss = 999999
        counter = 0

        if self.start_epoch == 0:
            # mkdir for new training log
            self.root = os.path.join(self.config.LOG_DIR, formatted_now())
            os.makedirs(self.root)

        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch: {epoch + 1} / {self.epochs}")
            train_loss, train_acc = self._train_epoch()
            valid_loss, valid_acc = self._eval_epoch()

            stat = Stat(
                train_loss=train_loss,
                train_acc=train_acc,
                valid_loss=valid_loss,
                valid_acc=valid_acc,
            )
            stat.display()

            if self.config.EARLY_STOPPING:
                if valid_loss > min_valid_loss:
                    counter += 1
                    if counter == self.config.EARLY_STOPPING_THRESHOLD:
                        print("Early stopping!")
                        self._save(epoch, stat)
                        break
                else:
                    min_valid_loss = valid_loss
                    counter = 0
            
            if epoch == self.epochs - 1:
                self._save(epoch, stat)
            
            elif self.config.EPOCHS_PER_CHECKPOINT and (epoch + 1) % self.config.EPOCHS_PER_CHECKPOINT == 0:
                self._save(epoch, stat)

            if epoch != self.epochs - 1:
                self._log(stat)

        print("Training is finish")
        test_loss, test_acc = self._eval_epoch()
        stat.test_loss = test_loss
        stat.test_acc = test_acc
        stat.display()
        path = self._log(stat)
        return path
    
    @staticmethod
    def _plot(title: str, train_data: list, valid_data: list, output_dir: str) -> str:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(train_data, label='train')
        plt.plot(valid_data, label='valid')
        plt.title(title)
        plt.xlabel("epochs")
        plt.show()
        path = os.path.join(output_dir, title.lower())
        plt.savefig(path)
        return path
    
    @staticmethod
    def plot(history_path: str, output_dir: str) -> str:
        """plot the loss-epoch figure

        Args:
            history_path (str): file of history (in json)
            output_dir (str): dir to export the result figure

        Returns:
            str: path to result figure
        """
        import json
        import pandas as pd
        history: History = json.load(history_path)
        
        df = pd.DataFrame(history["history"])
        ModelUtils._plot("Loss", df["train_loss"].tolist(), df["valid_loss"].tolist(), output_dir)
        ModelUtils._plot("Accuracy", df["train_acc"].tolist(), df["valid_acc"].tolist(), output_dir)
        return
    
    # def inference(self, ):



from datetime import datetime
def formatted_now():
    return datetime.now().strftime("%Y%m%dT%H-%M-%S")
