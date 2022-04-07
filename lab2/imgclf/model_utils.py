from typing import Tuple
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import pandas as pd
import numpy as np

from .dataset import Dataset
from .base.model_utils import BaseModelUtils 


class ModelUtils(BaseModelUtils):

    @staticmethod
    def _get_optimizer(model, config):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    @staticmethod
    def _get_criterion(config):
        return nn.CrossEntropyLoss()

    def _train_epoch(self, train_dataset: Dataset) -> Tuple[float, float]:
        """train a single epoch

        Returns:
            Tuple[float, float]: train_loss, train_acc
        """
        self.model.train()

        train_loss = 0.0
        train_correct = 0
        
        for data, label in tqdm(train_dataset.data_loader):

            data: Tensor
            label: Tensor

            data, label = data.to(self.config.device), label.to(self.config.device)
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
        
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        return train_loss, train_acc
    
    def _eval_epoch(self, eval_dataset: Dataset) -> Tuple[float, float]:
        """evaluate single epoch

        Returns:
            Tuple[float, float]: eval_loss, eval_acc
        """
        self.model.eval()
        eval_loss = 0.0
        correct = 0

        for data, target in eval_dataset.data_loader:
            data: Tensor
            target: Tensor

            data, target = data.to(self.config.device), target.to(self.config.device)

            output: Tensor = self.model(data)

            loss = self.criterion.forward(output, target)

            eval_loss += loss.item() * data.size(0)
            
            _, predicted = output.max(dim=1)
            correct += (predicted == target).sum().item()
        
        eval_loss = eval_loss / len(eval_dataset)
        eval_acc = correct / len(eval_dataset)
        return eval_loss, eval_acc
    
    
    def inference(self, dataset: Dataset, categories: list = None, confidence: bool = True):
        """inference for the given test dataset

        Args:
            confidence (boolean): whether output the `confidence` column. Default to True.
        
        Returns:
            df (pd.DataFrame): {"label": [...], "confidence"?: [...]}
        """

        categories = categories or list(range(self.config.num_class))
        
        def mapping(x):
            return categories[x]

        label_col = np.empty(len(dataset), dtype=type(categories[0]))
        if confidence:
            confidence_col = np.empty(len(dataset), dtype=float)
            data = {"label": label_col, "confidence": confidence_col}
        
        else:
            data = {"label": label_col}
        
        df = pd.DataFrame(data)
        
        with torch.inference_mode():
            for data, indexes in tqdm(dataset.data_loader):
                data: Tensor
                indexes: Tensor
                data = data.to(self.config.device)
                output: Tensor = self.model(data)

                confidences, indices = output.max(dim=1)

                labels = list(map(mapping, indices.tolist()))
                
                indexes = indexes.tolist()
                if confidence:
                    df.loc["label", indexes] = labels
                    df.loc["confidence", indexes] = confidences.tolist()
                else:
                    df.iloc[indexes] = labels
        
        return df
