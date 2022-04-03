from typing import Tuple
import torch
from torch import Tensor
import torch.cuda.amp as amp
from tqdm import tqdm
from imgclf.base.model_utils import BaseModelUtils
# from imgclf.nfnets import SGD_AGC, pretrained_nfnet # pylint: disable=no-name-in-module
from nfnets import SGD_AGC, pretrained_nfnet # pylint: disable=no-name-in-module
from imgclf.dataset import Dataset
from .config import NfnetConfig

class NfnetModelUtils(BaseModelUtils):

    def __init__(self, pretrained_path: str, config: NfnetConfig):

        model = pretrained_nfnet(pretrained_path)

        optimizer = SGD_AGC(
            # The optimizer needs all parameter names 
            # to filter them by hand later
            named_params=model.named_parameters(), 
            lr=config.learning_rate,
            momentum=config.momentum,
            clipping=config.clipping,
            weight_decay=config.weight_decay, 
            nesterov=config.nesterov,
        )
        # Find desired parameters and exclude them 
        # from weight decay and clipping
        for group in optimizer.param_groups:
            name = group['name'] 
            
            if model.exclude_from_weight_decay(name):
                group['weight_decay'] = 0

            if model.exclude_from_clipping(name):
                group['clipping'] = None

        # criterion = nn.CrossEntropyLoss()
        self.scaler = amp.GradScaler()
        return

    def _train_epoch(self, train_dataset: Dataset) -> Tuple[float, float]:
        self.model.train()
        dataloader = train_dataset.data_loader
        running_loss = 0.0
        correct_labels = 0
        for inputs, targets in tqdm(dataloader):

            inputs: Tensor = inputs.half().to(self.config.device)\
                if self.config['use_fp16'] else inputs.to(self.config.device)
            
            targets: Tensor = targets.to(self.config.device)

            self.optimizer.zero_grad()

            with amp.autocast(enabled=self.config['amp']):
                output = self.model(inputs)
            loss: Tensor = self.criterion(output, targets)
            
            # Gradient scaling
            # https://www.youtube.com/watch?v=OqCrNkjN_PM
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

        # epoch_padding = int(math.log10(epochs) + 1)
        # batch_padding = int(math.log10(len(dataloader.dataset)) + 1)
        # print(f"\rEpoch {epoch+1:0{epoch_padding}d}/{config['epochs']}"
        #     f"\tImg {processed_imgs:{batch_padding}d}/{len(dataloader.dataset)}"
        #     f"\tLoss {running_loss / (step+1):6.4f}"
        #     f"\tAcc {100.0*correct_labels/processed_imgs:5.3f}%\t",
        # sep=' ', end='', flush=True)
        running_loss = running_loss / len(train_dataset)
        train_acc = correct_labels / len(train_dataset)
        return running_loss, train_acc
    
    def _eval_epoch(self, eval_dataset: Dataset) -> Tuple[float, float]:
        self.model.eval()

        correct_labels = 0
        eval_loss = 0.0
        for inputs, targets in eval_dataset:
            with torch.no_grad():
                inputs: Tensor = inputs.to(self.config.device)
                targets: Tensor = targets.to(self.config.device)

                output = self.model(inputs).type(torch.float32)

                loss: Tensor = self.criterion.forward(output, targets)
                eval_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(output, 1)
                correct_labels += (predicted == targets).sum().item()

        eval_loss = correct_labels / len(eval_dataset)
        eval_acc = correct_labels / len(eval_dataset)
        return eval_loss, eval_acc