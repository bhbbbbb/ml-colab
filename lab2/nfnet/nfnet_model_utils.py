import os
from typing import Tuple
import torch
from torch import Tensor
import torch.cuda.amp as amp
from tqdm import tqdm
from imgclf.base.model_utils import BaseModelUtils
from imgclf.base.model_utils.model_utils import ModelStates, HistoryUtils
from imgclf.base.logger import Logger
# from imgclf.nfnets import SGD_AGC, pretrained_nfnet, NFNet # pylint: disable=no-name-in-module
from nfnets import SGD_AGC, pretrained_nfnet, NFNet # pylint: disable=no-name-in-module
from .model import MyNfnet
from imgclf.dataset import Dataset
from .config import NfnetConfig

class NfnetModelUtils(BaseModelUtils):

    def __init__(
        self,
        model: torch.nn.modules,
        config: NfnetConfig,
        optimizer: torch.optim.Optimizer,
        start_epoch: int,
        root: str,
        history_utils,
        logger,
    ):
        self.scaler = amp.GradScaler()
        super().__init__(
            model = model,
            config = config,
            optimizer = optimizer,
            start_epoch = start_epoch,
            root = root,
            history_utils = history_utils,
            logger = logger,
        )
        return
    
    @staticmethod
    def _inti_model_optimizer(model: NFNet, config: NfnetConfig):

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
            name = group["name"] 
            
            if model.exclude_from_weight_decay(name):
                group["weight_decay"] = 0

            if model.exclude_from_clipping(name):
                group["clipping"] = None
        return model, optimizer
    
    @staticmethod
    def init_model(config: NfnetConfig):
        model = MyNfnet(
            num_classes = config.num_class,
            variant = config["variant"], 
            stochdepth_rate=config["stochdepth_rate"], 
            alpha=config["alpha"],
            se_ratio=config["se_ratio"],
            activation=config["activation"]
        )
        model.to(config.device)
        return model
    
    @classmethod
    def start_new_training_from_pretrained(cls, pretrained_path: str, config: NfnetConfig):

        pretrained_model = pretrained_nfnet(pretrained_path)
        model = cls.init_model(config)
        model_state = pretrained_model.state_dict()
        model_state = MyNfnet.fix_output_layer(model_state, config.num_class)
        # for name, params in model_state.items():
        #     print(name)
        #     print(params.shape)
        
        model.load_state_dict(model_state)
        model, optimizer = cls._inti_model_optimizer(model, config)

        return super().start_new_training(model, config, optimizer)
    
    @classmethod
    def load_checkpoint(cls, model: NFNet, checkpoint_path: str,
                        config: NfnetConfig, optimizer = None):
        assert os.path.isfile(checkpoint_path)

        tem = torch.load(checkpoint_path, map_location=torch.device(config.device))
        checkpoint = ModelStates(**tem)

        model, optimizer = cls._inti_model_optimizer(model, config)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        
        root = os.path.dirname(checkpoint_path)
        logger = Logger(root)
        start_epoch = checkpoint.start_epoch
        history_utils = HistoryUtils.load_history(root, start_epoch, logger)
        logger.log(f"Checkpoint {os.path.basename(checkpoint_path)} is loaded.")
        return cls(
            model = model,
            config = config,
            optimizer = optimizer,
            start_epoch = start_epoch,
            root = root,
            history_utils = history_utils,
            logger = logger,
        )

    def _train_epoch(self, train_dataset: Dataset) -> Tuple[float, float]:
        self.model.train()
        dataloader = train_dataset.data_loader
        running_loss = 0.0
        correct_labels = 0
        for inputs, targets in tqdm(dataloader):

            inputs: Tensor = inputs.half().to(self.config.device)\
                if self.config["use_fp16"] else inputs.to(self.config.device)
            
            targets: Tensor = targets.to(self.config.device)

            self.optimizer.zero_grad()

            with amp.autocast(enabled=self.config["amp"]):
                output = self.model(inputs)
            loss: Tensor = self.criterion(output, targets)
            
            # Gradient scaling
            # https://www.youtube.com/watch?v=OqCrNkjN_PM
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tem = loss.item() * inputs.size(0)
            print(type(tem), flush=True)
            running_loss += tem
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

        # epoch_padding = int(math.log10(epochs) + 1)
        # batch_padding = int(math.log10(len(dataloader.dataset)) + 1)
        # print(f"\rEpoch {epoch+1:0{epoch_padding}d}/{config["epochs"]}"
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
        for inputs, targets in tqdm(eval_dataset.data_loader):
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