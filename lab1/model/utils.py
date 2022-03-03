"""usage
python utils.py --log="/path/to/log/root"
"""
import os
from typing import Tuple
from dataset import Dataset
from mlp import MLP
import torch
import argparse

epochs_list = [2, 4, 8, 16]
neurons_list = [2, 4, 8, 16, 64, 128, 1024]
layers_list = [1, 2, 4, 8, 16]


def generate_markdown_tables(log_root: str) -> list:
    loader = Loader(log_root)
    neurons_list_str = [str(i) for i in neurons_list]
    tables = []
    loss_tables = []
    
    for epochs in epochs_list:
        
        # ---------------------- row 1 --------------------
        table = f"| Epoch: {epochs} | "
        table += " | ".join(neurons_list_str)
        table += " |\n"

        # ---------------------- row 2 --------------------
        table += "|"
        for i in range(len(neurons_list) + 1):
            table += " -: |"
        
        table += "\n"

        table_loss = table

        # ---------------------- row 3~ --------------------
        for layers in layers_list:
            table += f"| {layers} | "
            table_loss += f"| {layers} | "

            for neurons in neurons_list:
                accuracy, loss = loader.load(layers, neurons, epochs)
                table += f" {accuracy: .2f}% |"
                table_loss += f" {loss: .3f} |"

            table += "\n"
            table_loss += "\n"
        
        print(table)
        tables.append(table)
        loss_tables.append(table_loss)
    tables += loss_tables
    return tables

class Loader:
    log_root: str

    def __init__(self, log_root):
        self.log_root = log_root
        _, self.test_loader = Dataset().load()
        return

    def load(self, layers, neurons, epochs) -> Tuple[float, float]:
        """

        Args:
            layers (_type_): _description_
            neurons (_type_): _description_
            epochs (_type_): _description_

        Returns:
            Tuple[float, float]: accuracy, loss
        """
        log_sub_dir = f"l{layers}_n{neurons}"
        log_sub_dir = os.path.abspath(os.path.join(self.log_root, log_sub_dir))

        assert os.path.isdir(log_sub_dir)

        model_name = f"l{layers}_n{neurons}_e{epochs}_of_16"

        model_path = os.path.join(log_sub_dir, model_name)

        assert os.path.isfile(model_path)

        model = torch.load(model_path)
        model = MLP(**model)

        return model.start_eval(self.test_loader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--log", required=True, metavar="/path/to/log/root/")

    args = parser.parse_args()
    
    with open(os.path.join("..", "tables.txt"), "w") as fin:
        tables = generate_markdown_tables(args.log)
        tables = "\n".join(tables)
        fin.write(tables)


