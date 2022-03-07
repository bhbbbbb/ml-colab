"""usage
python utils.py --log="/path/to/log/root"
"""
import os
from typing import Tuple
# from dataset import Dataset
from mlp import Model, ModelStates
import torch
import argparse

epochs_list = [2, 4, 8, 16]
neurons_list = [2, 4, 8, 16, 64, 128, 1024]
layers_list = [1, 2, 4, 8, 16]


def generate_markdown_tables(log_root: str) -> list:
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
                accuracy, loss = load(log_root, layers, neurons, learning_rate=0.001, epoch=epochs)
                table += f" {accuracy: .2f}% |"
                table_loss += f" {loss: .3f} |"

            table += "\n"
            table_loss += "\n"
        
        print(table)
        tables.append(table)
        loss_tables.append(table_loss)
    tables += loss_tables
    return tables

def load(log_root, layers, neurons, learning_rate, epoch) -> Tuple[float, float]:
    """
    Returns:
        Tuple[float, float]: accuracy, loss
    """
    log_sub_dir = f"l{layers}_n{neurons}_eta{learning_rate}"
    log_sub_dir = os.path.abspath(os.path.join(log_root, log_sub_dir))

    assert os.path.isdir(log_sub_dir)

    model_name = f"epoch_{epoch}"

    model_path = os.path.join(log_sub_dir, model_name)

    assert os.path.isfile(model_path)

    model: Model = torch.load(model_path)
    states: ModelStates = model["model_states"]
    return states["val_acc"], states["loss"]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--log", required=True, metavar="/path/to/log/root/")

    args = parser.parse_args()
    
    with open(os.path.join("..", "tables.txt"), "w") as fin:
        tables = generate_markdown_tables(args.log)
        tables = "\n".join(tables)
        fin.write(tables)


