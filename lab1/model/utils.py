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

def generate_cool_tables(log_root: str, checkpoints=epochs_list, eta: float = 0.001, enlarge_coef: int = 0) -> list:
    neurons_list_str = [str(i) for i in neurons_list]
    
    if not enlarge_coef:
        enlarge_coef = 64 // checkpoints[-1]
    # ---------------------- row 1 --------------------
    table = f"| Epoch: 1~{checkpoints[-1]} | "
    table += " | ".join(neurons_list_str)
    table += " |\n"

    # ---------------------- row 2 --------------------
    table += "|"
    for i in range(len(neurons_list) + 1):
        table += " -: |"
    
    table += "\n"

    # ---------------------- row 3~ --------------------
    for layers in layers_list:
        table += f"| {layers} | "

        for neurons in neurons_list:
            last_epoch = 0
            acc = []
            for epochs in checkpoints:
                accuracy, loss = load(log_root, layers, neurons, learning_rate=eta, epoch=epochs)
                width = epochs - last_epoch
                last_epoch = epochs
                acc.append((width, accuracy))

            table += f" {get_html_color_box(acc, enlarge_coef)} |"

        table += "\n"
    
    print(table)
    return table

from typing import List, Tuple
from math import floor
def get_html_color_box(acc: List[Tuple[int, float]], enlarge_coef:int = 1) -> str:
    """_summary_

    Args:
        width (int): _description_
        acc (List[Tuple[int, float]]): [ (width, accuracy),... ]

    Returns:
        str: html code
    """
    html = '<div style="display: inline-flex;">'
    BASE = 5
    for width, accuracy in acc:
        if accuracy < 10.0: accuracy = 10.0
        red = (100 - accuracy) / 90 # map to 0 ~ 1
        red = (BASE ** (-1 * red) - 1) / (BASE ** (-1) - 1) 
        red = floor(255 * red)
        # green = 255 * (accuracy - 10) // 90
        color = f"rgb({red},255,0)"
        html += f'<div style="width: {width * enlarge_coef}px; height: 10px; background-color: {color};"></div>'

    html += '</div>'
    return html

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
    # parser.add_argument("--eta", required=True, metavar="1e-5")

    args = parser.parse_args()
    
    # table = generate_cool_tables(args.log, checkpoints=(epochs_list), eta=0.001)
    table = generate_cool_tables(args.log, checkpoints=(epochs_list + [(i*8) for i in range(3, 256//8 + 1)]),
                eta=1e-5, enlarge_coef=1)
    # with open(os.path.join("..", "tables.txt"), "w") as fin:
    #     tables = generate_markdown_tables(args.log)
    #     tables = "\n".join(tables)
    #     fin.write(tables)


