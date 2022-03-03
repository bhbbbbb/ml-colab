from model.dataset import Dataset
from model.mlp import MLP

epochs_list = [2, 4, 8, 16]
neurons_list = [2, 4, 8, 16, 64, 128, 1024]
layers_list = [1, 2, 4, 8, 16]

def train_all():
    for layers in layers_list:
        for neurons in neurons_list:
            model = MLP(neurons=neurons, layers=layers, epochs=16, learning_rate=1)
            model.start_train(trainloader, checkpoints=epochs_list)
            model.start_eval(testloader)
    return

def train_from_checkpoint(log_path: str, epochs: int, checkpoints: list):
    import torch
    log = torch.load(log_path)
    log["epochs"] = epochs

    print(f"load weigths from: {log_path}")
    model = MLP(**log)
    model.start_train(trainloader, checkpoints=checkpoints)
    model.start_eval(testloader)
    return


if __name__ == "__main__":
    
    trainloader, testloader = Dataset().load()
    model = MLP(neurons=16, layers=16, epochs=1024, learning_rate=0.0001)
    model.start_train(trainloader, checkpoints=[(16*i) for i in range(1, (1024//16)+1)])
    model.start_eval(testloader)
    # train_from_checkpoint(log_path="D:\Documents\\1102\AI\ml-practices\lab1\model\log\\relu\l16_n16\l16_n16_e16_of_16",
    #     epochs=128, checkpoints=[(16*i) for i in range(1, (1024 // 16) + 1)])
    # train_all()

