from model.dataset import Dataset
from model.mlp import MLP

epochs_list = [2, 4, 8, 16]
neurons_list = [2, 4, 8, 16, 64, 128, 1024]
layers_list = [1, 2, 4, 8, 16]

def train_all():
    for layers in layers_list:
        for neurons in neurons_list:
            model = MLP(neurons=neurons, layers=layers, epochs=16)
            model.start_train(trainloader)
            model.start_eval(testloader)
    return

if __name__ == "__main__":
    
    trainloader, testloader = Dataset().load()
    train_all()

