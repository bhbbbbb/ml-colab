# %% [markdown]
# # Homework2: Convolution Neural Network for classification
# 
# In this homework, we are going to learn:
# 1. How to preprocess and load data in pytorch
# 2. How to build a CNN model for training classification
# 3. Training/Validation Process and plot the result.
# 4. Testing Inference.
# 
# <p align="center">
# <img src="https://miro.medium.com/max/895/1*RjZe7cfnhdRhhLimLvapow.png" width="800">
# </p>

# %% [markdown]
# ## 0.1 Preparation

# %%
import os
import csv
import torch
import numpy as np
import pandas as pd

from PIL import Image
from google.colab import drive

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use device:",device)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
##############################################
# Decide your work and save path
##############################################
DataPath = '/content/hw2_ex/'
SavePath = '/content/drive/MyDrive/Colab Notebooks/homework2/'

os.makedirs(DataPath, exist_ok = True)
os.makedirs(SavePath, exist_ok = True)

# %% [markdown]
# ## 0.2 Download DataSet: Cat and Dog
# 
# <p align="center">
# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg" width="400">
# </p>

# %%
if not os.path.isdir(DataPath+'/dataset'):
    !pip install --upgrade --no-cache-dir gdown
    !gdown --id 1hj2zrZI3Nd-C6nlGOE1crgR_gnpoKHQh --output 'dataset.zip'
    !unzip -q dataset.zip -d '/content/hw2_ex' # the -d should be the same as DataPath

else:
    print("File already exists.")

# %% [markdown]
# # 1 Data Preprocess
# In this chapter, we will learn how to preprocess image data.
# 
# Using pandas to preprocess, it is a good module for data analytic.
# 
# Using dataset and dataloader from pytorch to setup training dataflow.

# %% [markdown]
# ## 1.1 Pandas DataFrame
# >[documentation](https://pandas.pydata.org/docs/index.html)
# >
# >Pandas is a Python library used for working with data sets.
# It has functions for analyzing, cleaning, exploring, and manipulating data.
# >
# >The name "Pandas" has a reference to both "Panel Data", and "Python Data Analysis" and was created by Wes McKinney in 2008.
# 
# 
# <p align="center">
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width="600">
# </p>

# %%
import pandas as pd

df = pd.read_csv(os.path.join(DataPath, 'train.csv'))
df.describe()

# %%
df.head()

# %%
df['label']=='cat'

# %%
df['img'][df['label']=='dog']

# %%
df.at[10,'img']

# %%
df['imgpath'] = DataPath + df['img'].astype(str)
df.head()

# %%
Class = {'cat':0, 'dog':1}
df['category'] = df['label'].astype('category').cat.codes
df

# %%
df.iloc[0]

# %% [markdown]
# ## 1.2 Preprocess data to list
# 
# We compare two kinds of way for preprocessing.
# We can find that using the same framework for preprocessing would be efficient.

# %%
##############################################
# Create a list for downloaded files,
#  including the path of the image 
#  and the corresponding label.
##############################################
import time


train_img_path = os.path.join(DataPath, 'train_dataset/')
df = pd.read_csv(os.path.join(DataPath, 'train.csv'))

# small experiment
start_time = time.time()

Class = {'cat':0, 'dog':1}
imgpath = []
imglabel = []

# Load csv and add path
for idx, row in df.iterrows():
    imgpath.append(os.path.join(train_img_path, str(row["img"])))
    imglabel.append(Class[row["label"]])
print(f"Process with for loop in {(time.time()-start_time):.3f} s\n\n")

idx = np.random.randint(len(imgpath), size=10)
print(np.array(imgpath)[idx])
print(np.array(imglabel)[idx])

# %%
# small experiment
start_time = time.time()
imgpath = (train_img_path + df['img'].astype(str)).tolist()
imglabel = df['label'].astype('category').cat.codes.tolist()
print(f"Process with pandas framework in {(time.time()-start_time):.3f} s\n\n")

print(np.array(imgpath)[idx])
print(np.array(imglabel)[idx])

# %% [markdown]
# ## 1.3 Pytorch Dataset & Dataloaders
# >[documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
# >
# >Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives:  `torch.utils.data.DataLoader`  and  `torch.utils.data.Dataset`  that allow you to use pre-loaded datasets as well as your own data. 
# >
# >Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
# 
# 
# <p align="center">
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" width="600">
# </p>

# %% [markdown]
# ### 1.3.1 torchvision.transforms
# 
# >[documentation](https://pytorch.org/vision/stable/transforms.html)
# >
# >`torchvision.transforms` help us to transform image to tensor and can also help us apply augmentation, for more robust training.
# 
# <p align="center">
# <img src="https://pytorch.org/vision/stable/_images/sphx_glr_plot_transforms_024.png" width="600">
# </p>
# 

# %%
# you need change image to tensor
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

# %% [markdown]
# ### 1.3.2 torch.utils.data.Dataset

# %%
##############################################
# Use the list you created above
#  to create a class for DataLoader.
##############################################
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, imgpath, csvpath, transform = valid_transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        df = pd.read_csv(csvpath)
        self.images = (imgpath + df['img'].astype(str)).tolist()
        self.label = df['label'].astype('category').cat.codes.tolist()
        self.transform = transform
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.images[index]
        img = Image.open(imgpath).convert('RGB')
        label = self.label[index]
        img = self.transform(img)

        return img, label
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.images)


# %%
from torch.utils.data import random_split

imgpath = os.path.join(DataPath, 'train_dataset/')
csvpath = os.path.join(DataPath, 'train.csv')

trainset = dataset(imgpath, csvpath)
train_set_size = int(len(trainset) * 0.7)
valid_set_size = int(len(trainset) * 0.15)
test_set_size = len(trainset) - train_set_size - valid_set_size

trainset, validset, testset = random_split(trainset, [train_set_size, valid_set_size, test_set_size])

trainset.transform = train_transform
validset.transform = valid_transform
testset.transform = valid_transform

print(f'trainset: {len(trainset)}\nvalidset: {len(validset)}\ntestset: {len(testset)}')

idx = np.random.randint(len(trainset))
print(f'{idx:5d}/{len(trainset)} : {trainset[idx]}')

# %% [markdown]
# ### 1.3.3 torch.utils.data.DataLoader

# %%
# Loaded Datasets to DataLoaders

# batch_size also affect training step.
# higher: faster, stable
#   but inprecise on optimize, may use large memory

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers = 2)
validloader = DataLoader(validset, batch_size=64, shuffle=False, num_workers = 2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers = 2)

# %% [markdown]
# # 2 CNN Model

# %% [markdown]
# ### 2.1 Convolutional layers
# >[document](https://pytorch.org/docs/stable/nn.html#convolution-layers)
# >
# >Start to build our first CNN Model!

# %%
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=3),
    )

result = model(torch.rand((64 ,3, 28, 28)))
print(result.shape)

# %%
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    )
result = model(torch.rand((3, 3, 512, 1024)))
print(result.shape)

# %% [markdown]
# <p align="center">
# <img src="https://1.bp.blogspot.com/-db1Bv-YnKTo/XkrIvCYFtEI/AAAAAAAAArA/2b0zbD29CQML4IKxp_1Zng_r0ioaOCZkgCEwYBhgL/s1600/appendix_C_conv_formula.png" width="600">
# </p>
# 

# %%
data_flow = torch.rand((3, 3, 512, 1024))
print(data_flow.shape)

data_flow = nn.Conv2d(3,16,5)(data_flow)
print(data_flow.shape)

data_flow = nn.MaxPool2d(2,2)(data_flow)
print(data_flow.shape)

data_flow = nn.Conv2d(16,64,5)(data_flow)
print(data_flow.shape)

data_flow = nn.MaxPool2d(2,2)(data_flow)
print(data_flow.shape)

data_flow = nn.Conv2d(64,128,5)(data_flow)
print(data_flow.shape)

print((data_flow.shape))

# %% [markdown]
# ### 2.2 Convolutional model
# 
# Try to implement LeNet-5 with pytorch !
# 
# (but using `nn.ReLU()` as activation function)
# 
# <p align="center">
# <img src="https://cdn-images-1.medium.com/max/1024/1*DMcPgeekUftwk0GTMcNawg.png" width="900">
# </p>

# %%
##############################################
# Build your model here!
# 
# Practice:
#   Try to implement LeNet-5 with pytorch !
##############################################
class trainmodel(nn.Module):
    def __init__(self):
        super(trainmodel, self).__init__()
        self. conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # To Do
            )
        self.cls = nn.Sequential(
            nn.Linear(5*5*16,120),
            # To Do
            nn.Linear(84,2),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,5*5*16)       # Change if you modify network
        x = self.cls(x)
        return x

model = trainmodel()
model.to(device)

# %%
## test your model sequential output
conv = nn.Sequential(
    nn.Conv2d(3,16,3),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Conv2d(16,64,3),
    nn.ReLU(),
    nn.MaxPool2d(3,2),
    nn.Conv2d(64,128,3),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    )
batch = torch.rand(3,3,224,224)    
result = conv(batch) #It is your img input
print(result.shape)



# %% [markdown]
# ## 2.3 How to improve Network?
# 
# ### Layer
# - Adjustment of convolution/pooling layer
# - [Activation Layer](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
# - Global [Average](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d)/[Max](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d) Pooling
# 
# ### Training Robustness
# - [Augmentation](https://pytorch.org/vision/stable/transforms.html)
# - [Batch Normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
# - [Dropout Layers](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)
# 
# ### Optimizer: attempt to find global minimum
# - [SGD with momentum](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
# - [ADAM](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
# - [torch.optim](https://pytorch.org/docs/stable/optim.html)
# 
# ### Learning Rate Sceduler
# - [Cosine Annealing Learning Rate](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
# 

# %%
##############################################
# Build your model here!
# 
# Practice:
#   Add data augmentation !
##############################################

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Todo
    transforms.ToTensor(),
    ])

trainset.transform = train_transform
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers = 2)

# %%
##############################################
# Build your model here!
# 
# Practice:
#   Improve your own Model!
##############################################

class trainmodel(nn.Module):
    def __init__(self):
        super(trainmodel, self).__init__()
        # To do

    def forward(self, x):
        # To do
        return x

model = trainmodel()
model.to(device)

model = trainmodel()
model.to(device)

# %% [markdown]
# # 3 Training Routine

# %% [markdown]
# ## 3.1 Training module

# %%
def train(model, trainloader, optimizer, criterion):
    # keep track of training loss
    train_loss = 0.0
    train_correct = 0
    
    # train the model 
    model.train()
    for data, target in tqdm(trainloader):
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        # update training Accuracy
        _, predicted = torch.max(output.data, 1)
        train_correct += (predicted == target).sum().item()

    return train_loss/len(trainloader.dataset), train_correct/len(trainloader.dataset)

# %%
def test(model, testloader, criterion):
    # keep track of validation loss
    valid_loss = 0.0
    valid_correct = 0

    # evaluate the model 
    model.eval()
    for data, target in tqdm(testloader):
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # update validation Accuracy
        _, predicted = torch.max(output.data, 1)
        valid_correct += (predicted == target).sum().item()

    return valid_loss/len(testloader.dataset), valid_correct/len(testloader.dataset)

# %%
def modeltrain(model, trainloader, validloader, testloader, optimizer, criterion, epochs, save_model_path, earlystop=4):
    history = {
        'trainloss' : [],
        'trainacc' : [],
        'validloss' : [],
        'validacc' : [],
    }
    state = {
        'epoch' : 0,
        'state_dict' : model.state_dict(),
        'trainloss' : 10000,
        'trainacc' : 0,
        'validloss' : 10000,
        'validacc' : 0,
    }
    valid_loss_min = 10000
    trigger = 0
    for epoch in range(epochs):
        print(f'running epoch: {epoch+1}')
        trainloss, trainacc = train(model, trainloader, optimizer, criterion)
        validloss, validacc = test(model, validloader, criterion)

        # print training/validation statistics 
        history['trainloss'].append(trainloss)
        history['trainacc'].append(trainacc)
        history['validloss'].append(validloss)
        history['validacc'].append(validacc)
        print(f'Training Loss  : {trainloss:.4f}\t\tTraining Accuracy  : {trainacc:.4f}')
        print(f'Validation Loss: {validloss:.4f}\t\tValidation Accuracy: {validacc:.4f}')
        
        # save model if validation loss has decreased
        if validloss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.4f} --> {validloss:.4f}).  Saving model ...\n')
            state['epoch'] = epoch
            state['state_dict'] = model.state_dict()
            state['trainloss'] = trainloss
            state['trainacc'] = trainacc
            state['validloss'] = validloss
            state['validacc'] = validacc

            torch.save(state, save_model_path)
            valid_loss_min = validloss
            trigger = 0
        # if model dont improve for 5 times, interupt.
        else:
            trigger += 1
            print(f'Validation loss increased ({valid_loss_min:.4f} --> {validloss:.4f}). Trigger {trigger}/{earlystop}\n')
            if trigger == earlystop:
                break
    print('\nTest Evaluate:')
    testloss, testacc = test(model, testloader, criterion)
    state['testloss'] = testloss
    state['testacc'] = testacc
    torch.save(state, save_model_path)
    bestepoch = state['epoch']
    validloss = state['validloss']
    validacc = state['validacc']
    print(f'Best model on epoch : {bestepoch}/{epoch}')
    print(f'validation loss: {validloss:.4f}\t\t validation acc : {validacc:.4f}')
    print(f'test loss      : {testloss:.4f}\t\t test acc \t: {testacc:.4f}')
    return history

# %% [markdown]
# ## 3.2 Start Training!

# %%
from tqdm.notebook import tqdm
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
n_epochs = 1
save_model_path = os.path.join(SavePath, '/model_weight.pth')

history = modeltrain(
        model = model,
        trainloader = trainloader,
        validloader = validloader,
        testloader = testloader,
        optimizer = optimizer,
        criterion = criterion,
        epochs = n_epochs,
        save_model_path = save_model_path
        )

# %% [markdown]
# ## 3.3 Plot the result

# %%
import json
save_history = json.dumps(history)
with open(os.path.join(SavePath, 'history.json'), 'w') as f:
    json.dump(save_history, f)

# %%
import matplotlib.pyplot as plt

def plot(name, savedir, trainhistory, validhistory):
    plt.figure(figsize=(10,5))
    plt.plot(trainhistory, label = 'train')
    plt.plot(validhistory,  label = 'valid')
    plt.title(name)
    plt.xlabel("epochs")
    plt.show()
    plt.savefig(savedir)

# %%
plot('Training Loss', os.path.join(SavePath,'loss.png'), history['trainloss'], history['validloss'])

# %%
plot('Training Accuracy', os.path.join(SavePath,'acc.png'), history['trainacc'], history['validacc'])

# %% [markdown]
# # 4 Inference testing data

# %% [markdown]
# ### 4.1 load weight

# %%
## create model as same as your training 
model = trainmodel()
model.to(device)

## load weight
state = torch.load(save_model_path)
# state['state_dict']
model.load_state_dict(state['state_dict'])

# %% [markdown]
# ### 4.2 test and save result

# %%
transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
])

class test_set(Dataset):
    def __init__(self, img_path, csv_path, transform = transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.df = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = os.path.join(self.img_path, self.df.iloc[index, 0])
        img = Image.open(imgpath)
        img = self.transform(img)

        return img, index
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.df)

# %%
import torch.nn.functional as F

#load test data
testimg_path = os.path.join(DataPath, 'test_dataset')
testcvs_path = os.path.join(DataPath, 'test.csv')

df = pd.read_csv(testcvs_path)

Class = {'cat':0, 'dog':1}
inv_Class = dict((v, k) for k, v in Class.items())

print(len(df))

testset = test_set(testimg_path, testcvs_path)
testloader = DataLoader(testset, batch_size=1, shuffle=False,pin_memory=True,num_workers = 2)

with torch.no_grad():
    for data, idx in tqdm(testloader):
        data = data.to(device)
        pred = F.softmax(model(data),dim=1)
        pred = np.argmax(pred.detach().cpu().numpy(),axis=1)

        df.at[idx, 'label'] = inv_Class[int(pred)] # convert predicted integer back to class string, only works when batch = 1

df.head()
df.to_csv(os.path.join(SavePath, 'result.csv'), encoding='utf-8')

# %%
## Only examples have "test_ans.csv"
df = pd.read_csv(os.path.join(SavePath, 'test_ans.csv'))
ans = df["label"].to_numpy()

df = pd.read_csv(SavePath+'/result.csv')
output = df["label"].to_numpy()

print("Accuracy:",((ans == output).sum().item()/len(df)))

# %% [markdown]
# # 5 What Next? [Transfer Learning](https://hackmd.io/@allen108108/H1MFrV9WH)
# 
# 
# <p align="center">
# <img src="https://miro.medium.com/max/1400/1*Ww3AMxZeoiB84GVSRBr4Bw.png" width="700">
# </p>

# %% [markdown]
# ### 5.1 Transfer Learning on Pytorch 
# [Transfer learning for computer vision tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
# 
# [Models and pre-trained weights](https://pytorch.org/vision/stable/models.html)
# 

# %%
"""
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
efficientnet_b1 = models.efficientnet_b1(pretrained=True)
efficientnet_b2 = models.efficientnet_b2(pretrained=True)
efficientnet_b3 = models.efficientnet_b3(pretrained=True)
efficientnet_b4 = models.efficientnet_b4(pretrained=True)
efficientnet_b5 = models.efficientnet_b5(pretrained=True)
efficientnet_b6 = models.efficientnet_b6(pretrained=True)
efficientnet_b7 = models.efficientnet_b7(pretrained=True)
regnet_y_400mf = models.regnet_y_400mf(pretrained=True)
regnet_y_800mf = models.regnet_y_800mf(pretrained=True)
regnet_y_1_6gf = models.regnet_y_1_6gf(pretrained=True)
regnet_y_3_2gf = models.regnet_y_3_2gf(pretrained=True)
regnet_y_8gf = models.regnet_y_8gf(pretrained=True)
regnet_y_16gf = models.regnet_y_16gf(pretrained=True)
regnet_y_32gf = models.regnet_y_32gf(pretrained=True)
regnet_x_400mf = models.regnet_x_400mf(pretrained=True)
regnet_x_800mf = models.regnet_x_800mf(pretrained=True)
regnet_x_1_6gf = models.regnet_x_1_6gf(pretrained=True)
regnet_x_3_2gf = models.regnet_x_3_2gf(pretrained=True)
regnet_x_8gf = models.regnet_x_8gf(pretrained=True)
regnet_x_16gf = models.regnet_x_16gf(pretrained=True)
regnet_x_32gf = models.regnet_x_32gf(pretrained=True)
vit_b_16 = models.vit_b_16(pretrained=True)
vit_b_32 = models.vit_b_32(pretrained=True)
vit_l_16 = models.vit_l_16(pretrained=True)
vit_l_32 = models.vit_l_32(pretrained=True)
convnext_tiny = models.convnext_tiny(pretrained=True)
convnext_small = models.convnext_small(pretrained=True)
convnext_base = models.convnext_base(pretrained=True)
convnext_large = models.convnext_large(pretrained=True)

"""


