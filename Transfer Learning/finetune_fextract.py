from __future__ import print_function
from __future__ import division
from matplotlib.style import use
from requests import get
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets,  models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import argparse

#print the version of PyTorch and Torchvision used
print("PyTorch Version:", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, default='squeezenet')
args = parser.parse_args()

train_data_root = "./natural_scene_data/seg_train"
test_data_root = "./natural_scene_data/seg_test"
num_classes = 6
batch_size = 10
epoch = 20
input_size = 224
learning_rate = 0.001
weight_decay = 1e-4

#Load data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_data = datasets.ImageFolder(train_data_root, transform=data_transforms['train'])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data = datasets.ImageFolder(test_data_root, transform=data_transforms['test'])
test_dataloader =DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

#By default the requires_grad for all the parameters is True which means train from scratch, if True then requires_grad
#of reshaped layer should only be True
#(feature_extract) True: update only reshaped layer False: finetune entire network

def set_feature_extracting(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    finetune_model = None
    if model_name=='squeezenet':
        finetune_model = models.squeezenet1_0(pretrained = use_pretrained)
        set_feature_extracting(finetune_model, feature_extract)
        finetune_model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=(1,1), stride=(1,1))
        finetune_model.num_classes = num_classes
    
    elif model_name == 'resnet':
        finetune_model = models.resnet18(pretrained = use_pretrained)
        set_feature_extracting(finetune_model, feature_extract)
        num_filters = finetune_model.fc.in_features
        finetune_model.fc = nn.Linear(num_filters, num_classes)

    elif model_name == 'alexnet':
        finetune_model = models.alexnet(pretrained = use_pretrained)
        set_feature_extracting(finetune_model, feature_extract)
        num_filters = finetune_model.classifier[6].in_features
        finetune_model.classifier[6] = nn.Linear(num_filters, num_classes)
    
    elif model_name == 'vgg':
        finetune_model = models.vgg11_bn(pretrained = use_pretrained)
        set_feature_extracting(finetune_model, feature_extract)
        num_filters = finetune_model.classifier[6].in_features
        finetune_model.classifier[6] = nn.Linear(num_filters, num_classes)
    
    elif model_name == 'densenet':
        finetune_model = models.densenet121(pretrained = use_pretrained)
        set_feature_extracting(finetune_model, feature_extract)
        num_filters = finetune_model.classifier.in_features
        finetune_model.classifier = nn.Linear(num_filters, num_classes)

    else:
        print("Invalid model name..")
        exit()
    
    return finetune_model

training_loss = []
validation_loss = []
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epoch):
    print("...Start training...")
    for i in tqdm(range(epoch)):
        model.train()
        train_running_loss = []
        for t_input, t_label in train_dataloader:
            t_input = t_input.to(device)
            t_label = t_label.to(device)
            optimizer.zero_grad()
            pred = model(t_input)
            loss = criterion(pred, t_label)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Epoch {} training loss {}".format(i+1, np.mean(train_running_loss)))
        training_loss.append(np.mean(train_running_loss))

        #validate the model
        model.eval()
        val_running_loss = []
        for v_input, v_label in val_dataloader:
            v_input = v_input.to(device)
            v_label = v_label.to(device)
            pred = model(v_input)
            loss = criterion(pred, v_label)
            val_running_loss.append(loss.item())
        print("Epoch {} validation loss {}".format(i+1, np.mean(val_running_loss)))
        validation_loss.append(np.mean(val_running_loss))

    return model, training_loss, validation_loss

def get_params_to_learn(model, feature_extract):
    params_to_learn = model.parameters()
    if feature_extract:
        params_to_learn = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_learn.append(param)

    return params_to_learn

def plot_loss(train_loss, test_loss, path):
    x = np.arange(0, len(train_loss), 1, dtype=int)
    plt.plot(x, train_loss, label = 'Training Loss')
    plt.plot(x, test_loss, label = 'Validation Loss')
    plt.title("Training and Validation Plots")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.show()

# #training only last layer i.e feature extraction
fextract_model = initialize_model(args.model_name, num_classes, feature_extract=True)
params_to_learn = get_params_to_learn(fextract_model, True)

fextract_model = fextract_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_learn, lr=0.001, weight_decay=weight_decay)

fextract_model, training_loss, validation_loss = train_model(fextract_model, train_dataloader, test_dataloader, criterion, optimizer, epoch)
plot_loss(training_loss, validation_loss, './fextract_plot.png')

#training entire network i.e fine tuning
finetune_model = initialize_model(args.model_name, num_classes, feature_extract=False)
params_to_learn = get_params_to_learn(finetune_model, feature_extract=False)

finetune_model = finetune_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_learn, lr=0.001, weight_decay=weight_decay)

finetune_model, training_loss, validation_loss = train_model(finetune_model, train_dataloader, test_dataloader, criterion, optimizer, epoch)
plot_loss(training_loss, validation_loss, './finetune_plot.png')



