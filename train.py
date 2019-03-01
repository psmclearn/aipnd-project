#Imports here

import os
import sys
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from collections import OrderedDict
import time
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms , models
#from model_util import get_input_args, make_folder
#from data import build_vocab, get_coco_data, get_iterator

# Reference Material : https://www.pythonforbeginners.com/system/python-sys-argv

def main():
    # Set up defaults before getting command line arguments
    global architecture 
    checkpoint_dir= 'checkpoint_directory'
    data_directory = 'flowers'
    architecture = 'vgg16'
    learning_rate = 0.001
    hidden_units = 512
    epochs = 3
    device = 'cuda'
    gpu = False

    # Get command line arguments for all parameters

    print("The args are: ",sys.argv)
    len_args = len(sys.argv)
    if(len_args==0):
        error('Please provide a valid data directory')
    data_directory = sys.argv[1]
    print('The data directory for images is :',data_directory)
    for i in range(2,len(sys.argv)):
        if sys.argv[i] == '--save_dir':
            if i == len_args - 1:
                continue
            print('Save directory provided for checkpoint',sys.argv[i+1])
            checkpoint_dir = sys.argv[i+1]
            i = i + 1
        if sys.argv[i] == '--arch':
            if i == len_args - 1:
                                continue
            print('Preferred Architecture for the model was provided',sys.argv[i+1])
            architecture = sys.argv[i+1]
            i = i + 1
        if sys.argv[i] == '--learning_rate':
            if i == len_args - 1:
                                continue
            print('Learning rate was provided',sys.argv[i+1])
            learning_rate = float(sys.argv[i+1])
            i = i + 1
        if sys.argv[i] == '--hidden_units':
            if i == len_args - 1:
                                continue
            print('Hidden units were provided',sys.argv[i+1])
            hidden_units = int(sys.argv[i+1])
            i = i + 1
        if sys.argv[i] == '--epochs':
            if i == len_args - 1:
                                continue
            print('Number of epochs was provided',sys.argv[i+1])
            epochs = int(sys.argv[i+1])
            i = i + 1
        if sys.argv[i] == '--gpu':
            print('Use gpu for training')
            gpu = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            i = i + 1
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    transformations(data_dir,train_dir,valid_dir,test_dir)
    model = build_model(architecture)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if gpu == False:
         device = 'cpu'
    train_model(model,epochs,device,criterion,optimizer)
    test_model(model,epochs,device,criterion,optimizer)
    save_checkpoint(model,optimizer,checkpoint_dir)

def build_model(architecture): #### TODO: Support other models as well ####
    global in_features
    model = models.vgg16(pretrained = True)
    in_features = 25088
    if architecture == 'densenet':
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
	
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, 2048)),
                                  ('relu1', nn.ReLU()),
                                  ('fc2', nn.Linear(2048, 102)),
                                      ('output', nn.LogSoftmax(dim=1))
                                  ]))

        model.classifier = classifier
        return model

# Reference Material : Transfer Learning Chapter of Udacity course
def transformations(data_dir,train_dir,valid_dir,test_dir):
    global train_data_transforms,valid_data_transforms,test_data_transforms,train_dataset,valid_dataset,test_dataset,traindataloaders,validdataloaders,testdataloaders

    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, train_data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    traindataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    validdataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=0)
    testdataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)

def train_model(model,epochs,device,criterion,optimizer):
    print('CHecking of cuda is available',torch.cuda.is_available())
    #torch.cuda.is_available())
    running_loss =0
    steps =0
    print_every = 40
    print('The device is :',device)
    model.to(device)
    print('The Device is ',device)
    for epoch in range(epochs):
        for inputs, labels in traindataloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Calculating!!!")
                validation_loss = 0
                validation_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for valid_inputs, valid_labels in validdataloaders:
                        valid_inputs, valid_labels = inputs.to(device), labels.to(device)
                        logits = model.forward(valid_inputs)
                        batch_loss = criterion(logits, valid_labels)
                        validation_loss += batch_loss.item()
                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == valid_labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}  "
                      f"Train loss: {running_loss/print_every:.3f}  "
                      f"Validation loss: {validation_loss/len(validdataloaders):.3f}  "
                      f"Validation accuracy: {validation_accuracy/len(validdataloaders):.3f}")                
               # print('Validation accuracy: {validation_accuracy/len(validdataloaders):.3f}')
                running_loss = 0
                model.train()

    print('Completed training the NN on the train data set ')

def test_model(model,epochs,device,criterion,optimizer):
    print('CHecking of cuda is available',torch.cuda.is_available())
    #torch.cuda.is_available())
    running_loss =0
    steps =0
    print_every = 40
    print('The device is :',device)
    model.to(device)
    print('The Device is ',device)
    for epoch in range(epochs):
        for inputs, labels in traindataloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Calculating!!!")
                test_loss = 0
                test_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for test_inputs, test_labels in testdataloaders:
                        test_inputs, test_labels = inputs.to(device), labels.to(device)
                        logits = model.forward(test_inputs)
                        batch_loss = criterion(logits, test_labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == test_labels.view(*top_class.shape)
                        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}   "
                      f"Train loss: {running_loss/print_every:.3f}   "
                      f"Test loss: {test_loss/len(validdataloaders):.3f}   "
                      f"Test accuracy: {test_accuracy/len(validdataloaders):.3f}")
                running_loss = 0
                model.train()

    print('Completed training the NN on the train data set ')

def save_checkpoint(model,optimizer,checkpoint_dir):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'classifier': model.classifier,
    'optimizer': optimizer,
    'arch': architecture,
    'state_dict': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx,
    'input_features':in_features
    }
    # Reference : https://tecadmin.net/python-check-file-directory-exists/
    #checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
    torch.save(checkpoint,checkpoint_dir+'/checkpoint.pth')

main()