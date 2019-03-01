
#Local Testing Values 
#  flower_data/train/1/image_06743.jpg  newDir/checkpoint.pth
#
#This file take in an image of a flower and predict the probabilities and the classes of the flowers
# Usage 
#  python predict.py flowers/test/1/image_06743.jpg checkpoint_directory (Using just the saved checkpoint andimage file)
#  python predict.py flowers/test/1/image_06743.jpg checkpoint_directory --top_k 3
#  python predict.py flowers/test/1/image_06743.jpg checkpoint_directory --category_names cat_to_name.json
#  python predict.py flowers/test/1/image_06743.jpg checkpoint_directory --gpu 
#
# AUTHOR : Preethi Sahayaraj
# DATE   : 27 Feb 2019
#
###############################################################################################################################

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
from PIL import Image
import json
#from data import build_vocab, get_coco_data, get_iterator



def main():
    # Set up defaults before getting command line arguments
    global device
    checkpoint_path='checkpoint_directory'
    image_path = 'flowers/test/1/image_06760.jpg'
    topk =3
    gpu = False
    category_names ='cat_to_name.json'
    device ='cpu'

    # Get command line arguments for all parameters
    print("The args are: ",sys.argv)
    len_args = len(sys.argv)
    if(len_args==0):
        error('Please provide a valid checkpoint and image path')
    checkpoint_path = sys.argv[2]
    image_path = sys.argv[1]
    
    for i in range(3,len(sys.argv)):
        if sys.argv[i] == '--top_k':
            if i == len_args - 1:
                continue
            print('Return top number of classes ',sys.argv[i+1])
            topk = int(sys.argv[i+1])
            i = i + 1
        if sys.argv[i] == '--category_names' :
            if i == len_args - 1:
                continue
            print('Category names file name ',sys.argv[i+1])
            category_names = sys.argv[i+1]
            i = i + 1
        if sys.argv[i] == '--gpu':
            if i == len_args - 1:
                continue
            print('Use gpu for training ')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gpu = True
            i = i + 1
    model = load_checkpoint(checkpoint_path)
    prob, classes = predict(image_path,model,topk)
    display_prediction(prob,classes)

    # TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    global model
    checkpoint = torch.load(filepath+'/checkpoint.pth')
    print('CHekpoint is ',checkpoint)
    model = models.vgg16(pretrained = True)
    if checkpoint['arch'] == 'densenet':
        model = models.densenet121(pretrained=True)
   
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_features'], 2048)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(2048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    #model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.in_features = checkpoint['input_features']
    model.architecture = checkpoint['arch']
    return model

    #Predict Function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = image
    # TODO: Process a PIL image for use in a PyTorch model
    im_size = im.size
    
    if im_size[0] > im_size[1]:
        im.thumbnail((im_size[0], 256))
    else:
        im.thumbnail((256, im_size[1]))
        
    #Set the box values for cropping the image
    
    #print(im.size)
    width,height = im.size
    l = (width-224)/2
    top = (height-224)/2
    r = width - (width-224)/2
    bottom = height - (height-224)/2
    
    im = im.crop((l,top,r,bottom))
    
    #print (im.size)
 
    np_image = np.array(im)
    np_image = np_image/255
    
    means = [0.485, 0.456, 0.406]
    std_devs = [0.229, 0.224, 0.225]
    
    np_image = ((np_image - means) / std_devs)
    #print(np_image.shape)
    np_image = np_image.transpose((2,0,1))
    print('The Image shape is' ,np_image.shape)
    return np_image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    global prob
    # TODO: Implement the code to predict the class from an image file
    
    model.to(device)
    model.eval()
    img = process_image(Image.open(image_path))
    img = torch.tensor(np.array([img])).type(torch.FloatTensor)
    with torch.no_grad():
        img = img.to(device)
        logits = model.forward(img)
        prob , classes = torch.topk(logits, topk)
        probs = prob.exp()
        #class_to_idx = model.class_to_idx
        #idx_to_class = {str(value):int(key) for key, value in class_to_idx.items()}
        return probs, classes

def display_prediction(probs,classes):
    probs = prob.type(torch.FloatTensor).to('cpu').numpy()
    classes = classes.type(torch.FloatTensor).to('cpu').numpy()
    classes = classes.astype(int)
    classes = classes.astype(str)
    print("Probabilities are" , probs)
    print("Classes are", classes)
    reversed_indexes = dict([[v,k] for k,v in model.class_to_idx.items()])
    print(reversed_indexes)
    list_of_flower_indexes = list()
    for index in classes[0]:
        print("The index im searching for is ",int(index))
        list_of_flower_indexes.append(reversed_indexes[int(index)])
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[i] for i in list_of_flower_indexes]
    df = pd.DataFrame(
    {'flowers': pd.Series(data=flower_names),
     'probabilities': pd.Series(data=probs[0], dtype='float64')
    })
    print(df)

main()    