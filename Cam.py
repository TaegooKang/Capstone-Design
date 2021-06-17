from model import *
import os
import numpy as np
import time
import sys
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ChestXray14 import *
from CheXNet import *
import random
import pandas as pd
from FineTuning import *

# load model:CheXNet
model = DenseNet(1)
model.load_state_dict(torch.load('finetuned_models/CheXNet4_35_min.pth'))
model.cuda()
model.eval()

# define transform
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize(224))
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transform = transforms.Compose(transformList)

# Lesion index
Labels = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# define dataloader
'''
pathDirData = './database'
pathFileTest = './dataset/train_1.txt'
dataset = ChestXray14(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transform)
normal_dataset = ChestXray14_Normal(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transform)
print(len(normal_dataset))
lesion = Labels[1]
dataset.set(lesion)
'''

dataset = FineTuningSet('train')
dataset.set_abnormal()

normal_dataset = FineTuningSet2('train')
normal_dataset.set_normal()

dataLoaderTest = DataLoader(dataset=dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True)

print(len(dataset))

image_index = []
finding_labels = []

for idx in tqdm(range(len(dataset))):
    
    nrand = random.randint(0,1127)
    base = normal_dataset[nrand]
    base = transforms.Grayscale()(base)
    base = transforms.Resize((224,224))(base)
    base = np.asarray(base)

    img, copyimg, label = dataset[idx]
    img = img.cuda().unsqueeze_(0)
    pred = model(img)
    
    if pred[:,0] >= 0.5:
        pred[:,0].backward()
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(1024):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        #
        #img = transforms.Grayscale()(img)
        copy = transforms.Grayscale()(copyimg)
        copy = transforms.Resize((224,224))(copy)
        copy = np.asarray(copy)

        # draw the heatmap
        heatmap = np.asarray(heatmap)

        heatmap = cv2.resize(heatmap, (224, 224))

        sample = copy * heatmap
        
        back = 1 - heatmap
        back *= base
        sample += back
        
        if  not (True in np.isnan(sample)):
            cv2.imwrite('database/aug_images_f/img{}.png'.format(idx), sample)
            image_index.append('aug_images_f/img{}.png'.format(idx))
            finding_labels.append('Cardiomegaly')

csv = {}
csv['Image Index'] = image_index
csv['Finding Labels'] = finding_labels
csv = pd.DataFrame(csv)
csv.to_csv('dataset/aug_Cardiomegaly2.csv')      

