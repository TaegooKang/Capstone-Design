import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision


class DenseNet(nn.Module):
    def __init__(self, nclass):
        super(DenseNet, self).__init__()
        
        # get the pretrained DenseNet121 network
        self.densenet = torchvision.models.densenet121(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        
        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # get the classifier of the densenet121
        kernelCount = self.densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, nclass), nn.Sigmoid())
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((-1, 1024))
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)


    # freeze network's layers except classifier
    def freeze(self):
        for params in self.features_conv.parameters():
            params.required_grad = False

    def setmode_finetuning(self, nclass):
        self.classifier = nn.Sequential(nn.Linear(1024, nclass), nn.Sigmoid())


if __name__ == "__main__":
    model = DenseNet(14).cuda()
    model.freeze()
    

