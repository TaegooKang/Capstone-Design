import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ChestXray14 (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')

        #imageData = Image.open(imagePath)
        #copyData = imageData
        #imageData = imageData.convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
    #-------------------------------------------------------------------------------- 
    
    def set(self, lesion):

        pathDatasetFile = 'dataset/train_1.txt'
        pathImageDirectory = 'database'

        fileDescriptor = open(pathDatasetFile, "r")

        Labels = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        idx = Labels.index(lesion)

        self.listImagePaths = []
        self.listImageLabels = []
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                if imageLabel[idx] == 1:
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    def set_abnormal(self):

        pathDatasetFile = 'dataset/train_1.txt'
        pathImageDirectory = 'database'

        fileDescriptor = open(pathDatasetFile, "r")

        self.listImagePaths = []
        self.listImageLabels = []
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                if  1 in imageLabel:
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()


class ChestXray14_Aug(Dataset):
    def __init__(self):
        
        self.Data = pd.read_csv('dataset/aug_Cardiomegaly.csv')
        self.Copy = self.Data
        self.full_filenames = list(self.Data['Image Index'].values)
        

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getpath__(self, idx):
        # return the path of image
        path = 'database'
        return os.path.join(path, self.full_filenames[idx])

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.__getpath__(idx)).convert('RGB')   
        image.load()
        transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])  
        image = transform(image)

        Labels = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        row = self.Data.iloc[idx, :]
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(Labels))
        if labels != ['No Finding']:
            for lab in labels:
                lab_idx = Labels.index(lab)
                target[lab_idx] = 1
      
        return image, torch.FloatTensor(target)



class ChestXray14_Normal (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                if not 1 in imageLabel:
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        #imageData = Image.open(imagePath).convert('RGB')

        imageData = Image.open(imagePath)
        
        #imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        
        
        return imageData
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
    #-------------------------------------------------------------------------------- 
    
