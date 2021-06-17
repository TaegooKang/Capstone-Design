from ChestXray14 import *
from CheXNet import *
from model import *
import time
import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from Utils import *
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FineTuningSet(Dataset):
    
    def __init__(self, type):
        self.type = type
        if self.type == 'train':
            self.Data = pd.read_csv('dataset/origin_train.csv')
        elif self.type == 'val':
            self.Data = pd.read_csv('dataset/origin_val.csv')
        elif self.type == 'test':
            self.Data = pd.read_csv('dataset/origin_test.csv')
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
                            transforms.Resize((224,224)),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])  
        image = transform(image)

        row = self.Data.iloc[idx, :]
        labels = str.split(row['Finding Labels'], '|')
        if labels == ['Cardiomegaly']:
            target = [1.]
        else:
            target = [0.]
      
        return image, torch.FloatTensor(target)

    def set_normal(self):
        self.Data = self.Data[self.Data['Finding Labels']=='No Finding']
        self.full_filenames = list(self.Data['Image Index'].values)
    
    def set_abnormal(self):
        self.Data = self.Data[self.Data['Finding Labels']=='Cardiomegaly']
        self.full_filenames = list(self.Data['Image Index'].values)


class FineTuningSet_Aug(Dataset):
    def __init__(self):
        
        self.Data = pd.read_csv('dataset/aug_Cardiomegaly2.csv')
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
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])  
        image = transform(image)

        target = [1.]
    
        return image, torch.FloatTensor(target)


def main():
    model = DenseNet(14)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('models/CheXNet3_50.pth'))
    model.setmode_finetuning(1)
    model.freeze()
    model.to(device)

    dataset = FineTuningSet('train')
    dataset_aug = FineTuningSet3('train')
    dataset_aug.set_abnormal()
    dataset += dataset_aug
    train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    
    valset = FineTuningSet('val')
    validation_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False,  num_workers=4, pin_memory=True)

    optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-8)            
    #-------------------- SETTINGS: LOSS
    criterion = torch.nn.BCELoss(size_average = True)

    Loss = AverageMeter('Loss', 0)
    val_min = 100000
    print('Training Start!! (with ' + str(device) + ')')
    print('=================================================')
    
    epochs = 50
    for epoch in range(epochs):

        start_epoch = time.time()
        taken_time_for_batch = 0
        
        for batch_idx, (input, target) in enumerate(train_loader):
            
            batch_time = time.time()
            input, target = input.to(device), target.to(device)

            if True in torch.isnan(input):
                print('NAN Input, exit training')
                exit(-1)
            
            optimizer.zero_grad()
            output = model(input)

            #output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
            loss = criterion(output.float(), target.float())
            
            if torch.isnan(loss):
                print('NAN Loss, exit training')
                exit(-1)
            
            Loss.update(loss.item())
            loss.backward()
            optimizer.step()

            elapsed = time.time() - batch_time
            taken_time_for_batch += elapsed

            if batch_idx != 0 and (batch_idx + 1) % 100 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}]...({taken_time_for_batch:.2f}s) | Loss: {Loss.avg:.4f}')
                taken_time_for_batch = 0
            
            if batch_idx + 1 == len(train_loader):
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}]...({taken_time_for_batch:.2f}s) | Loss: {Loss.avg:.4f}')
        
        val_start = time.time()
        val_Loss = validate(model, device, validation_loader)   
        scheduler.step(val_Loss)
        val_elapsed = time.time() - val_start
        print(f'Validation Loss: {val_Loss:.4f}  Validation time: {val_elapsed:.2f}s')

        elapsed = time.time() - start_epoch

        #torch.cuda.empty_cache()
        print('=================================================')
        print(f'| Epoch: {epoch+1} | Loss: {Loss.avg:.4f} | Taken Time: {elapsed:.2f}s |')
        print('=================================================')

        if val_Loss <= val_min:
            val_min = val_Loss
            print('Minimum Validation Loss, Save model!')
            torch.save(model.state_dict(), 'finetuned_models/CheXNet9f_{}_min.pth'.format(epoch+1))

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 'finetuned_models/CheXNet9f_' + str(epoch+1)+'.pth')

def validate(model, device, validation_loader):

    model.eval()
    val_Loss = AverageMeter('val_Loss', 0)
    criterion = nn.BCELoss()

    for idx, (input, target) in enumerate(validation_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        val_Loss.update(loss.item())
    
    return val_Loss.avg

def computeAUROC (dataGT, dataPRED, classCount):
        
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

def test(model, device):

    model.eval()
    dataset = FineTuningSet('test')

    dataLoaderTest = DataLoader(dataset=dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True)
        
    outGT = torch.FloatTensor()
    outPRED = torch.FloatTensor()
       
    model.eval()
        
    for i, (input, target) in enumerate(dataLoaderTest):
            
        #target = target.cuda()
        outGT = torch.cat((outGT, target), 0)
            
        #bs, n_crops, c, h, w = input.size()
            
        #varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
        varInput = input.cuda()
        out = model(varInput).cpu()
        #outMean = out.view(bs, n_crops, -1).mean(1).cpu()
            
        #outPRED = torch.cat((outPRED, outMean.data), 0)
        outPRED = torch.cat((outPRED, out.data), 0)

        if i%50 == 0:
            print('({}/{})'.format(i+1,len(dataLoaderTest)))

    print(outGT.size(), outPRED.size())
    nnClassCount = 1
    aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
    #computeTHR(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()
        
    print ('AUROC mean ', aurocMean)
    
def getAcc():
    model = DenseNet(1)
    model.load_state_dict(torch.load('finetuned_models/CheXNet3_50.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataset = FineTuningSet('test')
    test_loader = DataLoader(dataset=dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True)

    ntotal = 0
    ncorrect = 0
    for idx, (input, target) in enumerate(test_loader):
        input = input.to(device)
        pred = model(input)
        pred = (pred >= 0.7).int().cpu()
        h,w = target.size()
        ntotal += h
        iscorrect = (pred==target).int().sum().item()
        ncorrect += iscorrect
    
    print(ncorrect/ntotal)


if __name__ == '__main__':
    main()
    