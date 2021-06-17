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


def main():
    
    #---- Path to the directory with images
    pathDirData = './database'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transCrop = 224    
    transformList = []
    transformList.append(transforms.RandomResizedCrop(transCrop))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence=transforms.Compose(transformList)

    transformList2 = []
    transformList2.append(transforms.RandomResizedCrop(transCrop))
    transformList2.append(transforms.RandomHorizontalFlip())
    transformList2.append(transforms.RandomAffine(
            degrees=30, translate=(0.2, 0.2),
            scale=(0.8, 1.2), shear=15))
    transformList2.append(transforms.ToTensor())
    transformList2.append(normalize) 
    transformSequence2=transforms.Compose(transformList2)

    #-------------------- SETTINGS: DATASET BUILDERS
    datasetTrain = ChestXray14(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
    datasetTrain2 = ChestXray14(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence2)
    datasetTrain2.set('Cardiomegaly')
    datasetTrain += datasetTrain2
    datasetVal = ChestXray14(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)

    trBatchSize = 16
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=2, pin_memory=True)
    dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)
    print(len(datasetTrain))
    #model = CheXNet(14, True)
    model = DenseNet(14)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train(model, device, dataLoaderTrain, dataLoaderVal, 50)
    
def train(model, device, train_loader, validation_loader, epochs):

    optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-8)            
    #-------------------- SETTINGS: LOSS
    criterion = torch.nn.BCELoss(size_average = True)

    Loss = AverageMeter('Loss', 0)
    val_min = 100000
    print('Training Start!! (with ' + str(device) + ')')
    print('=================================================')
    
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
            torch.save(model.state_dict(), 'models/CheXNet7_{}_min.pth'.format(epoch+1,val_min))

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 'models/CheXNet7_' + str(epoch+1)+'.pth')
        
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
    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transCrop = 224
    #-------------------- SETTINGS: DATASET BUILDERS
    transformList = []
    '''
    transformList.append(transforms.Resize(224))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    '''
    transformList.append(transforms.TenCrop(transCrop))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence=transforms.Compose(transformList)
        
    datasetTest = ChestXray14(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=2, num_workers=2, shuffle=False, pin_memory=True)
        
    outGT = torch.FloatTensor()
    outPRED = torch.FloatTensor()
       
    model.eval()
        
    for i, (input, target) in enumerate(dataLoaderTest):
            
        #target = target.cuda()
        outGT = torch.cat((outGT, target), 0)
            
        bs, n_crops, c, h, w = input.size()
            
        varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
        #varInput = input.cuda()
        out = model(varInput).cpu()
        outMean = out.view(bs, n_crops, -1).mean(1).cpu()
            
        outPRED = torch.cat((outPRED, outMean.data), 0)
        #outPRED = torch.cat((outPRED, out), 0)

        if i%50 == 0:
            print('({}/{})'.format(i+1,len(dataLoaderTest)))

    nnClassCount = 14
    aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()
        
    print ('AUROC mean ', aurocMean)
    
    CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    for i in range (0, len(aurocIndividual)):
        print (CLASS_NAMES[i], ' ', aurocIndividual[i])

if __name__ == "__main__":

    main()
    '''
    model = DenseNet(14)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('models/CheXNet7_50.pth'))
    model.to(device)
    test(model, device)
    '''
