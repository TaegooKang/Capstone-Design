from model import *
import os
import numpy as np
import time
import sys
from PIL import Image

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ChestXray14 import *
from CheXNet import *
from sklearn.metrics import roc_auc_score, roc_curve

def computeAUROC (dataGT, dataPRED, classCount):
        
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

def computeTHR(dataGT, dataPRED, nclass):

    Label = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    tar, pred = dataGT.numpy(), dataPRED.numpy()

    Thresholds = []
    Performances = []
    Thr_dic = {}

    for i in range (nclass):
        fpr, tpr, thresholds = roc_curve(tar[:,i], pred[:,i])
        fpr, tpr = torch.Tensor(fpr), torch.Tensor(tpr)
        performance = tpr + (1-fpr)
        maxidx = torch.argmax(performance).item()
        thr = thresholds[maxidx]
        per = performance[maxidx]
        Thresholds.append(thr)
        Performances.append(per)
        Thr_dic[Label[i]] = thr
    
    Thr_dic = pd.DataFrame(Thr_dic, index=[0])
    Thr_dic.to_csv('models/CheXNet3_50_thr.csv', index=False)

def test(model, device):

    model.eval()
    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transCrop = 224
    #-------------------- SETTINGS: DATASET BUILDERS
    transformList = []
    #'''
    transformList.append(transforms.Resize(224))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    '''
    transformList.append(transforms.TenCrop(transCrop))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    '''
    transformSequence=transforms.Compose(transformList)
    
    datasetTest = ChestXray14(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=16, num_workers=2, shuffle=False, pin_memory=True)
        
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

    nnClassCount = 14
    aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
    #computeTHR(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()
        
    print ('AUROC mean ', aurocMean)
    
    CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    for i in range (0, len(aurocIndividual)):
        print (CLASS_NAMES[i], ' ', aurocIndividual[i])

if __name__ == '__main__':
    model = DenseNet(14)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('models/CheXNet4_40.pth'))
    model.to(device)
    test(model, device)