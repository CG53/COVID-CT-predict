import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.read_dataset import CovidCTDataset
from models.squeezenet import squeezenet
from models.mobilenetv2 import MobileNetV2
from models.shufflenetv2 import ShuffleNetV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


batchsize = 10
total_epoch = 100
device = 'cuda'
#modelname = 'squeezenet'
#modelname = 'MobileNetV2'
modelname = 'shufflenetv2'

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    trainset = CovidCTDataset(root_dir='./COVID-CT-Image/',
                              txt_COVID='./COVID-CT-Image/Dataset-Split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='./COVID-CT-Image/Dataset-Split/NonCOVID/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir='./COVID-CT-Image/',
                            txt_COVID='./COVID-CT-Image/Dataset-Split/COVID/testCT_COVID.txt',
                            txt_NonCOVID='./COVID-CT-Image/Dataset-Split/NonCOVID/testCT_NonCOVID.txt',
                            transform=test_transformer)
    testset = CovidCTDataset(root_dir='./COVID-CT-Image/',
                             txt_COVID='./COVID-CT-Image/Dataset-Split/COVID/valCT_COVID.txt',
                             txt_NonCOVID='./COVID-CT-Image/Dataset-Split/NonCOVID/valCT_NonCOVID.txt',
                             transform=test_transformer)
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)   
    
    
    #model = squeezenet(class_num=2).cuda()
    #model = MobileNetV2(class_num=2).cuda()
    model = ShuffleNetV2(class_num=2).cuda()
    if os.path.exists('./models_pretrained/' + modelname + '.pth'):
        model.load_state_dict(torch.load('./models_pretrained/{}.pth'.format(modelname)))
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    
    
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_acc = []
    val_epoch_acc = []
    for epoch in range(1, total_epoch + 1):
        ''' model train '''
        model.train()
        train_loss = 0
        train_correct = 0
        tbar = tqdm(train_loader)
        print('\n epoch %d:' % (epoch))
        for i, batch_samples in enumerate(tbar):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)                
            output = model(data)        
            
            loss = criteria(output, target.long())
            
            train_loss += loss.item()
            tbar.set_description('train loss: %.3f' % (train_loss / (i + 1)))
            
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
        print('train average loss: %.4f' % (train_loss / len(train_loader)))
        print('train average acc: %.2f' % (100.0 * train_correct / len(train_loader.dataset)))
        train_epoch_acc.append(100.0 * train_correct / len(train_loader.dataset))
        train_epoch_loss.append(train_loss / len(train_loader))
        
        
        ''' model val '''
        with torch.no_grad():
            val_best_loss = 999999
            val_loss = 0
            val_correct = 0
            tbar = tqdm(val_loader)
            for i, batch_samples in enumerate(tbar):
                data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
                output = model(data)
                
                val_loss += criteria(output, target.long()).item()
                tbar.set_description('val loss: %.3f' % (val_loss / (i + 1)))
                
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.long().view_as(pred)).sum().item()
                
            print('val average loss: %.4f' % (val_loss / len(val_loader)))
            print('val average acc: %.2f' % (100.0 * val_correct / len(val_loader.dataset)))
            val_epoch_acc.append(100.0 * val_correct / len(val_loader.dataset))
            val_epoch_loss.append(val_loss / len(val_loader))
            
            if val_loss < val_best_loss:
                val_best_loss = val_loss
                torch.save(model.state_dict(), "./models_pretrained/{}.pth".format(modelname))
        
    x_epoch = np.arange(1, total_epoch + 1)
    plt.plot(x_epoch, train_epoch_loss, 'r', x_epoch, val_epoch_loss, 'g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(modelname)
    plt.legend(['train', 'val'])
    plt.show()
    plt.plot(x_epoch, train_epoch_acc, 'r', x_epoch, val_epoch_acc, 'g')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(modelname)
    plt.legend(['train', 'val'])
    plt.show()
    
        
    ''' model test '''
    test_loss = 0
    test_correct = 0
    predlist = []
    targetlist = []
    scorelist = []
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for i, batch_samples in enumerate(tbar):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)
            
            test_loss += criteria(output, target.long()).item()
            tbar.set_description('test loss: %.3f' % (test_loss / (i + 1)))
            
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.long().view_as(pred)).sum().item()
            
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            predlist = np.append(predlist, pred.cpu().numpy())
            targetlist = np.append(targetlist, target.long().cpu().numpy())
    print('test average loss: %.4f' % (test_loss / len(test_loader)))
    
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()
    
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    AUC = roc_auc_score(targetlist, scorelist)
    print('test dataset: recall:{:.4f}, precision:{:.4f}, F1:{:.4f}, accuracy:{:.4f}, AUC:{:.4f}'.format(r, p, F1, acc, AUC))
        


    
    

        
        
        
    
    
    
        
        
        
        
        
        
        
