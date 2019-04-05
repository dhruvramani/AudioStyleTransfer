import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from loss_nw import *
from dataset_lossnw import *
from utils import progress_bar

import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser(description='PyTorch Instrument Classification')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=16, type=int) 
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--preparedata', type=int, default=1)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('==> Preparing data..')

criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')
rowcnn = RowCNN().to(device)

print('==> Loading data..')
trainset = InstrumentsDataset()
testset = InstrumentsDataset(test=True)

def train_instruments(currepoch, epoch):
    dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Instrument Epoch: %d' % currepoch)
    
    train_loss, correct, total = 0, 0, 0
    params = rowcnn.parameters()
    optimizer = optim.Adam(params, lr=args.lr)

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = torch.tensor(inputs).type(torch.FloatTensor), torch.tensor(targets).type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        y_pred = rowcnn(inputs)

        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../logs_lossnw/instrument_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../logs_lossnw/instrument_train_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(rowcnn.state_dict(), '../weights_lossnw/network_train.ckpt')
        with open("../information_lossnw/instrument_info.txt", "w+") as f:
            f.write("{} {}".format(currepoch, batch_idx))
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f , Acc: %.3f%% (%d/%d)' % (batch_idx, len(dataloader), loss.item(), train_loss/(batch_idx+1), 100.0*correct/total, correct, total), end='\r')

    torch.save(rowcnn.state_dict(), '../checkpoints_lossnw/network_train_epoch_{}.ckpt'.format(currepoch + 1))
    print('=> Classifier Network : Epoch [{}/{}], Loss:{:.4f}'.format(currepoch+1, epoch, train_loss / len(dataloader)))

def test_instruments(currepoch, epoch):
    dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Instrument Testing Epoch: %d' % currepoch)
    
    test_loss, correct, total = 0, 0, 0

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = torch.tensor(inputs).type(torch.FloatTensor), torch.tensor(targets).type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        y_pred = rowcnn(inputs)

        loss = criterion(y_pred, targets)

        test_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../logs_lossnw/instrument_test_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(test_loss / total))

        with open("../logs_lossnw/instrument_test_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f , Acc: %.3f%% (%d/%d)' % (batch_idx, len(dataloader), loss.item(), test_loss/(batch_idx+1), 100.0*correct/total, correct, total), end='\r')

    print('=> Classifier Network Test: Epoch [{}/{}], Loss:{:.4f}'.format(currepoch+1, epoch, test_loss / len(dataloader)))

print('==> Training starts..')
for epoch in range(args.epochs):
    train_instruments(epoch, args.epochs)
    test_instruments(epoch, args.epochs)
   
