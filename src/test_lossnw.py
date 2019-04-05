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
from testdataset_lossnw import *
from utils import progress_bar

import matplotlib.pyplot as plt
import matplotlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('==> Preparing data..')

criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')
rowcnn = RowCNN().to(device)

path = '/home/nevronas/Projects/Nevronas-Projects/Audio/AudioStyleTransfer/checkpoints_lossnw/'


print('==> Loading data..')
testset = InstrumentsTestDataset()

def test_instruments(epoch):
    dataloader = DataLoader(testset, batch_size=1, shuffle=True)
    dataloader = iter(dataloader)
    
    train_loss, correct, total = 0, 0, 0

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs = torch.tensor(inputs).type(torch.FloatTensor)
        inputs = inputs.to(device)

        y_pred = rowcnn(inputs)
        _, predicted = y_pred.max(1)
        
        predicted = predicted.detach().cpu().numpy()
        targets = [i.numpy()[0] for i in targets]
        total += 1

        #   print(targets)
        if predicted[0] in targets:
            correct += 1

        print("Predicted: {}, Target: {}".format(predicted[0], targets))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()

        print('Epoch: [%d], Batch: [%d/%d], Test Acc: %.3f%% (%d/%d)' % (epoch, batch_idx, len(dataloader), 100.0*correct/total, correct, total))

print('==> Testing starts..')
for i in range(1, 29):
    print("Epoch : ", i)
    rowcnn.load_state_dict(torch.load(path+'network_epoch_{}.ckpt'.format(i)))
    test_instruments(i)
   
