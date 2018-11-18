import os
import torch
import argparse
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from model import *
from utils import progress_bar
from feature import *
from vctk import VCTK
from torch import nn

parser = argparse.ArgumentParser(description='PyTorch Audio Style Transfer')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--resume', '-r', action='store_false', help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')

# Loss network trainer
'''
parser.add_argument('--loss_lr', type=float, default=1e-3, help='The Learning Rate.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=5e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=10)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
'''

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 225, 5, dilation=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(225),
            nn.Conv2d(225, 256, 5, stride=2), 
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 100, 3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(100)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x

print('==> Preparing data..')

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    return images, captions


def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    stft = transform_stft(inp)
    stft = torch.Tensor(stft)
    stft = stft.unsqueeze(0)
    return stft


dataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)

print('==> Building transformation network..')
t_net = TransformationNetwork()
t_net = t_net.to(device)

print('==> Loading Autoencoder..')

encoder = Encoder()
encoder = encoder.to(device)
encoder.load_state_dict(torch.load('/home/nevronas/Projects/Nevronas-Projects/Audio/AudioStyleTransfer/misc/Autoencoder/models/conv_encoder_1.ckpt'))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../save/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../save/checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

print('==> Defining Loss and Optimizer..')

criterion = torch.nn.MSELoss(size_average=False)
optimizer = optim.LBFGS(t_net.parameters(), lr=args.lr) 

def train_transformation(epoch):
    print('\nTransformation Epoch: {}'.format(epoch))
    '''
    t_net.train()
    
    train_loss, correct, total = 0, 0, 0
    params = t_net.parameters()
    optimizer = optim.LBFGS(params, lr=args.lr) # TODO : modify hyperparams
    '''
    encoder_2 = nn.Sequential(*list(list(encoder.children())[0].children())[0:5]) 
    encoder_2 = encoder_2.to(device)
    encoder_1 = nn.Sequential(*list(list(encoder.children())[0].children())[0:3])
    encoder_1 = encoder_1.to(device)

    '''
    l_list = *list(l_net.children())
    conten_activ = torch.nn.Sequential(l_list[:-2]) # Might have to change later
    '''
    for param in encoder_2.parameters():
        param.requires_grad = False

    for param in encoder_1.parameters():
        param.requires_grad = False

    for param in encoder.parameters():
        param.requires_grad = False

    alpha, beta = 0.1, 0.1 # TODO : CHANGE hyperparams
    gram = GramMatrix()
    style_image = None  # load style audio here

    loss_ep = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if(type(inputs) == int):
            continue

        inputs = inputs.to(device)
        optimizer.zero_grad()

        y_t = t_net(inputs)

        '''
        Now here decide whether which layer of the autoencoder gives the content
        Currently taking third layer
        '''

        content = encoder(inputs)
        y_c = encoder(y_t)
        c_loss = criterion(y_c, content)

        '''
        for st_i in range(2, len(l_list)): # NOTE : Change later
            st_activ = torch.nn.Sequential(l_list[:-i])
            for param in st_activ.parameters():
                param.requires_grad = False

            y_s = gram(st_activ(y_t))
            style = gram(st_activ(style_image))

            s_loss += loss_fn(style, y_s)
        '''

        '''
        Decide which layers to take for style
        Here considering layers 1, 2, and 3
        '''

        inp_acts1 = encoder_1(style_audio)
        inp_acts2 = encoder_2(style_audio)
        inp_acts3 = encoder(style_audio)

        y_acts1 = encoder_1(y_t)
        y_acts2 = encoder_2(y_t)
        y_acts3 = encoder(y_t)

        s_loss = criterion(gram(y_acts1), gram(inp_acts1)) + criterion(gram(y_acts2), gram(inp_acts2)) + criterion(gram(y_acts3), gram(inp_acts3))

        loss = alpha * c_loss + beta * s_loss

        '''
        for param in l_net.parameters():
            param.requires_grad = False
        '''

        loss.backward()
        optimizer.step()

        print('epoch [{}/{}], step[{}/{}], loss:{:.4f}'.format(epoch+1, args.epochs, i, len(dataloader), loss.item()), end="\r")
        loss_ep = loss.item()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.epochs, loss_ep))
        '''
        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/logs/transform_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/logs/transform_train_acc", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''

def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            _, outputs = t_net(inputs, old_class=False)
            
            ''' TODO 
                Decide how to calculate accuracy for style transfer
            loss = loss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            with open("./logs/test_loss_{}.log".format(curr_class), "a+") as lfile:
                lfile.write(str(test_loss / total))
                lfile.write("\n")

            with open("./logs/test_acc_{}.log".format(curr_class), "a+") as afile:
                afile.write(str(correct / total))
                afile.write("\n")

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            ''' 

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('../save/checkpoint'):
            os.mkdir('../save/checkpoint')
        torch.save(state, '../save/checkpoint/ckpt.t7')
        best_acc = acc
'''
for epoch in range(start_epoch, start_epoch + 200):
    train_lossn(epoch)
'''

torch.save(l_net, "../save/loss_model/vgg19.sav")

start_epoch = 0
for epoch in range(start_epoch, start_epoch + args.epochs):
    train_transformation(epoch)
    # test(epoch)
    torch.save(t_net.state_dict(), './save/checkpoint/transform_{}.ckpt'.format(epoch))
