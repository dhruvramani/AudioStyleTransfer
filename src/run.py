import os
import torch
import argparse
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from resnet import resnet34

from model import *
from models.conv_autoencoder import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Audio Style Transfer')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')

# Loss network trainer
parser.add_argument('--loss_lr', type=float, default=0.1, help='The Learning Rate.')
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


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
'''
TODO : @Anirban, see what to do for preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''

# vctk_set = torchaudio.datasets.VCTK('../data/', download=True) # NOTE @Anirban - see this!
trainloader = torch.utils.data.DataLoader(vctk_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
# testloader 

print('==> Building transformation network..')
t_net = TransformationNetwork()
t_net = t_net.to(device)
l_net = resnet34() # Or try vgg19() / vgg19_bn()
l_net = l_net.to(device)

if device == 'cuda':
    t_net = torch.nn.DataParallel(t_net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../save/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../save/checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

ce_loss = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss(size_average=False)

def train_lossn(epoch):
    print('Loss Epoch: {}'.format(epoch))
    l_net.train()
    train_loss, correct, total = 0, 0, 0
    params = l_net.parameters()
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.decay, nesterov=True) # TODO : modify hyperparams
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        y_pred = l_net(inputs)
        loss = ce_loss(y_pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/logs/lossn_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/logs/lossn_train_acc", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def train_transformation(epoch):
    print('\nTransformation Epoch: {}'.format(epoch))
    t_net.train()
    
    l_net = torch.load("../save/loss_model/vgg19.sav")
    train_loss, correct, total = 0, 0, 0
    params = t_net.parameters()
    optimizer = optim.LBFGS(params, lr=args.lr) # TODO : modify hyperparams

    l_list = *list(l_net.children())
    conten_activ = torch.nn.Sequential(l_list[:-2]) # Might have to change later
    
    for param in conten_activ.parameters():
        param.requires_grad = False

    alpha, beta = 0.1, 0.1 # TODO : CHANGE hyperparams
    gram = GramMatrix()
    style_image = None

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        y_t = t_net(inputs)

        contnet = conten_activ(inputs)
        y_c = conten_activ(y_t)
        c_loss = loss_fn(y_c, content)

        s_loss = 0
        for st_i in range(2, len(l_list)): # NOTE : Change later
            st_activ = torch.nn.Sequential(l_list[:-i])
            for param in st_activ.parameters():
                param.requires_grad = False

            y_s = gram(st_activ(y_t))
            style = gram(st_activ(style_image))

            s_loss += loss_fn(style, y_s)

        loss = alpha * c_loss + beta * s_loss

        for param in l_net.parameters():
            param.requires_grad = False
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/logs/transform_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/logs/transform_train_acc", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


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

for epoch in range(start_epoch, start_epoch + 200):
    train_lossn(epoch)
    
torch.save(l_net, "../save/loss_model/vgg19.sav")

start_epoch = 0
for epoch in range(start_epoch, start_epoch + args.epochs):
    train_transformation(epoch)
    test(epoch)

