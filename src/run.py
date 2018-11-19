import os
import gc
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_audio

from model import *
from feature import *
from vctk import VCTK
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Audio Style Transfer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')

# Loss network trainer
parser.add_argument('--lresume', action='store_true', help='resume loss from checkpoint')
parser.add_argument('--loss_lr', type=float, default=1e-4, help='The Learning Rate.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=1e-5, help='Weight decay (L2 penalty).')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, tsepoch, tstep, lsepoch, lstep = 0, 0, 0, 0, 0

loss_fn = torch.nn.MSELoss(size_average=False)

print('==> Preparing data..')

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    inp = transform_stft(inp)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    return inp

vdataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
dataloader = iter(dataloader)

print('==> Creating networks..')
t_net = TransformationNetwork()
t_net = t_net.to(device)
encoder = Encoder().to(device)
decoder = Decoder().to(device)

if(args.lresume):
    if(os.path.isfile('../save/loss/loss_encoder.ckpt')):
        encoder.load_state_dict(torch.load('../save/loss/loss_encoder.ckpt'))
        decoder.load_state_dict(torch.load('../save/loss/loss_decoder.ckpt'))
        print("=> Loss Network : loaded")
    
    if(os.path.isfile("../save/loss/info.txt")):
        with open("../save/loss/info.txt", "r") as f:
            lsepoch, lstep = (int(i) for i in str(f.read()).split(" "))
            print("=> Loss Network : prev epoch found")

if(args.resume):
    if(os.path.isfile('../save/transform/trans_model.ckpt')):
        t_net.load_state_dict(torch.load('../save/transform/trans_model.ckpt'))
        print('==> Transformation network : loaded')

    if(os.path.isfile("../save/transform/info.txt")):
        with open("../save/transform/info.txt", "r") as f:
            sepoch, lstep = (int(i) for i in str(f.read()).split(" "))
        print("=> Transformation network : prev epoch found")

def train_lossn(epoch):
    print('\n=> Loss Epoch: {}'.format(epoch))
    train_loss, total = 0, 0
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.loss_lr, weight_decay=args.decay)
    
    for i in range(lstep, len(dataloader)):
        (audios, captions) = next(dataloader)
        if(type(audios) == int):
            print("=> Loss Network : Chucked Sample")
            continue
        
        del captions
        audios = (audios[:, :, :, 0:500].to(device), audios[:, :, :, 500:1000].to(device))

        for audio in audios:
            latent_space = encoder(audio)
            output = decoder(latent_space)
            optimizer.zero_grad()
            loss = criterion(output, audio[:, :, :, :-3])
            loss.backward()
            optimizer.step()

        del audios
        train_loss += loss.item()

        with open("../save/loss/logs/lossn_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / (i - lstep +1)))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(encoder.state_dict(), '../save/loss/loss_encoder.ckpt')
        torch.save(decoder.state_dict(), '../save/loss/loss_decoder.ckpt')

        with open("models/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - lstep + 1)))

    lstep = 0
    print('=> Loss Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(data_loader)))

def train_transformation(epoch):
    print('\n=> Transformation Epoch: {}'.format(epoch))
    t_net.train()
    
    train_loss = 0
    params = t_net.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    l_list = *list(encoder.children())
    conten_activ = torch.nn.Sequential(l_list[:-1]) # Not having batchnorm 
    
    for param in conten_activ.parameters():
        param.requires_grad = False

    alpha, beta = 0.7, 0.3 # TODO : CHANGE hyperparams
    gram = GramMatrix()
    style_audio = None # TODO : get style audio

    for i in range(tstep, len(dataloader)):
        (audios, captions) = next(dataloader)
        if(type(audios) == int):
            print("=> Transformation Network : Chucked Sample")
            continue

        audios = (audios[:, :, :, 0:500].to(device), audios[:, :, :, 500:1000].to(device))
        for audio in audios : # LOL - splitting coz GPU
            optimizer.zero_grad()

            y_t = t_net(audio)

            contnet = conten_activ(audio)
            y_c = conten_activ(y_t)
            c_loss = loss_fn(y_c, content)

            s_loss = 0
            for st_i in range(5, len(l_list), 3): # NOTE : gets relu of 1, 2, 3
                st_activ = torch.nn.Sequential(l_list[:-i])
                for param in st_activ.parameters():
                    param.requires_grad = False

                y_s = gram(st_activ(y_t))
                style = gram(st_activ(style_audio))

                s_loss += loss_fn(style, y_s)

            loss = alpha * c_loss + beta * s_loss

            for param in encoder.parameters():
                param.requires_grad = False
        
            loss.backward()
            optimizer.step()

        del audios
        train_loss += loss.item()

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(t_net.state_dict(), '../save/transform/trans_model.ckpt')
        with open("../save/transform/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/transform/logs/transform_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        progress_bar(i, len(dataloader), 'Loss: %.3f ' % (train_loss / (i - tstep + 1), ))

    tstep = 0
    print('=> Transformation Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, args.epochs, train_loss / len(data_loader)))


'''
TODO : TEST
def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (audios, targets) in enumerate(testloader):
            _, outputs = t_net(audios, old_class=False)
            
             TODO 
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

#for epoch in range(lsepoch, lsepoch + 5):
#    train_lossn(epoch)

for epoch in range(tsepoch, tsepoch + args.epochs):
    train_transformation(epoch)
    #test(epoch)

