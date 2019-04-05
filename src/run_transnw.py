import os
import gc
import torch
import argparse
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import *
from feature import *
from loss_nw import RowCNN as LossNetwork
from vctk import VCTK
from dataset_transnw import VCCDataset
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Audio Style Transfer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=4, help='Number of epochs to train.')

# Loss network trainer
parser.add_argument('--lresume', type=int, default=1, help='resume loss from checkpoint')
parser.add_argument('--loss_lr', type=float, default=1e-4, help='The Learning Rate.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=1e-5, help='Weight decay (L2 penalty).')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, tsepoch, tstep, lsepoch, lstep = 0, 0, 0, 0, 0

loss_fn = torch.nn.MSELoss() # MaskedMSE()

print('==> Preparing data..')

# To get logs of current run only
with open("../save/transform/logs/transform_train_loss.log", "w+") as f:
    pass 

def load_audio(audio_path):
    signal, fs = librosa.load(audio_path)
    return signal, fs

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions

def inp_transform(inp):
    # inp = inp.numpy()
    inp = inp.astype(np.float32)
    inp = inp.flatten()                                                                                                                                                                                                                                 
    inp, _ = transform_stft(inp, pad = False)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    return inp

def test_transform(inp):
    inp = inp.numpy()
    inp = inp.astype(np.float32)
    inp = inp.flatten()
    inp, phase = transform_stft(inp, pad=False)
    inp = torch.Tensor(inp)                                                                                                                                 
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp, phase

print('==> Creating networks..')
t_net = Transformation()
l_net = LossNetwork()

t_net = t_net.to(device)
l_net = l_net.to(device)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

if(args.lresume):
    if(os.path.isfile('../save/network.ckpt')):
        l_net.load_state_dict(torch.load('../checkpoints_lossnw/network_epoch_29.ckpt'))
        print("=> Network : loaded")
    
    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            lsepoch, lstep = (int(i) for i in str(f.read()).split(" "))
            print("=> Network : prev epoch found")
if(args.resume):
    if(os.path.isfile('../save/transform/trans_model_new.ckpt')):
        t_net.load_state_dict(torch.load('../save/transform/trans_model_new.ckpt'))
        print('==> Transformation network : loaded')

    if(os.path.isfile("../save/transform/transform_train_loss_new.log.txt")):
        with open("../save/transform/info.txt", "r") as f:
            tsepoch, tstep = (int(i) for i in str(f.read()).split(" "))
        print("=> Transformation network : prev epoch found")

def get_style(path='../save/style/flu.wav'):
    N_FFT = 1024
    S, fs = librosa.load(path)
    S = librosa.stft(S, N_FFT)
    S, phase = librosa.magphase(S)
    S = np.log1p(S)
    S = S[:, 0:250]
    # S = np.expand_dims(S, 0)
    #matplotlib.image.imsave('../save/plots/input/style_part.png', signal)
    S = torch.from_numpy(S)
    S = S.unsqueeze(0)
    return S

def forward_loss(conten_activ, x):
    xs = []
    for conv in conten_activ:
        # print(conv)
        # print(x.shape)
        x2 = F.relu(conv(x))      
        x2 = torch.squeeze(x2, -1)
        x2 = F.max_pool1d(x2, x2.size(2)) 
        xs.append(x2)
    x = torch.cat(xs, 2) 
    # x = x.view(x.size(0), -1)
    return x

def train_transformation(epoch, alpha, beta, gamma, case_id):
    global tstep
    print('\n=> Transformation Epoch: {}'.format(epoch))
    print('=> Case id: {}'.format(case_id))
    t_net.train()
    
    vdataset = VCCDataset(load=False, transform=inp_transform)
    dataloader = DataLoader(vdataset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    train_loss = 0
    tr_con = 0
    tr_sty = 0
    tr_mse = 0

    params = t_net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    l_list = list(l_net.children())
    l_list = list(l_list[0].children())
    conten_activ = torch.nn.Sequential(*l_list[:-1]) # Not having batchnorm
    st_activ = torch.nn.Sequential(*l_list[:-1])

    for param in conten_activ.parameters():
        param.requires_grad = False
    for param in st_activ.parameters():
        param.requires_grad = False

    # alpha, beta, gamma = 10, 30, 20 # TODO : CHANGEd from 7.5, 100
    gram = GramMatrix()

    style_audio = get_style()
   
    for i in range(tstep, len(dataloader)):
        try :
            # (audios, captions) = next(dataloader)
            audios = next(dataloader)
        except ValueError:
            break
        if(type(audios) == int):
            print("=> Transformation Network : Chucked Sample")
            continue
        # print(audios.shape)

        # matplotlib.image.imsave('../save/plots/transform/audio.png', audios.cpu().numpy()[0][0])
        # break
        # audios = (audios[:, :, :, 400:650].to(device), audios[:, :, :, 650:900].to(device)  , audios[:, :, :, 900:1150].to(device), audios[:, :, :, 1150:1400].to(device))

        audios = [audios.to(device)]
        for audio in audios : # LOL - splitting coz GPU
            optimizer.zero_grad()
            y_t = t_net(audio)
            # print("y_t shape: ")
            # print(y_t.shape)
            # # y_t = y_t[:,:,:-2,:-1]
            # print("Audio shape: ")
            # print(audio.shape)
            # print("Style shape: ")
            # print(style_audio.shape)
            # print("y_t shape: ")
            # print(y_t.shape)
            content = forward_loss(conten_activ, audio)
            y_c = forward_loss(conten_activ, y_t)

            # c_loss = loss_fn(y_c, content)
            c_loss = loss_fn(y_t, audio)

            sty_aud = []
            for k in range(audio.size()[0]): # No. of style audio == batch_size
                sty_aud.append(style_audio)
            sty_aud = torch.stack(sty_aud).to(device)

            y_s = forward_loss(st_activ,y_t)
            style = forward_loss(st_activ,sty_aud)

            y_s_gram = gram(forward_loss(st_activ,y_t))
            style_gram = gram(forward_loss(st_activ,sty_aud))
		
            s_loss = loss_fn(y_s, style)

            sg_loss = loss_fn(y_s_gram, style_gram)
            
            del sty_aud	
            loss = alpha * c_loss + beta * s_loss + gamma * sg_loss 

            train_loss = loss.item()
            tr_con = c_loss.item()
            tr_sty = s_loss.item()
            tr_gram = sg_loss.item()
            
            loss.backward()
            optimizer.step()

        a1 = audios[0].cpu().numpy()
        a2 = y_t[0].detach().cpu().numpy()

        matplotlib.image.imsave('../save/plots/transform/before.png', a1[0][0])
        matplotlib.image.imsave('../save/plots/transform/after.png', a2[0])

        del audios

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(t_net.state_dict(), '../save/cases/trans_model_{}.ckpt'.format(case_id))
        with open("../save/cases/info_{}.txt".format(case_id), "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/cases/logs/transform_train_loss_{}.log".format(case_id), "a+") as lfile:
            lfile.write("{}\n".format(train_loss))

        progress_bar(i, len(dataloader), 'L: {}, CL: {}, SL: {}, GL: {} '.format(train_loss, tr_con, tr_sty, tr_gram))

    tstep = 0
    del dataloader
    del vdataset
    print('=> Transformation Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, args.epochs, train_loss))


def test(case_id, filename = '/home/nevronas/dataset/vctk/raw/p374_422.wav'):
    global t_net
    t_net.load_state_dict(torch.load('../save/cases/trans_model_{}.ckpt'.format(case_id)))
    vdataset = VCTK('/home/nevronas/dataset/', download=False)
    #dataloader = DataLoader(vdataset, batch_size=1)
    #audio, _ = next(iter(dataloader))
    audio, fs = load_audio(filename)
    # audio, fs = load_audio('../save/style/imperial.wav')
    audio = torch.Tensor(audio)
    audio, phase = test_transform(audio)
    audio = audio.to(device)
    out = t_net(audio)
    out = out[0].detach().cpu().numpy()
    audio = audio[0].cpu().numpy()
    matplotlib.image.imsave('../save/cases/plots/input/input_{}_{}.png'.format(case_id, filename[-9:-4]), audio[0])
    matplotlib.image.imsave('../save/cases/plots/output/output_{}_{}.png'.format(case_id, filename[-9:-4]), out[0])
    aud_res = invert_spectrogram(audio[0])
    out_res = invert_spectrogram(out[0])
    librosa.output.write_wav("../save/cases/plots/input/raw_input_{}_{}.wav".format(case_id, filename[-9:-4]), aud_res, fs)
    librosa.output.write_wav("../save/cases/plots/output/raw_output_{}_{}.wav".format(case_id, filename[-9:-4]), out_res, fs)
    print("Testing Finished")


'''
Testing cases
'''
'''
cases = [{'alpha' : 0, 'beta' : 10, 'gamma' : 10},
        {'alpha' : 0, 'beta' : 10, 'gamma' : 20},
        {'alpha' : 0, 'beta' : 20, 'gamma' : 10},
        {'alpha' : 10, 'beta' : 0, 'gamma' : 10},
        {'alpha' : 10, 'beta' : 0, 'gamma' : 20},
        {'alpha' : 20, 'beta' : 0, 'gamma' : 10},
        {'alpha' : 10, 'beta' : 10, 'gamma' : 0},
        {'alpha' : 20, 'beta' : 10, 'gamma' : 0},
        {'alpha' : 10, 'beta' : 20, 'gamma' : 0},
        {'alpha' : 10, 'beta' : 10, 'gamma' : 10},
        {'alpha' : 5, 'beta' : 10, 'gamma' : 10},
        {'alpha' : 5, 'beta' : 10, 'gamma' : 20},
        {'alpha' : 5, 'beta' : 20, 'gamma' : 10},
        {'alpha' : 10, 'beta' : 5, 'gamma' : 10},
        {'alpha' : 10, 'beta' : 5, 'gamma' : 20},
        {'alpha' : 20, 'beta' : 5, 'gamma' : 10},
        {'alpha' : 10, 'beta' : 10, 'gamma' : 5},
        {'alpha' : 20, 'beta' : 10, 'gamma' : 5},
        {'alpha' : 10, 'beta' : 20, 'gamma' : 5},
    ]
'''

cases = [
        {'alpha' : 10, 'beta' : 20, 'gamma' : 5}
]

case_id = 18
for case in cases:
    for epoch in range(tsepoch, tsepoch + args.epochs):

        train_transformation(epoch, case['alpha'], case['beta'], case['gamma'], case_id)
        test(case_id, '../save/style/imperial.wav')
        test(case_id)
    case_id += 1
# for epoch in range(tsepoch, tsepoch + args.epochs):
#     train_transformation(epoch)
#     test()
# t_net.load_state_dict(torch.load('../save/cases/trans_model_1.ckpt'))

