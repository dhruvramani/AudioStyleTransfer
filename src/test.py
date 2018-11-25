import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from PIL import Image
from feature import *
import librosa
from torch import nn
import matplotlib
from models import *
from vctk import VCTK
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio(audio_path):
    signal, _ = librosa.load(audio_path)
    return signal

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.astype(np.float32)
    inp = inp.flatten()
    inp = transform_stft(inp, pad=False)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp

def main():

    trans_net = Transformation()
    trans_net = trans_net.to(device)
    trans_net.load_state_dict(torch.load('../save/transform/trans_model.ckpt'))
    
    vdataset = VCTK('/home/nevronas/dataset/', download=False)
    dataloader = DataLoader(vdataset, batch_size=1)

    audio, targets = next(iter(dataloader))
    audio = inp_transform(audio)
    audio = audio.to(device)
    out = trans_net(audio)
    out = out[0].detach().cpu().numpy()
    audio = audio[0].cpu().numpy()
    matplotlib.image.imsave('../save/plots/input/audio.png', audio[0])
    matplotlib.image.imsave('../save/plots/output/stylized_audio.png', out[0])

    #matplotlib.image.imsave('out.png', out[0])

    # Print out the image and the generated caption
    
    '''
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    '''
    
if __name__ == '__main__':
    main()
