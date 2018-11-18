import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from PIL import Image
from feature import *
import librosa
from data_loader import get_loader
from PIL import Image
from torch import nn
import matplotlib
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from collections import OrderedDict

def load_audio(audio_path):
    signal, _ = librosa.load(audio_path)
    return signal

def inp_transform(inp):
    inp = inp.numpy()
    #print(inp.shape)
    inp = inp.flatten()
    #print(inp.shape)
    stft = transform_stft(inp)
    #print(stft.shape)
    stft = torch.Tensor(stft)
    #print(stft.shape)
    stft = stft.unsqueeze(0)
    stft = stft.unsqueeze(0)
    #print(stft.shape)
    return stft

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 225, 5, dilation=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.BatchNorm2d(225),
            nn.Conv2d(225, 256, 5, stride=2),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            #nn.Conv2d(256, 256, 3, stride=2),   # b, 8, 2, 2
            #nn.ReLU(True),
            #nn.BatchNorm2d(256),
            nn.Conv2d(256, 100, 3, stride=1),   # b, 8, 2, 2
            nn.ReLU(True),
            nn.BatchNorm2d(100)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x
def main(args):

     # Build models
    encoder = Encoder().eval()  # eval mode (batchnorm uses moving mean/variance)
    # decoder = Decoder()
    encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))

    #encoder_list = list(encoder.children())
    print(list(list(encoder.children())[0].children()))
    encoder_2 = nn.Sequential(*list(list(encoder.children())[0].children())[0:5]) # Might have to change later
    encoder_2 = encoder_2.to(device)
    encoder_1 = nn.Sequential(*list(list(encoder.children())[0].children())[0:3]) # Might have to change later
    encoder_1 = encoder_1.to(device)
    # Prepare an image
    '''
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    '''
    data_loader, _ = get_loader(transforms=False)
    inp, targets = next(iter(data_loader))
    print(inp.shape)
    audio = inp_transform(inp)
    audio = audio.to(device)
    print(audio.shape)
    # Generate an caption from the image
    out = encoder(audio)
    out2 = encoder_2(audio)
    out1 = encoder_1(audio)
    # final = decoder(out)
    '''
    out = decoder(feature)
    out = out[0].detach().cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    '''
    out1 = out1[0].detach().cpu().numpy()
    out2 = out2[0].detach().cpu().numpy()
    out = out[0].detach().cpu().numpy()
    # final = final[0].detach().cpu().numpy()

    print(out1.shape)
    print(out2.shape)
    print(out.shape)
    # for i in range(out.shape[0]):
    #     matplotlib.image.imsave('./fea Features3_1/vis_layer3_{}.png'.format(i), out[i])


    # for i in range(out2.shape[0]):
    #     matplotlib.image.imsave('./Features2_1/vis_layer2_{}.png'.format(i), out2[i])

    # for i in range(out1.shape[0]):
    #     matplotlib.image.imsave('./Features1_1/vis_layer1_{}.png'.format(i), out1[i])

    audio = audio[0].cpu().numpy()
    
    # matplotlib.image.imsave('./Features0_1/audio.png', audio[0])
    # matplotlib.image.imsave('./Features0_1/final_decoded.png', final[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, default='/home/nevronas/dataset/IEMOCAP/Session1/dialog/wav/Ses01F_impro01.wav', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/conv1_encoder_1.ckpt', help='path for trained encoder')
    # parser.add_argument('--decoder_path', type=str, default='./models/conv1_decoder_1.ckpt', help='path for trained decoder')
    
    args = parser.parse_args()
    main(args)

