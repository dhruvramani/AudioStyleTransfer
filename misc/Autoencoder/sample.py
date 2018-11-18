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
            nn.Conv2d(256, 256, 3, stride=2),   # b, 8, 2, 2
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 100, 3, stride=1),   # b, 8, 2, 2
            nn.ReLU(True),
            nn.BatchNorm2d(100)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 5, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 11, stride=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, 1, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def load_audio(audio_path):
    signal, _ = librosa.load(audio_path)
    return signal

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    inp = transform_stft(inp)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp


def main(args):

    # Build models
    encoder = Encoder().eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Decoder()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    '''
    Samarjit's shit
    encoder_list = list(encoder.children())
    
    print(list(list(encoder.children())[0].children()))
    encoder_2 = nn.Sequential(*list(list(encoder.children())[0].children())[0:5]) # Might have to change later
    encoder_2 = encoder_2.to(device)
    encoder_1 = nn.Sequential(*list(list(encoder.children())[0].children())[0:3]) # Might have to change later
    encoder_1 = encoder_1.to(device
    '''

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
    
    # Generate an caption from the image
    out = encoder(audio)
    
    #out2 = encoder_2(audio)
    #out1 = encoder_1(audio)
    
    final = decoder(out)
    '''
    out = decoder(feature)
    out = out[0].detach().cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    '''
    #out1 = out1[0].detach().cpu().numpy()
    #out2 = out2[0].detach().cpu().numpy()
    out = out[0].detach().cpu().numpy()
    final = final[0].detach().cpu().numpy()

    #print(out1.shape)
    #print(out2.shape)
    #print(out.shape)
    #for i in range(out.shape[0]):
        #matplotlib.image.imsave('./fea Features3_1/vis_layer3_{}.png'.format(i), out[i])


    #for i in range(out2.shape[0]):
        #matplotlib.image.imsave('./Features2_1/vis_layer2_{}.png'.format(i), out2[i])

    #for i in range(out1.shape[0]):
        #matplotlib.image.imsave('./Features1_1/vis_layer1_{}.png'.format(i), out1[i])

    audio = audio[0].cpu().numpy()
    
    matplotlib.image.imsave('./Features/audio.png', audio[0])
    matplotlib.image.imsave('./Features/final_decoded.png', final[0])

    #matplotlib.image.imsave('out.png', out[0])

    # Print out the image and the generated caption
    
    '''
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    '''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, default='/home/nevronas/dataset/IEMOCAP/Session1/dialog/wav/Ses01F_impro01.wav', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/conv1_encoder.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/conv1_decoder.ckpt', help='path for trained decoder')
    
    args = parser.parse_args()
    main(args)
