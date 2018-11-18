__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torchvision import transforms
from torchvision.utils import save_image
from feature import transform_stft
from feature import transform_stft_new
from vctk import VCTK
import gc
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 24
learning_rate = 1e-4


'''
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
'''


def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    images, captions = zip(*data)
    data = None
    del data
    images = torch.stack(images, 0)

    return images, captions


def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    inp = transform_stft(inp)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    return inp

dataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
dataloader = iter(dataloader)

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

encoder = Encoder().to(device)
decoder = Decoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=learning_rate, weight_decay=1e-5)

if(os.path.isfile('./models/conv1_encoder.ckpt')):
    encoder.load_state_dict(torch.load('./models/conv1_encoder.ckpt'))
    decoder.load_state_dict(torch.load('./models/conv1_decoder.ckpt'))

sepoch, step = 0, 0
if(os.path.isfile("./models/info.txt")):
    with open("./models/info.txt", "r") as f:
        sepoch, step = (int(i) for i in str(f.read()).split(" "))

for epoch in range(sepoch, num_epochs):
    for i in range(step, len(dataloader)):
        (images, captions) = next(dataloader)
        if(type(images) == int):
            print("continued")
            continue
        # Set mini-batch dataset
        del captions
        images = (images[:, :, :, 0:500].to(device), images[:, :, :, 500:1000].to(device))
        # ===================forward=====================
        latent_space = encoder(images[0])
        output = decoder(latent_space)
        #output = output.view(-1, list(images.shape))
        optimizer.zero_grad()
        loss = criterion(output, images[0][:, :, :, :-3])
        # ===================backward====================
        loss.backward()
        optimizer.step()

        #output = output.to(torch.device('cpu'))
        #images1 = images1.to(torch.device('cpu'))
        #images1 = 
        # ===================forward=====================
        latent_space = encoder(images[1])
        output = decoder(latent_space)
        #output = output.view(-1, list(images.shape))
        optimizer.zero_grad()
        loss = criterion(output, images[1][:, :, :, :-3])
        # ===================backward====================
        loss.backward()
        optimizer.step()
        del images
        print('epoch [{}/{}], step[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, i, len(dataloader), loss.item()), end="\r")

        if i%25==0:
            print('epoch [{}/{}], step[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, i, len(dataloader), loss.item()))
        
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(encoder.state_dict(), './models/conv1_encoder.ckpt')
        torch.save(decoder.state_dict(), './models/conv1_decoder.ckpt')

        with open("models/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

    step = 0
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

    #if epoch % 1 == 0:
        #pic = to_img(output.cpu().data)
        #save_image(pic, './dc_img/image_{}.png'.format(epoch))

