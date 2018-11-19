import torch
import torch.nn.functional as F
from torch.autograd import Variable

class conv_bn(torch.nn.Module):
    def __init__(self, inp_ch, outp_ch):
        super(conv_bn, self).__init__()
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inp_ch, outp_ch, 3),
                torch.nn.BatchNorm2d(outp_ch),
                torch.nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class encode(torch.nn.Module):
    def __init__(self, inp_ch, outp_ch):
        super(encode, self).__init__()
        self.mpconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            conv_bn(inp_ch, outp_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

# Unet based, can change
class decode(torch.nn.Module):
    def __init__(self, inp_ch, outp_ch, bilinear=False):
        super(decode, self).__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose2d(inp_ch//2, inp_ch//2, 2, stride=2)
        self.conv = conv_bn(inp_ch, outp_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x = x1
        if(x2):
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class GramMatrix(nn.Module):
    def forward(self, y):
        # TODO - modify this
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class TransformationNetwork(torch.nn.Module):
    def __init__(self, n_channels=1):
        super(TransformationNetwork, self).__init__()
        self.inc = conv_bn(n_channels, 64)
        self.e1 = encode(64, 128)
        self.e2 = encode(128, 256)
        self.e3 = encode(256, 512)
        self.e4 = encode(512, 512)
        self.d1 = decode(1024, 256)
        self.d2 = decode(512, 128)
        self.d3 = decode(256, 64)
        self.d4 = decode(128, 64)
        self.outc = torch.nn.Conv2d(64, n_channels, 1)

    def forward(self, x):
        h1 = self.inc(x)
        h2 = self.e1(h1)
        h3 = self.e2(h2)
        h4 = self.e3(h3)
        h5 = self.e4(h4)
        h = self.d1(h5, h4)
        h = self.d2(h, h3)
        h = self.d3(h, h2)
        h = self.d4(h, h1)
        return self.outc(h)

class SoundNet(torch.nn.Module):
    # KL Divergence Loss - Refer https://github.com/Kajiyu/Modern_SoundNet/blob/master/soundnet.ipynb 
    def __init__(self):
        super(SoundNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = torch.nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = torch.nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = torch.nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = torch.nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = torch.nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = torch.nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = torch.nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = torch.nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8_1 = torch.nn.Conv1d(1024, 1000, 4, stride=2, padding=0)
        self.conv8_2 = torch.nn.Conv1d(1024, 401, 4, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(859000, 1000)
        self.fc2 = torch.nn.Linear(344459, 365)

    def forward(self, input_wav):
        x = self.pool1(F.relu(torch.nn.BatchNorm1d(16)(self.conv1(input_wav))))
        x = self.pool2(F.relu(torch.nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(torch.nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(torch.nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(torch.nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(torch.nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(torch.nn.BatchNorm1d(1024)(self.conv7(x)))
        x_object = Flatten()(F.relu(self.conv8_1(x)))
        x_place = Flatten()(F.relu(self.conv8_2(x)))
        x_object = self.fc1(x_object)
        x_place = self.fc2(x_place)
        y = [x_object, x_place]
        return y

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