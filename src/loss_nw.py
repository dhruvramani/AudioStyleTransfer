import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*14*13, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size()[0], 128*14*13)
        x = self.classifier(x)
        return x


class RowCNN(nn.Module):
    def __init__(self, num_classes=10, split_size=250):
        super(RowCNN, self).__init__()
        self.window_sizes = [5, 10, 20, 40, 50, 100]
        self.n_filters = 512
        self.num_classes = num_classes
        self.split_size = split_size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.n_filters, [window_size, self.split_size], padding=(window_size - 1, 0))
            for window_size in self.window_sizes
        ])

        self.linear = nn.Linear(self.n_filters * len(self.window_sizes), self.num_classes)

    def forward(self, x):
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))      
            x2 = torch.squeeze(x2, -1)  
            x2 = F.max_pool1d(x2, x2.size(2)) 
            xs.append(x2)
        x = torch.cat(xs, 2) 
        x = x.view(x.size(0), -1)  
        logits = self.linear(x) 
        return logits
