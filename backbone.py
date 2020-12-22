import torch
from torch import nn

class Backbone(nn.Module):
    def __init__(self,num_classes=10):
        super(Backbone, self).__init__()
        self.num_classes = num_classes
        self.layer1 = Basicblock(3, 64)
        self.layer2 = Basicblock(64, 128)
        self.layer3 = Basicblock(128, 196)
        # TODO: remember to flatten before fc
        self.fc1 = nn.Linear(3136, 256)
        self.fc2 = nn.Linear(256,num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm1d(256)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Basicblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3):
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel)
        self.conv_2 = nn.Conv2d(in_channel, out_channel, kernel)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x




