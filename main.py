import torch
import numpy as np
from data import cifar10
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

import argparse

from backbone import Backbone
from loss import ReverseCrossEntropy

parser = argparse.ArgumentParser()
parser.add_argument('--aug', action='store_true')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('-a', '--alpha', type=float, default=None)
parser.add_argument('-b', '--beta', type=float, default=None)
parser.add_argument('-e', '--eta', type=float, default=0.6)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--resume', type=str)

args = parser.parse_args()
# check validity of arguments
if args.baseline:
    assert args.a is None
    assert args.b is None
else:
    assert not(args.a is None)
    assert not(args.b is None)

if args.aug:
    augmentation = transforms.Compose([transforms.RandomApply(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
else:
    augmentation = transforms.ToTensor()

# configuring model
model = Backbone()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
if args.baseline:
    criterion = nn.CrossEntropyLoss()
else:
    reversed = ReverseCrossEntropy()
    MCE = nn.CrossEntropyLoss()
# training
train_dataset = cifar10(transform=augmentation)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
for epoch in range(120):







