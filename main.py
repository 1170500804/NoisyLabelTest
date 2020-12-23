import torch
import numpy as np
from data import cifar10
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

import argparse
import time

from backbone import Backbone
from loss import MixedEntropy

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


parser = argparse.ArgumentParser()
parser.add_argument('--aug', action='store_true')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('-a', '--alpha', type=float, default=None)
parser.add_argument('-b', '--beta', type=float, default=None)
parser.add_argument('-e', '--eta', type=float, default=0.6)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--resume', type=str)
parser.add_argument('--print', type=int, default=20)

args = parser.parse_args()
# check validity of arguments
if args.baseline:
    assert args.alpha is None
    assert args.beta is None
else:
    assert not(args.alpha is None)
    assert not(args.beta is None)

if args.aug:
    augmentation = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
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
    criterion = MixedEntropy()

# training
train_dataset = cifar10(transform=augmentation)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
model.train()
for epoch in range(120):
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % args.print == 0:
            batch_time = time.time() - end
            end = time.time()
            print(f'training loss: {loss},time: {batch_time}')

        if epoch in [20, 40, 60, 80, 100]:
            state_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state_dict, filename=f'/data_b/lius/code/TestCaseCkps/checkpoint_{epoch}.pth.tar')

# save last
state_dict = {
                'epoch': -1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
save_checkpoint(state_dict, filename=f'/data_b/lius/code/TestCaseCkps/checkpoint_{epoch}.pth.tar')











