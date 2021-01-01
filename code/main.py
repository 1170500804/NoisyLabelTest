import torch
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import datetime
import os

from backbone import Backbone
from loss import MixedEntropy
from data import cifar10
from evaluate import validate


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    end = time.time()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.long().cuda()
        outputs = model(images)
        if epoch == 0 and i == 0:
            print(f'outputs: {outputs.size()}')
            print(f'labels: {labels.size()}')
        loss = criterion(outputs, labels)
        total_loss += (loss.clone() * images.size(0)).cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.print == 0:
            batch_time = time.time() - end
            end = time.time()
            # print(optimizer.param_groups[0]['lr'])
            print(
                f'epoch: {epoch}, batch: {i}, training loss: {loss}, lr: {optimizer.param_groups[0]["lr"]}, time: {batch_time}')
    scheduler.step()
    return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('-a', '--alpha', type=float, default=None)
    parser.add_argument('-b', '--beta', type=float, default=None)
    parser.add_argument('-e', '--eta', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--print', type=int, default=20)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--exp-aug', action='store_true')

    args = parser.parse_args()
    # check validity of arguments
    if args.baseline:
        assert args.alpha is None
        assert args.beta is None
    else:
        if not args.eval:
            assert not (args.alpha is None)
            assert not (args.beta is None)
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    if args.aug:
        augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    else:
        augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    # configuration
    currentTime = datetime.datetime.now()
    currentTime = currentTime.strftime('%m%d%H%M%S')
    writer = SummaryWriter()
    model = Backbone()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
    if not args.eval:
        train_dataset = cifar10(transform=augmentation, eta=args.eta)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
    else:
        test_dataset = cifar10(transform=augmentation, if_test=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.workers, pin_memory=True)
    prefix = f'baseline_{currentTime}' if args.baseline else f'{args.alpha}_{args.beta}_{args.eta}_{currentTime}'
    if args.baseline:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = MixedEntropy(alpha=args.alpha, beta=args.beta).cuda()

    # training
    if args.eval:
        if args.resume:
            dicts = torch.load(args.resume)
            model_dict = dicts['state_dict']
            model.load_state_dict(model_dict)
            model.cuda()
        else:
            raise RuntimeError('please specify the path to the model')
        acc = validate(test_loader, model, args)
        print(f'the accuracy is: {acc}')

    else:
        model.train()
        for epoch in range(120):
            print(f'training epoch {epoch}!')
            total_train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
            total_train_loss /= train_dataset.__len__()
            writer.add_scalar('Train/Loss', total_train_loss, epoch)
            if epoch in [20, 40, 60, 80, 100]:
                state_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                if not os.path.exists(os.path.join('/data_b/lius/code/TestCaseCkps', prefix)):
                    os.makedirs(os.path.join('/data_b/lius/code/TestCaseCkps', prefix))
                save_checkpoint(state_dict,
                                filename=f'/data_b/lius/code/TestCaseCkps/{prefix}/checkpoint_{epoch}_{currentTime}.pth.tar')

            print(f'validating epoch {epoch}!')

            # acc, prec, recall, f1 = validate(test_loader, model, epoch, args)
            # writer.add_scalar('Val/Accracy', acc, epoch)
            # writer.add_scalar('Val/Precision', prec, epoch)
            # writer.add_scalar('Val/Recall', recall, epoch)
            # writer.add_scalar('Val/f1_score', f1, epoch)

        # save last
        state_dict = {
            'epoch': -1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'sceduler': scheduler.state_dict()
        }

        save_checkpoint(state_dict,
                        filename=f'/data_b/lius/code/TestCaseCkps/{prefix}/checkpoint_{epoch}_{currentTime}.pth.tar')
