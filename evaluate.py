import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time



def validate(val_loader, model, args):
    end = time.time()
    out = None
    tar = None
    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        # print(i)
        images = images.cuda()
        labels = labels.long()
        outputs = model(images)

        if out is None:
            out = outputs.detach().clone().cpu()
            tar = labels.detach().clone().cpu()
        else:
            out = torch.cat([out, outputs.detach().clone().cpu()])
            tar = torch.cat([tar, labels.detach().clone().cpu()])

        if i == 0:
            print(f'outputs: {outputs.size()}')
            print(f'labels: {labels.size()}')


        if i % args.print == 0:
            # print('print fre')
            _out = torch.argmax(out, dim=1)
            print(_out.size())
            print(tar.size())
            acc = np.sum((_out.clone().numpy() == tar.clone().numpy())) / _out.size(0)
            batch_time = time.time() - end
            print(f'batch: {i}, accuracy: {acc}, time: {batch_time}')

    out = torch.argmax(out, dim=1)
    out = out.view(1,-1).numpy()
    tar = tar.view(1,-1).numpy()
    acc = np.sum((out == tar))/out.shape[1]
    return acc

