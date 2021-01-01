import torch
from torch import nn
import numpy as np

class ReverseCrossEntropy(nn.Module):
    def __init__(self, neg_inf=-4):
        super(ReverseCrossEntropy, self).__init__()
        self.A = neg_inf
        self.check = 0
        self.softmax = nn.Softmax()

    def forward(self,input, target):
        input = self.softmax(input)
        if self.check == 0:
            self.check += 1
            print(f'input: {input.size()}')
            print(f'target: {target.size()}')
        gt = torch.gather(input, 1, target.view(-1,1).long())
        log_target = (torch.ones(input.size()) * self.A).cuda()
        log_target = torch.cat([(torch.ones([input.size(0),1]) * (-self.A)).cuda(), log_target], dim=1)
        _input = torch.cat([gt, input], dim=1)
        # assert _input.size() == log_target.size,f'{_input.size()},{log_target.size()}'
        output = _input * log_target
        output = torch.sum(output) / (output.size(0))
        # print(output)
        return output

class MixedEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=0.6, neg_inf=-4):
        super(MixedEntropy, self).__init__()
        print(f'constructing Mixed Entropy loss: alpha={alpha}, beta={beta}')
        self.A = neg_inf
        self.CE = nn.CrossEntropyLoss()
        self.reverse_CE = ReverseCrossEntropy(neg_inf=neg_inf)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        reverse_CE_loss = self.reverse_CE(input, target)
        CE_loss = self.CE(input, target)
        loss = CE_loss * self.alpha - reverse_CE_loss * self.beta
        return loss




