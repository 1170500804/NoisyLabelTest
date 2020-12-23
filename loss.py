import torch
from torch import nn
import numpy as np

class ReverseCrossEntropy(nn.Module):
    def __init__(self, neg_inf=-4):
        super(ReverseCrossEntropy, self).__init__()
        self.A = neg_inf

    def forward(self,input, target):
        gt = torch.gather(input, 1, target.view(-1,1))
        log_target = torch.ones(input.size()) * self.A
        log_target = torch.cat([torch.ones([input.size(0),1]) * (-self.A), log_target], dim=1)
        _input = torch.cat([gt, input], dim=1)
        assert _input.size() == log_target.size,f'{_input.size()},{log_target.size()}'
        output = _input * log_target
        output = torch.sum(output)
        return output

class MixedEntropy(nn.Module):
    def __init__(self, neg_inf, alpha, beta):
        super(MixedEntropy, self).__init__()
        self.A = neg_inf
        self.CE = nn.CrossEntropyLoss()
        self.reverse_CE = ReverseCrossEntropy(neg_inf=neg_inf)
        self.alpha = alpha
        self.beta = beta

    def forward(self, target, input):
        reverse_CE_loss = self.reverse_CE(input, target)
        CE_loss = self.CE(input, target)
        loss = CE_loss * self.alpha + reverse_CE_loss * self.beta
        return loss




