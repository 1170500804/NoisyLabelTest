import torch
from torch import nn
import numpy as np
from torch.autograd import Function

class ReverseCrossEntropy(Function):
    def __init__(self, neg_inf=-4):
        super(ReverseCrossEntropy, self).__init__()
        self.A = neg_inf

    def forward(self,input, target):
        # _target = target.clone().cpu().numpy().reshape(1,-1)
        # log_target = torch.ones(input.size()) * self.A
        # log_target = log_target.numpy()
        # log_target[np.arange(input.size(0)),_target] = 0
        # log_target = torch.from_numpy(log_target).cuda()
        # input = log_target * input
        # loss = torch.sum(input)
        # return loss
        gt = torch.gather(input, 1, target.view(-1,1))
        log_target = torch.ones(input.size()) * self.A

    def backward(grad_output):
        return grad_output



