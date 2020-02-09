#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import math
from torch import nn
from torch.nn import functional as F

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class gelu_new(nn.Module):
    def __init__(self):
        super(gelu_new, self).__init__()
        #Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        #Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.sigmoid(x)




ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}