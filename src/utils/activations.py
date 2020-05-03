import torch
import torch.nn as nn


def swish(x):
    """
    Simple implementation of Swish activation function
    https://arxiv.org/pdf/1710.05941.pdf
    """
    return x * torch.sigmoid(x)

def mish(x):
    """
    Simple implementation of Mish activation Function
    https://arxiv.org/abs/1908.08681
    """
    tanh = nn.Tanh()
    softplus = nn.Softplus()
    return x * tanh( softplus(x))

def penalized_tanh(x):
    """
    http://aclweb.org/anthology/D18-1472
    """
    alpha = 0.25
    return torch.max(torch.tanh(x), alpha*torch.tanh(x))