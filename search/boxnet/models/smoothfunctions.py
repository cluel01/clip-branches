import torch
import torch.nn as nn

def scaled_sigmoid(logits, tau=0.1):
    # scaled sigmoid version (still differentiable)
    logits = logits / tau
    y_soft = torch.sigmoid(logits)

    # "step function" (not differentiable)
    threshold = 0.5
    y_hard = (y_soft >= threshold).float()
    # forward pass: simply y_hard
    # backward pass: 
    # - grad of y_hard is zero
    # - grad of y_soft.detach() is zero (due to detach)
    # - hence: gradient of y is y_soft in the backward pass
    y = y_hard - y_soft.detach() + y_soft

    return y_soft

def smooth_max(x, alpha=0.1):
    y_hard = torch.max(x, -1)[0]
    
    e_x = torch.exp(x/alpha)
    y_soft = torch.sum(x*e_x, axis=-1) / torch.sum(e_x, axis=-1)
    
    y = y_hard - y_soft.detach() + y_soft

    return y_soft