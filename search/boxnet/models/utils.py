import numpy as np
import torch
import torch.nn as nn

def numpy_to_torch(data, device):
    if type(data) is np.ndarray:
        data = torch.from_numpy(data).float()
    
    if device != "cpu":
        data = data.to(device)

    return data

class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedBCELoss, self).__init__()
        assert len(weight) == 2
        self.weight = weight

    def forward(self, output, target):
        output = torch.clamp(output,min=1e-8,max=1-1e-8)  
       
        
        loss = self.weight[1] * (target * torch.log(output)) + \
                self.weight[0] * ((1 - target) * torch.log(1 - output))

        return torch.neg(torch.mean(loss))

