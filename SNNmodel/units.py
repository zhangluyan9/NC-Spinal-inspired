from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=7):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

class Clamp_q_(nn.Module):
    def __init__(self, min=0.0, max=1,q_level = 7):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max
        self.q_level = q_level

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = Quantization_(x, self.q_level)
        return x
    
    
class CustomDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.data = data['data']
        self.labels = data['label']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class Totensor_(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        x_ = torch.tensor(tensor)
        
        x_=x_.float()
        
        return x_
    
class AddQuantization(object):
    def __init__(self, min=0., max=1,T=7,Hybrid=False):
        self.min = min
        self.max = max
        self.T = T
        self.Hybrid=Hybrid
        
    def __call__(self, tensor):
        #print(self.Hybrid)
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        if self.Hybrid== 'True':
            return torch.clamp(tensor,0,1)
        else:
            return torch.clamp(torch.div(torch.floor(torch.mul(tensor, self.T)), self.T),0,1)
