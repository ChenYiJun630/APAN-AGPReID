import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, init_type='normal'):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        if init_type == 'normal':
            nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))
        elif init_type == 'xaiver':
            nn.init.xavier_normal_(self.cheby_coeffs)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(self.cheby_coeffs)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = torch.tanh(x)
        
        if len(x.shape) == 2:
            
            cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
            if self.degree > 0:
                cheby[:, :, 1] = x
            for i in range(2, self.degree + 1):
                cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()

            y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)

            return y
        
        elif len(x.shape) == 3:
            
            cheby = torch.ones(x.shape[0], x.shape[1], self.inputdim, self.degree + 1, device=x.device)
            if self.degree > 0:
                cheby[:, :, :, 1] = x
            for i in range(2, self.degree + 1):
                cheby[:, :, :, i] = 2 * x * cheby[:, :, :, i - 1].clone() - cheby[:, :, :, i - 2].clone()

            y = torch.einsum('bnid,iod->bno', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)

            return y
        
class ChebyKAN(nn.Module):
    def __init__(self, width=None, degree=4, init_type='normal'):
        super(ChebyKAN, self).__init__()

        self.width = width 
        self.depth = len(width) - 1

        act_fun, norm_fun = [], []
        for l in range(self.depth):
            act_fun.append(ChebyKANLayer(self.width[l], self.width[l + 1], degree, init_type=init_type)) 
            norm_fun.append(nn.LayerNorm(self.width[l + 1]))

        self.act_fun = nn.ModuleList(act_fun)

        self.degree = degree
        
    def forward(self, x):

        for l in range(self.depth):
            x = self.act_fun[l](x)  

        return x
