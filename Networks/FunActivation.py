# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-09-05 15:19:50 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-09-05 15:19:50 
#  */
import torch.nn as nn
import torch
import math

class Sinc(nn.Module):

    def __init__(self):
        super(Sinc, self).__init__()
    
    def forward(self, x):
        return x * torch.sin(x)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * nn.functional.sigmoid(x)

class Tanh_Sin(nn.Module):

    def __init__(self):
        super(Tanh_Sin, self).__init__()
        self.main_act = nn.Tanh()

    def forward(self, x):
        x1 = torch.sin(math.pi * (x+1.))

        return self.main_act(x1) + x

class SiLU_Sin(nn.Module):

    def __init__(self):
        super(SiLU_Sin, self).__init__()
        self.main_act = nn.SiLU()

    def forward(self, x):
        x1 = torch.sin(math.pi * (x+1.))

        return self.main_act(x1) + x

class SiLU_Id(nn.Module):

    def __init__(self):
        super(SiLU_Id, self).__init__()
        self.main_act = nn.SiLU()

    def forward(self, x):

        return self.main_act(x) + x

class FunActivation:

    def __init__(self, **kwrds):
        self.activation = {
            'Identity': nn.Identity(),
            'ReLU': nn.ReLU(),
            'ELU': nn.ELU(),
            'Softplus': nn.Softplus(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'Swish': Swish(),
            'Sinc': Sinc(),
            'Tanh_Sin': Tanh_Sin(),
            'SiLU_Sin': SiLU_Sin(),
            'SiLU_Id': SiLU_Id(),
            }
    
    def __call__(self, type=str):
        return self.activation[type]
