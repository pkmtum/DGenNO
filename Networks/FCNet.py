# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:09:45 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:09:45 
#  */
import torch.nn as nn
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation

class FCNet(nn.Module):

    def __init__(self, layers_list:list, activation:str='Tanh', 
                 dtype=None, kernel_init=None):
        super(FCNet, self).__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # Network Sequential
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:]:
            net.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in = hidden
        self.net = nn.Sequential(*net)

    def forward(self, x):
        for net in self.net[:-1]:
            x = net(x)
            x = self.activation(x)
        # Output layer
        x = self.net[-1](x)

        return x