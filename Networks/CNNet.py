# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:09:18 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:09:18 
#  */
import torch.nn as nn
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation

############### 1d CNN+FC
class CNNet1d(nn.Module):

    def __init__(self, conv_arch:list, fc_arch:list, 
                 activation_conv:str='Tanh', activation_fc:str='Tanh', 
                 kernel_size=5, stride:int=3, dtype=None):
        super(CNNet1d, self).__init__()
        # Activation
        if isinstance(activation_conv, str):
            self.activation_conv = FunActivation()(activation_conv)
        else:
            self.activation_conv = activation_conv
        # Activation
        if isinstance(activation_fc, str):
            self.activation_fc = FunActivation()(activation_fc)
        else:
            self.activation_fc = activation_fc
        # The Conv layer:
        net = []
        self.arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            net.append(nn.Conv1d(self.arch_in, arch, kernel_size=kernel_size, 
                                 stride=stride, dtype=dtype))
            self.arch_in =  arch 
        self.conv_net = nn.Sequential(*net)
        # The fc layer
        net = []
        self.arch_in = fc_arch[0]
        for arch in fc_arch[1:]:
            net.append(nn.Linear(self.arch_in, arch, dtype=dtype))
            self.arch_in = arch
        self.fc_net = nn.Sequential(*net)

    def forward(self, x):
        '''
        Input: 
            x: size(batch_size, conv_arch[0], m_size)
        Return: 
            out: size(batch_size, fc_arch[-1])
        '''
        ######## The conv layer: 
        # (n_batch, conv_arch[0], mesh_size) -> (n_batch, conv_arch[-1], mesh_size)
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation_conv(x)
        ######## The fc layer 
        # size(n_batch, conv_arch[-1]*?) -> (n_batch, fc_arch[-1])
        x = x.reshape(x.shape[0], -1)
        for fc in self.fc_net[:-1]:
            x = fc(x)
            x = self.activation_fc(x)
        ######## The output layer
        x = self.fc_net[-1](x)

        return x

################## 2d pure
class CNNPure2d(nn.Module):

    def __init__(self, conv_arch:list, activation:str='Tanh', 
                 kernel_size=(3,3), stride:int=2, dtype=None):
        super(CNNPure2d, self).__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # The Conv layer:
        net = []
        self.arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            net.append(nn.Conv2d(self.arch_in, arch, kernel_size=kernel_size, 
                                 stride=stride, dtype=dtype))
            self.arch_in =  arch 
        self.conv_net = nn.Sequential(*net)

    def forward(self, x):
        '''
        Input: 
            x: size(batch_size, conv_arch[0], my_size, mx_size)
        Return: 
            out: size(batch_size, conv_arch[-1], my_szie, mx_size)
        '''
        # (n_batch, conv_arch[0], ny, nx) -> (n_batch, conv_arch[-1], ?, ?)
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation(x)

        return x

############### 2d CNN+FC
class CNNet2d(nn.Module):

    def __init__(self, conv_arch:list, fc_arch:list, 
                 activation_conv:str='Tanh', activation_fc:str='Tanh', 
                 kernel_size=(5,5), stride:int=3, dtype=None):
        super(CNNet2d, self).__init__()
        # Activation
        if isinstance(activation_conv, str):
            self.activation_conv = FunActivation()(activation_conv)
        else:
            self.activation_conv = activation_conv
        # Activation
        if isinstance(activation_fc, str):
            self.activation_fc = FunActivation()(activation_fc)
        else:
            self.activation_fc = activation_fc
        # The Conv layer:
        net = []
        self.arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            net.append(nn.Conv2d(self.arch_in, arch, kernel_size=kernel_size, 
                                 stride=stride, dtype=dtype))
            self.arch_in =  arch 
        self.conv_net = nn.Sequential(*net)
        # The fc layer
        net = []
        self.arch_in = fc_arch[0]
        for arch in fc_arch[1:]:
            net.append(nn.Linear(self.arch_in, arch, dtype=dtype))
            self.arch_in = arch
        self.fc_net = nn.Sequential(*net)

    def forward(self, x):
        '''
        Input: 
            x: size(batch_size, conv_arch[0], my_size, mx_size)
        Return: 
            out: size(batch_size, fc_arch[-1])
        '''
        ######## The conv layer: 
        # (n_batch, conv_arch[0], ny, nx) -> (n_batch, conv_arch[-1], ?, ?)
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation_conv(x)
        ######## The fc layer 
        # size(n_batch, conv_arch[-1]*?*?) -> (n_batch, fc_arch[-1])
        x = x.reshape(x.shape[0], -1)
        for fc in self.fc_net[:-1]:
            x = fc(x)
            x = self.activation_fc(x)
        ######## The output layer
        x = self.fc_net[-1](x)

        return x