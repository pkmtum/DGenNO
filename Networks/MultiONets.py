# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:06:51 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:06:51 
#  */
import torch 
import torch.nn as nn 
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation

class MultiONetBatch(nn.Module):

    def __init__(self, in_size_x:int, in_size_a:int,
                 trunk_layers: list[int],
                 branch_layers: list[int],
                 activation_trunk='SiLU_Sin',
                 activation_branch='SiLU', 
                 sum_layers:int=4, dtype=None):
        super(MultiONetBatch, self).__init__()
        assert sum_layers<len(branch_layers)
        self.l=sum_layers
        ################################ Activation
        # For trunk 
        if isinstance(activation_trunk, str):
            self.activation_trunk = FunActivation()(activation_trunk)
        else:
            self.activation_trunk = activation_trunk
        # For branch
        if isinstance(activation_branch, str):
            self.activation_branch = FunActivation()(activation_branch)
        else:
            self.activation_branch = activation_branch
        ################################# The trunk layer
        self.fc_trunk_in = nn.Linear(in_size_x, trunk_layers[0], dtype=dtype)
        trunk_net = []
        hidden_in = trunk_layers[0]
        for hidden in trunk_layers[1:]:
            trunk_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.trunk_net = nn.Sequential(*trunk_net)
        ################################## The branch layer
        self.fc_branch_in = nn.Linear(in_size_a, branch_layers[0], dtype=dtype)
        branch_net = []
        hidden_in = branch_layers[0]
        for hidden in branch_layers[1:]:
            branch_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.branch_net = nn.Sequential(*branch_net)
        ################################## Weights and bias
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.l)]
            )
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
    
    def forward(self, x, a):
        '''
        Input:
            x: size(n_batch, n_mesh, dx)
            a: size(n_batch, latent_size)
        '''
        assert x.shape[0]==a.shape[0]
        ######### The trunk network
        # size(n_batch, n_mesh, hidden_size)
        x = self.activation_trunk(self.fc_trunk_in(x))
        ######### The branch network
        # size(n_batch, hidden_size)
        a = self.activation_branch(self.fc_branch_in(a))
        ######### 
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
        #########
        out = 0.
        for net_t, net_b, w in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:], self.w):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
            out += torch.einsum('bnh,bh->bn', x, a) * w 

        ##### The output layer
        out = out/self.l + self.b

        return out

#####################
class MultiONetBatch_X(nn.Module):
    '''Multi-Input&Output case'''

    def __init__(self, in_size_x:int, in_size_a:int,
                 latent_size:int, out_size:int,
                 trunk_layers: list[int],
                 branch_layers: list[int],
                 activation_trunk='SiLU_Sin',
                 activation_branch='SiLU', 
                 sum_layers:int=4, dtype=None):
        super(MultiONetBatch_X, self).__init__()
        assert sum_layers<len(branch_layers)
        self.l=sum_layers
        ################################ Activation
        # For trunk 
        if isinstance(activation_trunk, str):
            self.activation_trunk = FunActivation()(activation_trunk)
        else:
            self.activation_trunk = activation_trunk
        # For branch
        if isinstance(activation_branch, str):
            self.activation_branch = FunActivation()(activation_branch)
        else:
            self.activation_branch = activation_branch
        ################################# The trunk layer
        self.fc_trunk_in = nn.Linear(in_size_x, trunk_layers[0], dtype=dtype)
        trunk_net = []
        hidden_in = trunk_layers[0]
        for hidden in trunk_layers[1:]:
            trunk_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.trunk_net = nn.Sequential(*trunk_net)
        ################################## The branch layer
        self.fc_branch_in = nn.Linear(in_size_a, branch_layers[0], dtype=dtype)
        branch_net = []
        hidden_in = branch_layers[0]
        for hidden in branch_layers[1:]:
            branch_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.branch_net = nn.Sequential(*branch_net)
        ################################# The output layer
        self.fc_out = nn.Linear(latent_size, out_size, dtype=dtype)
    
    def forward(self, x, a):
        '''
        Input:
            x: size(n_batch, n_mesh, dx)
            a: size(n_batch, latent_size, da)
        '''
        assert x.shape[0]==a.shape[0]
        ######### The trunk network
        # size(n_batch, n_mesh, hidden_size)
        x = self.activation_trunk(self.fc_trunk_in(x))
        ######### The branch network
        # size(n_batch, latent_size, hidden_size)
        a = self.activation_branch(self.fc_branch_in(a))
        #########
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
        #########
        out = 0.
        for net_t, net_b in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
            out += torch.einsum('bnh,bmh->bnm', x, a)

        ##### The output layer
         #size(batch_size, n_mesh, latent_size)  -> size(batch_size, n_mesh, out_size)
        out = self.fc_out(out/self.l)

        return out
    
############################################ Cartesian types
class MultiONetCartesianProd(nn.Module):

    def __init__(self, in_size_x:int, in_size_a:int,
                 trunk_layers: list[int],
                 branch_layers: list[int],
                 activation_trunk='SiLU_Sin',
                 activation_branch='SiLU', 
                 sum_layers:int=4, dtype=None):
        super(MultiONetCartesianProd, self).__init__()
        assert sum_layers<len(branch_layers)
        self.l=sum_layers
        ################################ Activation
        # For trunk 
        if isinstance(activation_trunk, str):
            self.activation_trunk = FunActivation()(activation_trunk)
        else:
            self.activation_trunk = activation_trunk
        # For branch
        if isinstance(activation_branch, str):
            self.activation_branch = FunActivation()(activation_branch)
        else:
            self.activation_branch = activation_branch
        ################################# The trunk layer
        self.fc_trunk_in = nn.Linear(in_size_x, trunk_layers[0], dtype=dtype)
        trunk_net = []
        hidden_in = trunk_layers[0]
        for hidden in trunk_layers[1:]:
            trunk_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.trunk_net = nn.Sequential(*trunk_net)
        ################################## The branch layer
        self.fc_branch_in = nn.Linear(in_size_a, branch_layers[0], dtype=dtype)
        branch_net = []
        hidden_in = branch_layers[0]
        for hidden in branch_layers[1:]:
            branch_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.branch_net = nn.Sequential(*branch_net)
        ################################## Weights and bias
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.l)]
            )
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
    
    def forward(self, x, a):
        '''
        Input:
            x: size(mesh_size, dx)
            a: size(n_batch, latent_size)
        '''
        ######### The trunk network
        # size(n_mesh, hidden_size)
        x = self.activation_trunk(self.fc_trunk_in(x))
        ######### The branch network
        # size(n_batch, hidden_size)
        a = self.activation_branch(self.fc_branch_in(a))
        #########
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
        #########
        out = 0.
        for net_t, net_b, w in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:], self.w):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
            out += torch.einsum('bh,mh->bm', a, x) * w

        ##### The output layer
        # size(n_batch, mesh_size) 
        out = out/self.l + self.b

        return out

#############################
class MultiONetCartesianProd_X(nn.Module):
    '''Multi-Input&Output case'''

    def __init__(self, in_size_x:int, in_size_a:int, 
                 latent_size:int, out_size:int,
                 trunk_layers: list[int],
                 branch_layers: list[int],
                 activation_trunk='SiLU_Sin',
                 activation_branch='SiLU', 
                 sum_layers:int=4, dtype=None):
        super(MultiONetCartesianProd_X, self).__init__()
        assert sum_layers<len(branch_layers)
        self.l=sum_layers
        ################################ Activation
        # For trunk 
        if isinstance(activation_trunk, str):
            self.activation_trunk = FunActivation()(activation_trunk)
        else:
            self.activation_trunk = activation_trunk
        # For branch
        if isinstance(activation_branch, str):
            self.activation_branch = FunActivation()(activation_branch)
        else:
            self.activation_branch = activation_branch
        ################################# The trunk layer
        self.fc_trunk_in = nn.Linear(in_size_x, trunk_layers[0], dtype=dtype)
        trunk_net = []
        hidden_in = trunk_layers[0]
        for hidden in trunk_layers[1:]:
            trunk_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.trunk_net = nn.Sequential(*trunk_net)
        ################################## The branch layer
        self.fc_branch_in = nn.Linear(in_size_a, branch_layers[0], dtype=dtype)
        branch_net = []
        hidden_in = branch_layers[0]
        for hidden in branch_layers[1:]:
            branch_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.branch_net = nn.Sequential(*branch_net)
        ################################## The output layer
        self.fc_out = nn.Linear(latent_size, out_size, dtype=dtype)
    
    def forward(self, x, a):
        '''
        Input:
            x: size(mesh_size, dx)
            a: size(n_batch, latent_size, da)
        '''
        ######### The trunk network
        # size(n_mesh, hidden_size)
        x = self.activation_trunk(self.fc_trunk_in(x))
        ######### The branch network
        # size(n_batch, latent_size, hidden_size)
        a = self.activation_branch(self.fc_branch_in(a))
        #########
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
        #########
        out = 0.
        for net_t, net_b in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
            out += torch.einsum('bmh,nh->bnm', a, x)

        ##### The output layer
        #size(batch_size, n_mesh, latent_size)  -> size(batch_size, n_mesh, out_size)
        out = self.fc_out(out/self.l)

        return out