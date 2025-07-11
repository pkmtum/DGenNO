import torch 
import torch.nn as nn
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation
#
try:
    from Networks.DeepONets_strategy import (
        SingleOutputStrategy,
        IndependentStrategy,
        SplitBranchStrategy,
        SplitTrunkStrategy,
        SplitBothStrategy)
except:
    from DeepONets_strategy import (
        SingleOutputStrategy,
        IndependentStrategy,
        SplitBranchStrategy,
        SplitTrunkStrategy,
        SplitBothStrategy)
#
try:
    from Networks.FCNet import FCNet
    from Networks.CNNet import CNNet2d
except:
    from FCNet import FCNet
    from CNNet import CNNet2d

class DeepONetBatch(nn.Module):

    def __init__(self, num_output, 
                 layers_branch, layers_trunk,
                 activation_branch:str=None,
                 activation_trunk:str=None,
                 multi_output_strategy:str=None,
                 kernel_init=None,
                 dtype=None,
                 device='cpu') -> None:
        super(DeepONetBatch, self).__init__()
        '''Deep operator network for dataset in the format of Batch product.
        '''
        self.num_output = num_output
        self.kernel_init = kernel_init
        self.dtype = dtype
        self.device = device
        ################### Activation
        self.activation_branch = FunActivation()(activation_branch)
        self.activation_trunk =FunActivation()(activation_trunk)
        #################### Multi_output_strategy
        if self.num_output == 1:
            if multi_output_strategy is not None:
                raise ValueError("num_outputs is set to 1,"
                    "but multi_output_strategy is not None.")
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(f"Warning: There are {num_output} outputs, "
                "but no multi_output_strategy selected."
                'Use "independent" as the multi_output_strategy.')
        #
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)
        #
        self.branch, self.trunk = self.multi_output_strategy.build(
            layers_branch, layers_trunk)
        #
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_output)]
            )
    
    def build_branch_net(self, layers_branch):
        if callable(layers_branch[0]):
            # User-defined network
            return layers_branch[0].to(self.device)
        else:
            # Fully connected network
            return FCNet(layers_branch, self.activation_branch, 
                         self.dtype, self.kernel_init).to(self.device)

    def build_trunk_net(self, layer_sizes_trunk):
        #
        return FCNet(layer_sizes_trunk, self.activation_trunk, 
                     self.dtype, self.kernel_init).to(self.device)

    def merge_branch_trunk(self, x_func, x_loc, index):
        '''
        Input:
            x_loc: size(n_batch, n_mesh, out_size)
            x_func: size(n_batch, out_size)
        '''
        y = torch.einsum("bmi,bi->bm", x_loc, x_func)
        y += self.b[index]

        return y.unsqueeze(-1)

    @staticmethod
    def concatenate_outputs(ys):
        return torch.concat(ys, dim=-1)

    def forward(self, x_loc, x_func):
        '''
        Input:
            x_loc: size(n_batch, n_mesh, dx)
            x_func: size(n_batch, n_mesh)
        '''
        x = self.multi_output_strategy.call(x_func, x_loc)

        return x

####################################
class DeepONetCartesianProd(nn.Module):

    def __init__(self, num_output, 
                 layers_branch, layers_trunk,
                 activation_branch:str=None,
                 activation_trunk:str=None,
                 multi_output_strategy:str=None,
                 kernel_init=None, dtype=None, 
                 device='cpu') -> None:
        super(DeepONetCartesianProd, self).__init__()
        '''Deep operator network for dataset in the format of Cartesian product.
        '''
        self.num_output = num_output
        self.kernel_init = kernel_init
        self.dtype = dtype
        self.device = device
        ################### Activation
        self.activation_branch = FunActivation()(activation_branch)
        self.activation_trunk =FunActivation()(activation_trunk)
        #################### Multi_output_strategy
        if self.num_output == 1:
            if multi_output_strategy is not None:
                raise ValueError("num_outputs is set to 1,"
                    "but multi_output_strategy is not None.")
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(f"Warning: There are {num_output} outputs, "
                "but no multi_output_strategy selected."
                'Use "independent" as the multi_output_strategy.')
        #
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)
        #
        self.branch, self.trunk = self.multi_output_strategy.build(
            layers_branch, layers_trunk)
        #
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_output)]
            )
    
    def build_branch_net(self, layers_branch):
        if callable(layers_branch[0]):
            # User-defined network
            return layers_branch[0].to(self.device)
        else:
            # Fully connected network
            return FCNet(layers_branch, self.activation_branch, 
                         self.dtype, self.kernel_init).to(self.device)

    def build_trunk_net(self, layer_sizes_trunk):
        #
        return FCNet(layer_sizes_trunk, self.activation_trunk, 
                     self.dtype, self.kernel_init).to(self.device)

    def merge_branch_trunk(self, x_func, x_loc, index):
        '''
        Input:
            x_func: size(n_batch, out_size)
            x_loc: size(n_mesh, out_size)
        '''
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]

        return y.unsqueeze(-1)

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=2)

    def forward(self, x_loc, x_func):
        '''
        Input:
            x_loc: size(n_mesh, dx)
            x_func: size(n_batch, n_mesh)
        '''
        x = self.multi_output_strategy.call(x_func, x_loc)

        return x
    
#######################################################   
class _Test_Branch(nn.Module):

    def __init__(self, conv_arch:list, fc_arch:list, 
                 nx_size:int, ny_size:int, dtype=None):
        super(_Test_Branch, self).__init__()
        '''
        '''
        self.nx_size, self.ny_size = nx_size, ny_size
        self.conv = CNNet2d(conv_arch=conv_arch, fc_arch=fc_arch, 
                            kernel_size=(5,5), stride=2, dtype=dtype)
        
    def forward(self, x):
        '''
        Input:
            x: size(?, ny*nx)
        Return:
            output: size(?, fc_arch[-1])
        '''
        x = x.reshape(x.shape[0], self.ny_size, self.nx_size).unsqueeze(1)
        x = self.conv(x)

        return x