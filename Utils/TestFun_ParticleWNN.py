# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:14:08 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:14:08 
#  */
import numpy as np 
import torch 
import math
import sys
#
try:
    from Utils.PlotFigure import Plot
except:
    from PlotFigure import Plot

class TestFun_ParticleWNN():
    '''
    (1) Bump 
    (2) Wendland 
    (3) Cosin 
    (4) Wendland with power k (k>=2)
    '''
    def __init__(self, fun_type:str='Cosin', dim:int=1,
                 n_mesh_or_grid:int=9, grid_method:str='mesh', 
                 dataType=torch.float32):
        '''
        Input:
            type: the test function type
            dim: the dimension of the problem
        '''
        self._dim = dim
        self._eps = sys.float_info.epsilon
        # Setups for generating integral_grids
        self._n_mesh_or_grid = n_mesh_or_grid
        self._grid_method = grid_method
        self._dtype = dataType
        #
        fun_dict = {
            "Cosin": self._Cosin,
            "Bump": self._Bump,
            "Wendland": self._Wendland,
            "Wendland_k": self._Wend_powerK}
        #
        if fun_type in fun_dict.keys():
            self.testFun = fun_dict[fun_type]
        else:
            raise NotImplementedError(f'No {fun_type} test function type.')

    def _dist(self, x:torch.tensor)->torch.tensor:
        '''
        Input:
          x: (?,d) or (?, m, d)
        Output:
          y: the norm of x
        '''
        return torch.linalg.norm(x, dim=-1, keepdims=True)

    def _grad(self, x:torch.tensor, y:torch.tensor)->torch.tensor:
        '''
        Input:
            x: the variables
            y: the function values
        Output:
            dy: the grad dy/dx
        '''
        dy = torch.autograd.grad(inputs=x, outputs=y, 
                                 grad_outputs=torch.ones_like(y), 
                                 create_graph=True)[0]
        
        return dy
    
    def _Bump(self, x_mesh:torch.tensor, dim:int=1)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function values
            dv: the grad dv/dx_mesh
        '''
        ############
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        r_list = [r]
        for _ in range(3):
            r_list.append(r*r_list[-1])
        #
        v = torch.exp(1. - 1. / (1. - r_list[1] + self._eps))
        ########## Use definition
        dv_dr_divide_by_r = v * (-2.) / ((1.-r_list[1])**2 + self._eps)
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
        else:
            dv =  dv_dr_divide_by_r * x_mesh
        
        return v.detach(), dv.detach()
    
    def _Wendland(self, x_mesh:torch.tensor, dim:int=1)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function values
            dv: the grad dv/dx_mesh
        '''
        ############ 
        l = math.floor(dim / 2) + 3
        #
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        r_list = [r]
        for _ in range(1):
            r_list.append(r*r_list[-1])
        #
        v = (1-r) ** (l+2) * ( (l**2+4.*l+3.) * r_list[1] + (3.*l+6.) * r + 3.) / 3.
        #
        dv_dr_divide_by_r = (1-r)**(l+1) * (- (l**3+8.*l**2+19.*l+12) * r - (l**2+7.*l+12)) / 3.
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
        else:
            dv =  dv_dr_divide_by_r * x_mesh
        
        return v.detach(), dv.detach()

    def _Cosin(self, x_mesh:torch.tensor, dim:int=1)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function values
            dv: the grad dv/dx_mesh
        '''
        ############ 
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        v = (1. - torch.cos(torch.pi * (r + 1.))) / torch.pi
        #
        dv_dr_divide_by_r = torch.sin(torch.pi * (r+1.)) / (r + self._eps)
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh 
        
        return v.detach(), dv.detach()

    def _Wend_powerK(self, x_mesh:torch.tensor, dim:int=1, k:int=4)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function values
            dv: the grad dv/dx_mesh
        '''
        ############ 
        l = math.floor(dim / 2) + 3
        #
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        r_list = [r]
        for _ in range(1):
            r_list.append(r*r_list[-1])
        #
        v_wend = (1-r) ** (l+2) * ( (l**2+4.*l+3.) * r_list[1] + (3.*l+6.) * r + 3.) / 3.
        dv_dr_divide_by_r_wend = (1-r)**(l+1) * (- (l**3+8.*l**2+19.*l+12) * r - (l**2+7.*l+12)) / 3.
        #
        v = v_wend ** k
        dv_dr_divide_by_r = k * v_wend**(k-1) * dv_dr_divide_by_r_wend
        #
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
        else:
            dv =  dv_dr_divide_by_r * x_mesh

        return v.detach(), dv.detach()

    def integral_grid(self, n_mesh_or_grid:int, method='mesh', 
                      dtype=torch.float32):
        '''Meshgrid for calculating integrals from [-1.,1.]^2
        Input:
            n_mesh_or_grid: the number of meshgrids
            method: the way of generating mesh
        Ouput:
            grid: size(?, d)
        '''
        if method=='mesh':
            if self._dim==1:
                grid_scaled = np.linspace(-1., 1., n_mesh_or_grid).reshape(-1,1) 
            elif self._dim==2:
                x_mesh, y_mesh = np.meshgrid(np.linspace(-1., 1., n_mesh_or_grid), 
                                            np.linspace(-1., 1., n_mesh_or_grid))
                grid = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
                #
                index = np.where(np.linalg.norm(grid, axis=1, keepdims=True) <1.)[0]
                grid_scaled = grid[index,:]
            else:
                NotImplementedError(f'dim>{self._dim} is not available')
        else:
            raise NotImplementedError(f'No {method} method.')
        #
        return torch.tensor(grid_scaled, dtype=dtype)
    
    def get_testFun(self, grids:torch.tensor=None)->torch.tensor:
        '''
        Get the test function
        Input:
            None or grids: size(?,d)
        Output:
            grids, phi, dphi_scaled
        '''
        if grids is None:
            grids = self.integral_grid(self._n_mesh_or_grid, 
                                       self._grid_method, 
                                       self._dtype)
        #
        v, dv = self.testFun(grids, self._dim)

        return grids, v, dv
    
    def __check__(self):
        '''
        check the test function
        '''
        lb, ub = -1., 1.
        if self._dim==1:
            x_in = torch.tensor(np.random.uniform(lb, ub, [1000, 1]).astype(np.float64))
            grid, v, dv = self.get_testFun(x_in) # grid:size(m,d) v:size(m, 1) dv_dr:size(m,d) 
            #
            Plot.show_1d_list(grid, [v, dv], ['v', 'dv'])   # show v vs. dv vs. Lv
        elif self._dim==2:
            x_in = torch.tensor(np.random.uniform(lb, ub, [1000, 2]).astype(np.float64))
            grid, v, dv_dr = self.get_testFun(x_in) #grid:size(m,d) v:size(m, 1) dv_dr:size(m,d) 
            Plot.show_3d_list(grid, [v, dv_dr[:,0:1], dv_dr[:,1:]], ['v', 'dv_dx', 'dv_dy'])

if __name__=='__main__':
    demo = TestFun_ParticleWNN(fun_type='Wendland', dim=2, 
                               n_mesh_or_grid=100)
    demo.__check__()