# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:13:57 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:13:57 
#  */
import numpy as np
import math
import torch 
import torch.nn as nn
from itertools import combinations_with_replacement
'''
The original code: https://github.com/ArmanMaesumi/torchrbf
'''
class RadialFun(object):

    def __init__(self):
        '''
        Input: 
            eps: shape parameter for the kernel function
        '''
        self.fun = {
        "linear": self.linear,
        "thin_plate_spline": self.thin_plate_spline,
        "cubic": self.cubic,
        "quintic": self.quintic,
        "multiquadric": self.multiquadric,
        "inverse_multiquadric": self.inverse_multiquadric,
        "inverse_quadratic": self.inverse_quadratic,
        "gaussian": self.gaussian }
        #
        self.min_degree = {
            "multiquadric": 0,
            "linear": 0,
            "thin_plate_spline": 1,
            "cubic": 1,
            "quintic": 2
        }
    
    def linear(self, r):
        return r 
    
    def thin_plate_spline(self, r, min_eps=1e-7):
        r = torch.clamp(r, min=min_eps)
        return r
    
    def cubic(self, r):
        return r**3
    
    def quintic(self, r):
        return -r**5
    
    def multiquadric(self, r):
        return -torch.sqrt(r**2 + 1)

    def inverse_multiquadric(self, r):
        return 1/torch.sqrt(r**2 + 1)

    def inverse_quadratic(self, r):
        return 1/(r**2 + 1)

    def gaussian(self, r):
        return torch.exp(-r**2)

class RBFInterpolator(nn.Module):

    def __init__(self, x_mesh:torch.tensor,
                 kernel:str=None, eps:float=None, 
                 degree:int=None, smoothing:float=0., 
                 dtype=torch.float32) -> None:
        super(RBFInterpolator, self).__init__()
        '''Radial basis function interpolator in Pytorch.
        Input:
            x_mesh: size(n_mesh, d)
            kernel: str, The kernel type.
            eps: float, shape parameter for the kernel function.
            degree: int, degree of the polynomial added to the interpolation function
            smoothing: float or (n,), tensor of smoothing parameters
            dtype: the datatype of tensors
        '''
        assert x_mesh.ndim==2
        #
        scale_fun = {"linear", "thin_plate_spline", "cubic", "quintic"}
        self.x_mesh = x_mesh
        self.n_mesh, self.dx = x_mesh.shape
        self.dtype = dtype
        ######## Setup of kernel function
        self.Kernels = RadialFun()
        self.kernel_fun = self.Kernels.fun[kernel]
        ####### Setup of eps
        if eps is None:
            if kernel in scale_fun:
                self.eps = 1.
            else:
                raise ValueError('Require eps for the kernel.')
        else:
            self.eps = float(eps)
        ######## Setup of smoothing
        if isinstance(smoothing, (int, float)):
            self.smoothing = torch.full((self.n_mesh,), smoothing, dtype=dtype).to(x_mesh.device)
        elif isinstance(smoothing, np.ndarray):
            smoothing = torch.tensor(smoothing, dtype=dtype).to(x_mesh.device)
        elif isinstance(smoothing, torch.tensor):
            smoothing = smoothing.to(x_mesh.device)
        else:
            raise ValueError('smoothing must be a scalar or a 1-dimensional tensor')
        ######### Setup of degree
        min_degree = self.Kernels.min_degree.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError('degree must be at least -1')
            elif degree < min_degree:
                raise ValueError(f'degree must be larger than {min_degree}')
        ########### Setup of powers: size(n_monos, d)
        self.powers = self.monomial_powers(self.dx, degree).to(x_mesh.device)
        self.n_monos = self.powers.shape[0]
        if self.n_monos > self.n_mesh:
            raise ValueError('The data is not compatible with the requested degree')
        ########### Get the Matrix A
        # lhs: size(n_mesh+n_monos, n_mesh+n_monos)
        # shift: size(dx, )
        # scale: size(dx, )
        self.lhs, self.shift, self.scale= self.build()
        # x_eps_mesh: size(n_mesh, dx)
        self.x_eps_mesh = self.x_mesh * self.eps

    def forward(self, x:torch.tensor, a_batch:torch.tensor):
        '''Returns the interpolated data at the given points `x`
        Input:
            x: size(n_batch, nx, d)
            a_batch: size(n_batch, n_mesh, 1)
        Return:
            a_pred: size(n_batch, nx)
        '''
        assert x.ndim==3 and a_batch.ndim==3
        assert a_batch.shape[1]==self.n_mesh
        ######################## Get the coeff
        # coeff: size(n_batch, n_mesh+n_monos, 1)
        self.coeff = self.solve(a_batch)
        ######################## Get the intropolation
        # x_eps: size(n_batch, nx, d); x_hat: size(n_batch, nx, d)
        x_eps = x * self.eps
        x_hat = (x - self.shift) / self.scale
        # # kv: size(n_batch, nx, n_mesh)
        x_eps_mesh = self.x_eps_mesh.repeat((x.shape[0], 1, 1))
        kv = self.kernel_matrix(x_eps, x_eps_mesh)
        # # pmat: size(n_batch, nx, n_monos)
        pmat = self.polynomial_matrix(x_hat, self.powers)
        # # vec: size(n_batch, nx, n_mesh+n_monos)
        vec = torch.cat([kv, pmat], dim=-1)
        # # (n_batch, nx, n_mesh+n_momos) * (n_batch, n_mesh+n_monos, 1) -> (n_batch, nx, 1)
        a_pred = torch.matmul(vec, self.coeff)

        return a_pred

    def solve(self, a_batch:torch.tensor):
        '''Build then solve the RBF linear system
        Input:
            a_batch: size(n_batch, n_mesh, 1)
        Return:
            coeffs: size(n_batch, n_mesh+n_monos, 1)
        '''
        assert a_batch.ndim==3 and a_batch.shape[1]==self.n_mesh
        # The left hand side: 
        # size(n_mesh+n_monos, n_mesh+n_monos) -> (n_batch, n_mesh+n_monos, n_mesh+n_monos)
        lhs = self.lhs.repeat((a_batch.shape[0], 1, 1))
        # The right hand side: size(n_batch, n_mesh+n_monos, 1)
        rhs = torch.empty((a_batch.shape[0], self.n_monos+self.n_mesh, 1), 
                          dtype=self.dtype, device=a_batch.device)
        rhs[:, :self.n_mesh, :] = a_batch
        rhs[:, self.n_mesh:, :] = 0.
        ############################
        try:
            coeffs = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            msg = 'Singlar matrix.'
            if self.n_monos>0:
                pmat = self.polynomial_matrix((self.x_mesh - self.shift) / self.scale, self.powers)
                rank = torch.linalg.matrix_rank(pmat) # size(n_batch,)
                flag = rank < self.n_monos # rank less than n_monos -> not full rank
                if sum(flag) > 0:
                    index = torch.tensor([i for i in range(1, a_batch.shape[0]+1)])
                    index = index[flag==True]
                    msg = ("Singular matrix. The matrix of monomials evaluated at" 
                            f"the data point coordinates ({index}) does not have full column" 
                            f"rank ({rank[flag]}/{self.n_monos}).")
            raise ValueError(msg)
        #
        return coeffs

    def build(self):
        '''Build the linear equation: lhs * coeff = rhs
        '''
        # mins, maxs: size(dx,)
        mins = torch.min(self.x_mesh, dim=0).values 
        maxs = torch.max(self.x_mesh, dim=0).values
        shift = (maxs + mins) / 2.
        scale = (maxs - mins) / 2.
        scale[scale==0.] = 1.
        # x_eps, x_hat: size(n_mesh, dx)
        x_eps = self.x_mesh * self.eps
        x_hat = (self.x_mesh - shift) / scale
        # lhs: size(n_mesh+n_monos, n_mesh+n_monos)
        lhs = torch.empty((self.n_mesh+self.n_monos, self.n_mesh+self.n_monos), 
                          dtype=self.dtype, device=self.x_mesh.device)
        lhs[:self.n_mesh, :self.n_mesh] = self.kernel_matrix(x_eps, x_eps)
        lhs[:self.n_mesh, self.n_mesh:] = self.polynomial_matrix(x_hat, self.powers)
        lhs[self.n_mesh:, :self.n_mesh] = lhs[:self.n_mesh, self.n_mesh:].T
        lhs[self.n_mesh:, self.n_mesh:] = 0.
        lhs[:self.n_mesh, :self.n_mesh] += torch.diag(self.smoothing)

        return lhs, shift, scale

    def kernel_matrix(self, x_eps, x_eps_base):
        '''Returns radial function values for all pairs of points in `x`
        '''
        return self.kernel_fun(torch.cdist(x_eps, x_eps_base))
    
    def polynomial_matrix(self, x_hat, powers):
        '''Evaluate monomials at `x` with given `powers`
        Input:
            x_hat: size(n_batch, nx, dx) or size(n_mesh, dx)
            powers: size(n_monos, dx)
        Out:
            out: size(n_batch nx, n_monos) or size(n_mesh, n_monos)
        '''
        if x_hat.ndim==3:
            # x_: size(n_batch, nx*n_monos, d)
            x_ = torch.repeat_interleave(x_hat, repeats=powers.shape[0], dim=1)
            # powers_: size(nx*n_monos, d)
            powers_ = powers.repeat(x_hat.shape[1], 1)
            # out: size(n_batch, nx, n_monos)
            out = torch.prod(x_**powers_, dim=-1, keepdim=True).view(
                x_hat.shape[0], x_hat.shape[1], powers.shape[0])
        elif x_hat.ndim==2:
            # x_: size(n_mesh*n_monos, dx)
            x_ = torch.repeat_interleave(x_hat, repeats=powers.shape[0], dim=0)
            # powers_: size(n_mesh*n_monos, dx)
            powers_ = powers.repeat(x_hat.shape[0], 1)
            # out: size(n_mesh, n_monos)
            out = torch.prod(x_**powers_, dim=1).view(x_hat.shape[0], powers.shape[0])
        else:
            raise TypeError('x_has has a wrong type.')

        return out

    def monomial_powers(self, dx:int, degree:int):
        '''Return the powers for each monomial in a polynomial.
        Input:
            dx: int, Number of variables in the polynomial.
            degree: int, Degree of the polynomial.
        Output: 
            out: size(n_monos, dx), Array where each row contains the powers 
                for each variable in a monomial.
        '''
        n_monos = math.comb(degree+dx, dx)
        out = torch.zeros( (n_monos, dx), dtype=torch.int32)
        count = 0 
        for deg in range(degree+1):
            for mono in combinations_with_replacement(range(dx), deg):
                for var in mono:
                    out[count, var] += 1
                count += 1
        return out 

if __name__=='__main__':
    #
    grid_x, grid_y = np.mgrid[0:1:5j, 0:1:5j]
    grid = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    #
    a1 = np.sin(grid[:,0])*np.cos(grid[:,1])
    a2 = np.sin(grid[:,0])*np.cos(grid[:,1])
    a_batch = np.array([a1, a2]).reshape(2, -1, 1)
    #
    grid = torch.tensor(grid.astype(np.float32))
    a_batch = torch.tensor(a_batch.astype(np.float32))
    print('a shape', a_batch.shape, 'grid shape', grid.shape)
    #
    demo = RBFInterpolator(grid, kernel='gaussian', eps=10., smoothing=0., 
                           degree=2, dtype=torch.float32)
    a_pred = demo(grid.repeat((2,1,1)), a_batch)
    print('The shape of coeffs:', demo.coeff.shape)
    print(a_batch.shape, a_pred.shape)
    print(torch.sum( (a_batch-a_pred)**2, dim=1, keepdim=True))