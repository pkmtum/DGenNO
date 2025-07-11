# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:14:18 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:14:18 
#  */
import numpy as np
import torch

def np2tensor(x:np.array, dtype=torch.float32):
    '''From numpy.array to torch.tensor
    '''
    return torch.tensor(x, dtype=dtype)

def detach2np(x:torch.tensor):
    '''Detach -> cpu -> numpy
    '''
    return x.detach().cpu().numpy()

def mesh1d(n, sub:int=1, low=0., high=1.):
    '''
    '''
    assert low<high
    assert sub<=n
    #
    mesh = np.linspace(low, high, n).reshape(-1,1)

    return mesh[::sub,:]

def mesh2d(nx, ny, subx:int=1, suby:int=1, xlow=0., xhigh=1., ylow=0., yhigh=1.):
    '''
    '''
    assert xlow<xhigh and ylow<yhigh
    assert subx<=nx and suby<=ny
    #
    x_mesh = np.linspace(xlow, xhigh, nx)[::subx]
    y_mesh = np.linspace(ylow, yhigh, ny)[::suby]
    xy_mesh = np.meshgrid(x_mesh, y_mesh)
    mesh = np.vstack([xy_mesh[0].flatten(), xy_mesh[1].flatten()]).T

    return mesh

if __name__=='__main__':
    mesh = mesh2d(nx=5, ny=5, subx=1, suby=1)
    print(mesh)

