# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-11-06 16:29:54 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-11-06 16:29:54 
#  */
import numpy as np 
import torch 
from scipy.stats import qmc
import matplotlib.pyplot as plt

######################################## 1d
class Point1D():

    def __init__(self, x_lb=[0.], x_ub=[1.], 
                 dataType=torch.float32,  
                 random_seed=None):
        self.lb = x_lb
        self.ub = x_ub 
        self.dtype = dataType
        #
        np.random.seed(random_seed)
        self.lhs_t = qmc.LatinHypercube(1, seed=random_seed)
        self.lhs_x = qmc.LatinHypercube(1, seed=random_seed+10086)
    
    def inner_point(self, nx:int, nt:int, method='hypercube', 
                    t0=[0.], tT=[1.]):
        ''' The inner points
        Return:
            XT: size(nx*nt, 2)
        '''
        if method=='mesh':
            X = np.linspace(self.lb, self.ub, nx)
            T = np.linspace(t0, tT, nt)
            xx, tt = np.meshgrid(X, T)
            XT = np.vstack((xx.flatten(), tt.flatten())).T
        elif method=='uniform':
            T = np.linspace(t0, tT, nt).repeat(nx, axis=0)
            X = np.random.uniform(self.lb, self.ub, nx*nt).reshape(-1,1)
            XT = np.vstack((X.flatten(), T.flatten())).T
        elif method=='hypercube':
            T = np.linspace(t0, tT, nt).repeat(nx, axis=0)
            X = qmc.scale(self.lhs_x.random(nx*nt), self.lb, self.ub)
            XT = np.vstack((X.flatten(), T.flatten())).T
        else:
            raise NotImplementedError

        return torch.tensor(XT, dtype=self.dtype)
    
    def boundary_point(self, num_sample:int, t0=[0.], tT=[1.]):
        ''' The boundary points
        Return:
        '''
        T = np.linspace(t0, tT, num_sample)
        # The lower boundary
        X = np.array(self.lb).repeat(num_sample, axis=0)
        XT_lb = np.vstack((X.flatten(), T.flatten())).T
        # The upper boundary
        X = np.array(self.ub).repeat(num_sample, axis=0)
        XT_ub = np.vstack((X.flatten(), T.flatten())).T

        return torch.tensor(XT_lb, dtype=self.dtype),\
            torch.tensor(XT_ub, dtype=self.dtype)
    
    def init_point(self, num_sample:int, t_stamp=[0.], method='mesh'):
        ''' The initial points
        '''
        if method=='mesh':
            X = np.linspace(self.lb, self.ub, num_sample)
        elif method=='uniform':
            X = np.random.uniform(self.lb, self.ub, num_sample).reshape(-1,1)
        elif method=='hypercube':
            X = qmc.scale(self.lhs_x.random(num_sample), self.lb, self.ub)
        else:
            raise NotImplementedError
        #
        T = np.array(t_stamp).repeat(num_sample, axis=0)
        XT = np.vstack((X.flatten(), T.flatten())).T

        return torch.tensor(XT, dtype=self.dtype)
        
    def weight_centers(self, n_center:int, nt:int, 
                       Rmax:float=1e-4, Rmin:float=1e-4,
                       method='hypercube', t0=[0.], tT=[1.]):
        '''Generate centers of compact support regions for test functions
        Input:
            n_center: The number of centers
            nt: The number of times
            Rmax: The maximum of Radius
            Rmin: The minimum of Radius
        Return:
            xc: size(?, 1, 1)
            tc: size(?, 1, 1)
            R: size(?, 1, 1)
        '''
        if Rmax<Rmin:
            raise ValueError('R_max should be larger than R_min.')
        elif (2.*Rmax)>np.min(np.array(self.ub) - np.array(self.lb)):
            raise ValueError('R_max is too large.')
        elif (Rmin)<1e-4 and self.dtype is torch.float32:
            raise ValueError('R_min<1e-4 when data_type is torch.float32!')
        elif (Rmin)<1e-10 and self.dtype is torch.float64:
            raise ValueError('R_min<1e-10 when data_type is torch.float64!')
        #
        R = np.random.uniform(Rmin, Rmax, [n_center*nt, 1])
        lb, ub = np.array(self.lb) + R, np.array(self.ub) - R # size(?,2)
        #
        if method=='mesh':
            X = np.linspace([0.], [1.], n_center)
            T = np.linspace(t0, tT, nt)
            xx, tt = np.meshgrid(X, T)
            X, T = xx.reshape(-1,1), tt.reshape(-1, 1)
        elif method=='uniform':
            X = np.random.uniform([0.], [1.], n_center*nt).reshape(-1,1)
            T = np.linspace(t0, tT, nt).repeat(n_center, axis=0)
        elif method=='hypercube':
            X = self.lhs_x.random(n_center*nt)
            T = np.linspace(t0, tT, nt).repeat(n_center, axis=0)
        else:
            raise NotImplementedError
        # 
        xc = X * (ub - lb) + lb

        return torch.tensor(xc, dtype=self.dtype).view(-1,1,1),\
            torch.tensor(T, dtype=self.dtype).view(-1,1,1),\
                torch.tensor(R, dtype=self.dtype).view(-1, 1, 1)
    
    def integral_grid(self, n_mesh_or_grid:int=9, method='mesh'):
        '''Meshgrid for calculating integrals in [-1.,1.]
        Input:
            n_mesh_or_grid: the number of meshgrids
            method: the way of generating mesh
        Ouput:
            grid: size(?, 1)
        '''
        if method=='mesh':
            grid = np.linspace(-1., 1., n_mesh_or_grid).reshape(-1,1)
        else:
            raise NotImplementedError(f'No {method} method.')
        #
        return torch.tensor(grid, dtype=self.dtype)

    def _test_plot1d(self, point_dict:dict, title=''):
        ''' Visualize the generated points
        '''
        fig = plt.figure(figsize=(9,6))
        xc, tc, R = point_dict['xc'], point_dict['tc'], point_dict['R']
        grid = point_dict['grid']
        print('xc shape:', xc.shape, 'tc shape:', tc.shape, 'R shape:', 
              R.shape, 'grid shape:', grid.shape)
        x_all = (xc + R*grid).reshape(-1,1)
        t_all = tc.repeat(1, grid.shape[0], 1)
        plt.scatter(t_all, x_all, label='all')
        plt.scatter(tc, xc, label='weight_centes')
        #
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        plt.show()

######################################## 2d
class Point2D():

    def __init__(self, x_lb=[0., 0.], x_ub=[1., 1.], 
                 dataType=torch.float32,  
                 random_seed=None):
        self.lb = x_lb
        self.ub = x_ub 
        self.dtype = dataType
        #
        np.random.seed(random_seed)
        self.lhs_t = qmc.LatinHypercube(1, seed=random_seed)
        self.lhs_x = qmc.LatinHypercube(2, seed=random_seed)
    
    def inner_point(self, nx:int, nt:int, method='hypercube', 
                    t0=[0.], tT=[1.]):
        ''' The inner points
        Return:
            XT: size(nx*nt, 2)
        '''
        if method=='mesh':
            x_mesh = np.linspace(self.lb[0], self.ub[0], nx)
            y_mesh = np.linspace(self.lb[1], self.ub[1], nx)
            x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
            X = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T
            T = np.linspace(t0, tT, nt).repeat(X.shape[0], axis=0)
            #
            XT = np.concatenate([np.tile(X, (nt,1)), T.reshape(-1,1)], axis=-1)
        elif method=='hypercube':
            T = np.linspace(t0, tT, nt).repeat(nx, axis=0)
            X = qmc.scale(self.lhs_x.random(nx*nt), self.lb, self.ub)
            XT = np.concatenate([X, T.reshape(-1,1)], axis=-1)
        else:
            raise NotImplementedError

        return torch.tensor(XT, dtype=self.dtype)
    
    def boundary_point(self, nx_each_edge:int, nt:int, t0=[0.], tT=[1.], 
                       method='hypercube'):
        ''' The boundary points
        Return:
            XT: size(nt*nx_each_edge*4, 3)
        '''
        T = np.linspace(t0, tT, nt).repeat(nx_each_edge*4, axis=0)
        #
        X_bd = []
        for _ in range(nt):
            # For each time_stamp, generate different boundary points
            for d in range(2):
                if method=='mesh':
                    x_mesh = np.linspace(self.lb[0], self.ub[0], nx_each_edge)
                    y_mesh = np.linspace(self.lb[1], self.ub[1], nx_each_edge)
                    X_lb = np.vstack((x_mesh, y_mesh)).T
                    X_ub = np.vstack((x_mesh, y_mesh)).T
                elif method=='uniform':
                    X_lb = np.random.uniform(self.lb, self.ub, (nx_each_edge,2))
                    X_ub = np.random.uniform(self.lb, self.ub, (nx_each_edge,2))
                elif method=='hypercube':
                    X_lb = qmc.scale(self.lhs_x.random(nx_each_edge), 
                                    np.array(self.lb), np.array(self.ub))
                    X_ub = qmc.scale(self.lhs_x.random(nx_each_edge), 
                                    np.array(self.lb), np.array(self.ub))
                else:
                    raise NotImplementedError
                # The lower boundary on dimension d
                X_lb[:,d]=self.lb[d]
                X_bd.append(X_lb)
                # The upper boundary on dimension d
                X_ub[:,d]=self.ub[d]
                X_bd.append(X_ub)
        X_bd = np.concatenate(X_bd, axis=0)
        #
        XT = np.concatenate([X_bd, T.reshape(-1,1)], axis=-1)

        return torch.tensor(XT, dtype=self.dtype)
    
    def init_point(self, nx_or_mesh:int, t_stamp=[0.], 
                   method='mesh'):
        ''' The initial points
        '''
        if method=='mesh':
            x_mesh = np.linspace(self.lb[0], self.ub[0], nx_or_mesh)
            y_mesh = np.linspace(self.lb[1], self.ub[1], nx_or_mesh)
            x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
            X = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T
        elif method=='uniform':
            x_rand = np.random.uniform(self.lb[0], self.ub[0], nx_or_mesh)
            y_rand = np.random.uniform(self.lb[1], self.ub[1], nx_or_mesh)
            X = np.vstack((x_rand.flatten(), y_rand.flatten())).T
        elif method=='hypercube':
            X = qmc.scale(self.lhs_x.random(nx_or_mesh), self.lb, self.ub)
        else:
            raise NotImplementedError
        #
        T = np.array(t_stamp).repeat(X.shape[0], axis=0)
        XT = np.concatenate([X, T.reshape(-1,1)], axis=-1)

        return torch.tensor(XT, dtype=self.dtype)
        
    def weight_centers(self, n_center:int, nt:int, 
                       Rmax:float=1e-4, Rmin:float=1e-4,
                       t0=[0.], tT=[1.]):
        '''Generate centers of compact support regions for test functions
        Input:
            n_center: The number of centers
            nt: The number of times
            Rmax: The maximum of Radius
            Rmin: The minimum of Radius
        Return:
            xc: size(?, 1, 1)
            tc: size(?, 1, 1)
            R: size(?, 1, 1)
        '''
        if Rmax<Rmin:
            raise ValueError('R_max should be larger than R_min.')
        elif (2.*Rmax)>np.min(np.array(self.ub) - np.array(self.lb)):
            raise ValueError('R_max is too large.')
        elif (Rmin)<1e-4 and self.dtype is torch.float32:
            raise ValueError('R_min<1e-4 when data_type is torch.float32!')
        elif (Rmin)<1e-10 and self.dtype is torch.float64:
            raise ValueError('R_min<1e-10 when data_type is torch.float64!')
        #
        R = np.random.uniform(Rmin, Rmax, [n_center*nt, 1])
        lb, ub = np.array(self.lb) + R, np.array(self.ub) - R # size(?,2)
        #
        T = np.linspace(t0, tT, nt).repeat(n_center, axis=0)
        X = self.lhs_x.random(n_center*nt)
        xc = X * (ub - lb) + lb

        return torch.tensor(xc, dtype=self.dtype).view(-1,1,2),\
            torch.tensor(T, dtype=self.dtype).view(-1,1,1),\
                torch.tensor(R, dtype=self.dtype).view(-1, 1, 1)
    
    def integral_grid(self, n_mesh_or_grid:int=9, method='mesh'):
        '''Meshgrid for calculating integrals in [-1.,1.]^2
        Input:
            n_mesh_or_grid: the number of meshgrids
            method: the way of generating mesh
        Ouput:
            grid: size(?, 2)
        '''
        if method=='mesh':
            x_mesh, y_mesh = np.meshgrid(np.linspace(-1., 1., n_mesh_or_grid), 
                                         np.linspace(-1., 1., n_mesh_or_grid))
            grid = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            #
            index = np.where(np.linalg.norm(grid, axis=1, keepdims=True) <1.)[0]
            grid_scaled = grid[index,:]
        else:
            raise NotImplementedError(f'No {method} method.')
        #
        return torch.tensor(grid_scaled, dtype=self.dtype)

    def _test_plot2d(self):
        ''' '''
        ############### The inner point
        xt_in = self.inner_point(nx=4, nt=3, method='hypercube')
        ############## The boundary point
        xt_bd = self.boundary_point(nx_each_edge=4, nt=3, method='mesh')
        ############## Initial point
        xt_init = self.init_point(nx_or_mesh=20, t_stamp=[0.], method='hypercube')
        ############## Particles
        xc, tc, R = self.weight_centers(n_center=1, nt=100)
        xt_nc = torch.cat([xc, tc], dim=-1).squeeze(1)
        print('xc shape:', xc.shape, 'tc shape:', tc.shape, 'R shape:', R.shape)
        ############## int_grids
        int_grids = self.integral_grid(n_mesh_or_grid=50, method='mesh')
        print('The number of int_grids:', int_grids.shape)
        ###############
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xt_in[...,-1], xt_in[...,0], xt_in[...,1], 
        #            c='r', marker='o')
        ax.scatter(xt_bd[...,-1], xt_bd[...,0], xt_bd[...,1],
                   label='boundary points')
        ax.scatter(xt_init[...,-1], xt_init[...,0], xt_init[...,1], 
                   label='init points')
        ax.scatter(xt_nc[...,-1], xt_nc[...,0], xt_nc[...,1], 
                   label='particles')
        #
        ax.set_xlabel('t (s)')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        #
        plt.legend()
        plt.show()
        ###############
        plt.figure()
        plt.scatter(int_grids[:,0], int_grids[:,1], label='int_grids')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__=='__main__':
    ############# For 1d testing
    # demo = Point1D(x_lb=[-1.], x_ub=[1.], random_seed=1234)
    # grid = demo.integral_grid(n_mesh_or_grid=20)
    # xc, tc, R = demo.weight_centers(n_center=1, nt=10, Rmax=0.1, Rmin=0.1, method='hypercube')
    # demo._test_plot1d({'xc':xc, 'tc':tc, 'grid':grid, 'R': R})
    ############ For 2d testing
    demo = Point2D(x_lb=[0., 0.], x_ub=[1., 1.], random_seed=1234)
    demo._test_plot2d()