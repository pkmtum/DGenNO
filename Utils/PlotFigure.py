# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:13:24 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:13:24 
#  */
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator
#
import scienceplots
plt.rcParams['axes.titlesize'] = 26 
plt.rcParams['axes.labelsize'] = 24  
plt.rcParams['xtick.labelsize'] = 18  
plt.rcParams['ytick.labelsize'] = 18  
plt.rcParams['legend.fontsize'] = 24
# plt.rcParams['font.sans-serif'] = ['simhei'] 
# plt.rcParams['font.weight'] = 'light'
# plt.rcParams['axes.unicode_minus'] = False  
# plt.rcParams['axes.linewidth'] = 1
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['savefig.format'] = 'pdf'
#
colormap = 'jet'
# colormape = 'seismic'

class Plot():

    def show_error(time_list:list[np.array], error_list:list[np.array], 
                   label_list:list[str], save_path:str=None, title:str=None)->None:
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(7,5))
            for time, error, label in zip(time_list, error_list, label_list):
                plt.semilogy(time.ravel(), error.ravel(), linewidth=1.5, label=label)
            plt.xlabel('time(s)')
            plt.ylabel(r'Relative error (avg)')
            plt.tight_layout()
            plt.legend()
        #
        if title is not None:
            plt.title(title)
        #
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        #
        plt.show()

    def show_loss(loss_list:list[np.array], label_list:list[str], 
                  save_path:str=None)->None:
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(7,5))
            for loss, label in zip(loss_list, label_list):
                plt.semilogy(loss.ravel(), linewidth=1.5, label=label)
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.tight_layout()
            plt.legend()
        # save the figure
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        plt.show()

    def show_1d_list(x_list, y_list, label_list, title=None, 
                     pause=None, save_path=None, point_list=None,
                     lb=-1., ub=1.):
        x_plot = np.linspace(lb, ub, 100)
        plt.figure(figsize=(8,5))
        #
        if type(x_list).__name__!='list':
            x_list = [x_list] * len(y_list)
        #
        with plt.style.context(['science', 'no-latex']):
            for x,y,label in zip(x_list, y_list, label_list):
                y_plot = griddata(x.flatten(), y.flatten(), x_plot, method='cubic')
                plt.plot(x_plot, y_plot, '-.', linewidth=3., label=label)
            ## Plot the points
            if point_list is not None:
                for points in point_list:
                    plt.scatter(points[:,0], np.zeros_like(points[:,0]), s=20, lw=1.)
            ## Show the title
            if title is not None:
                plt.title(title)
            #
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.legend()
        # save the figure
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        # pause the figure
        if pause is not None:
            plt.pause(pause)
            plt.close()

        plt.show()

    def show_1dt(xt, u, title=None, pause=None, 
                 save_path=None, point_list=None,
                 t0=0., tT=1., lb=-1., ub=1.):
        mesh = np.meshgrid(np.linspace(lb, ub, 100), np.linspace(t0, tT, 200))
        x_plot, t_plot = mesh[0], mesh[1]
        x, t = xt[...,0], xt[...,-1]
        #
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
        #
        z_plot = griddata((t.flatten(), x.flatten()), np.ravel(u), (t_plot, x_plot), method='cubic')
        cntr = axs.contourf(t_plot, x_plot, z_plot, levels=14, cmap=colormap)
        fig.colorbar(cntr, ax=axs)
        axs.set_xlabel('t')
        axs.set_ylabel('x')
        #
        if point_list is not None:
            for points in point_list:
                plt.scatter(points[:,-1], points[:,0], s=20, lw=2., marker='o')
        # show the title
        if title is not None:
            axs.set_title(title)
        # save the figure 
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        # pause the figure
        if pause is not None:
            plt.pause(pause)
            plt.close()
        #
        plt.show()
        
    #######################
    def show_2d(x, y, title=None, pause=None, 
                save_path=None, point_list=None,
                lb=-1., ub=1.):
        if isinstance(lb, list):
            mesh = np.meshgrid(np.linspace(lb[0], ub[0], 100), 
                               np.linspace(lb[1], ub[1], 100))
        else:
            mesh = np.meshgrid(np.linspace(lb, ub, 100), 
                               np.linspace(lb, ub, 100))
        x_plot, y_plot = mesh[0], mesh[1]
        #
        with plt.style.context(['science', 'no-latex']):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
            #
            z_plot = griddata((x[:,0], x[:,1]), np.ravel(y), (x_plot, y_plot), method='cubic')
            cntr = axs.contourf(x_plot, y_plot, z_plot, levels=14, cmap=colormap)
            fig.colorbar(cntr, ax=axs)
            #
            if point_list is not None:
                for points in point_list:
                    plt.scatter(points[:,0], points[:,1], s=20, lw=2., marker='o')
            #
            axs.set_xlabel('x')
            axs.set_ylabel('y')
            plt.tight_layout()
        # show the title
        if title is not None:
            axs.set_title(title)
        # save the figure 
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        # pause the figure
        if pause is not None:
            plt.pause(pause)
            plt.close()
        #
        plt.show()

    def show_2d_list(x_list, y_list, label_list, pause=None, 
                     save_path=None, lb=-1., ub=1.):
        n_col = len(y_list)
        if type(x_list).__name__!='list':
            x_list = [x_list] * n_col
        #
        if isinstance(lb, list):
            mesh = np.meshgrid(np.linspace(lb[0], ub[0], 100), 
                               np.linspace(lb[1], ub[1], 100))
        else:
            mesh = np.meshgrid(np.linspace(lb, ub, 100), 
                               np.linspace(lb, ub, 100))
        x_plot, y_plot = mesh[0], mesh[1]
        #
        with plt.style.context(['science', 'no-latex']):
            fig, axs = plt.subplots(nrows=1, ncols=n_col, figsize=(6*n_col,5))
            for i, x, y, title in zip([i for i in range(n_col)],x_list,y_list,label_list):
                #
                z_plot = griddata((x[:,0], x[:,1]), np.ravel(y), (x_plot, y_plot), 
                                  method='cubic')
                cntr = axs.flat[:][i].contourf(x_plot, y_plot, z_plot, 
                                               levels=14, cmap=colormap)
                fig.colorbar(cntr, ax=axs.flat[:][i])
                # 
                axs.flat[:][i].set_title(title)
                axs.flat[:][i].set_xlabel('x')
                axs.flat[:][i].set_ylabel('y')
            #
            plt.tight_layout()
        # save the figure 
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        # pause the figure
        if pause is not None:
            plt.pause(pause)
            plt.close()
        #
        plt.show()

    ########################

    def show_3d_list(x_list:np.array, Z_list:list, label_list:list, 
                     pause:float=None, save_path:str=None,
                     lb=-1., ub=1.)->None:
        '''show 3d figure
            mesh: [(n,n), (n,n)]
            Z: (n,n)
        '''
        n_col = np.ceil(len(Z_list)/2)
        fig, axs = plt.subplots(nrows=int(n_col), ncols=2, figsize=(11,6),
                                subplot_kw={"projection": "3d"})
        #
        if isinstance(lb, list):
            mesh = np.meshgrid(np.linspace(lb[0], ub[0], 100), 
                               np.linspace(lb[1], ub[1], 100))
        else:
            mesh = np.meshgrid(np.linspace(lb, ub, 100), 
                               np.linspace(lb, ub, 100))
        x_plot, y_plot = mesh[0], mesh[1]
        if type(x_list).__name__!='list':
            x_list = [x_list] * len(Z_list)
        #
        for i in range(len(Z_list)):
            z_plot = griddata((x_list[i][:,0], x_list[i][:,1]), np.ravel(Z_list[i]), 
                              (x_plot, y_plot), method='linear')
            axs.flat[:][i].plot_surface(x_plot, y_plot, z_plot, cmap=cm.coolwarm, 
                                        linewidth=0, antialiased=False)
            #
            axs.flat[:][i].set_title(label_list[i])
            axs.flat[:][i].set_xlabel('x')
            axs.flat[:][i].set_ylabel('y')
            axs.flat[:][i].zaxis.set_major_locator(LinearLocator(5))
            # A StrMethodFormatter is used automatically
            axs.flat[:][i].zaxis.set_major_formatter('{x:.02f}')
        #
        plt.tight_layout()
        # save the figure 
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        # pause the figure
        if pause is not None:
            plt.pause(pause)
            plt.close()
        #
        plt.show()