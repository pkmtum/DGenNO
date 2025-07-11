# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:12:02 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:12:02 
#  */
import torch
import time 
import os
import scipy.io
from tqdm import trange
#
from Solvers import Module
from Networks.FCNet import FCNet
from Networks.MultiONets import MultiONetBatch, MultiONetBatch_X
from Networks.DeepONets import DeepONetBatch
#
from Utils.RBFInterpolatorMesh import RBFInterpolator
from Utils.Losses import MyError, MyLoss

##########################
class Solver(Module.Solver):

    def __init__(self, device='cuda:0', 
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        #
        self.iter = 0
        self.time_list = []
        self.loss_train_list = []
        self.loss_test_list = []
        #
        self.loss_data_list = []
        self.loss_pde_list = []
        #
        self.error_list = []
        #
        self.error_setup()
        self.loss_setup()
    
    def loadModel(self, path:str, name:str):
        '''Load trained model
        '''
        return torch.load(path+f'{name}.pth', map_location=self.device)

    def saveModel(self, path:str, name:str, model_dict:dict):
        '''Save trained model (the whole model)
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        #
        torch.save(model_dict, path+f'{name}.pth')

    def loadLoss(self, path:str, name:str):
        '''Load saved losses
        '''
        loss_dict = scipy.io.loadmat(path+f'{name}.mat')

        return loss_dict

    def saveLoss(self, path:str, name:str):
        '''Save losses
        '''
        dict_loss = {}
        dict_loss['loss_train'] = self.loss_train_list
        dict_loss['loss_test'] = self.loss_test_list
        dict_loss['loss_data'] = self.loss_data_list 
        dict_loss['loss_pde'] = self.loss_pde_list
        dict_loss['error'] = self.error_list
        dict_loss['time'] = self.time_list
        scipy.io.savemat(path+f'{name}.mat', dict_loss)

    def callBack(self, loss_train, loss_data, loss_pde,
                 loss_test, error_test, t_start):
        '''call back
        '''
        self.loss_train_list.append(loss_train.item())
        self.loss_test_list.append(loss_test.item())
        self.loss_data_list.append(loss_data.item())
        self.loss_pde_list.append(loss_pde.item())
        self.time_list.append(time.time()-t_start)
        #
        if isinstance(error_test, list):
            errs = [err.item() for err in error_test]
            self.error_list.append(errs)
        else:
            self.error_list.append(error_test.item())

    def error_setup(self, err_type:str='lp_rel', d:int=2, p:int=2, 
                    size_average=True, reduction=True):
        ''' setups of error
        Input:
            err_type: from {'lp_rel', 'lp_abs'}
        '''
        Error = MyError(d, p, size_average, reduction)
        #
        if err_type=='lp_rel':
            self.getError = Error.Lp_rel
        elif err_type=='lp_abs':
            self.getError = Error.LP_abs
        else:
            raise NotImplementedError(f'{err_type} has not defined.')
    
    def loss_setup(self, loss_type:str='mse_org', 
                   size_average=True, reduction=True):
        '''setups of loss
        Input:
            loss_type: from {'mse_org', 'mse_rel'}
        '''
        Loss = MyLoss(size_average, reduction)
        #
        if loss_type=='mse_org':
            self.getLoss = Loss.mse_org
        elif loss_type=='mse_rel':
            self.getLoss = Loss.mse_rel
        else:
            raise NotImplementedError(f'{loss_type} has not defined.')
        
    def getModel_a(self, Exact_a:object=None, approximator:str='RBF',
                   **kwrds):
        '''The model for coefficient a: 
        If the exact definition of a is given, then use it.
        Ohterwise, using the approximator instead (which requires the meshgrid 
        where the value of a is available.)
        '''
        if Exact_a is not None:
            print('Using the exact definition of a.')
            model_a = Exact_a
        elif approximator=='RBF':
            x_mesh = kwrds['x_mesh'].to(self.device)
            model_a = RBFInterpolator(
                x_mesh=x_mesh, kernel=kwrds['kernel'], 
                eps=kwrds['eps'], degree=kwrds['degree'], 
                smoothing=kwrds['smoothing'], 
                dtype=self.dtype).to(self.device)
        else:
            raise NotImplementedError(f'No such approximator: {approximator}.')
        
        return model_a

    def getModel(self, x_in_size:int, beta_in_size:int, 
                 trunk_layers:list, branch_layers:list,
                 latent_size:int=None, out_size:int=1, 
                 activation_trunk='SiLU_Sin', activation_branch='SiLU', 
                 netType:str='MultiONetBatch', **kwrds):
        '''Get the neural network model
        '''
        if netType=='MultiONetBatch':
            model = MultiONetBatch(
                in_size_x=x_in_size, in_size_a=beta_in_size, 
                 trunk_layers=trunk_layers,
                 branch_layers=branch_layers,
                 activation_trunk=activation_trunk,
                 activation_branch=activation_branch,
                dtype= self.dtype, **kwrds)
        elif netType=='MultiONetBatch_X':
            model = MultiONetBatch_X(
                in_size_x=x_in_size, in_size_a=beta_in_size,
                latent_size=latent_size, out_size=out_size,
                 trunk_layers=trunk_layers,
                 branch_layers=branch_layers,
                 activation_trunk=activation_trunk,
                 activation_branch=activation_branch,
                dtype= self.dtype, **kwrds)
        elif netType=='DeepONetBatch':
            model = DeepONetBatch(dtype= self.dtype, **kwrds)
        elif netType=='FCNet':
            model = FCNet(dtype=self.dtype, **kwrds)
        else:
            raise NotImplementedError
        
        return model.to(self.device)
    
    def train_setup(self, model_dict:dict, lr:float=1e-3, 
                    optimizer='Adam', scheduler_type:str=None,
                    step_size=500, gamma=1/3, patience=20, factor=1/2):
        '''Setups for training
        '''
        self.model_dict = model_dict
        ########### The models' parameters
        param_list = []
        for model in model_dict.values():
            param_list += list(model.parameters())
        ########### Set the optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(
                params=param_list, lr=lr, weight_decay=1e-4)
        elif optimizer=='AdamW':
            self.optimizer = torch.optim.AdamW(
                params=param_list, lr=lr, weight_decay=1e-4)
        elif optimizer=='RMSprop':
            self.optimizer = torch.optim.RMSprop(
                params=param_list, lr=lr, weight_decay=1e-4)
        else:
            raise NotImplementedError
        # ####### Set the scheduler
        if scheduler_type=='StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, 
                gamma=gamma, last_epoch=-1)
        elif scheduler_type=='Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience)
        self.scheduler_type = scheduler_type
        #
        self.t_start = time.time()
        self.best_err_test = 1e10
    
    def train(self, LossClass:Module.LossClass, 
              a_train, u_train, x_train,
              a_test, u_test, x_test, 
              w_pde:float=1., w_data:float=1.,
              batch_size:int=100, epochs:int=1,
              epoch_show:int=10, shuffle=True, **kwrds):
        '''Train the model
        '''
        ############# The training and testing data
        train_loader = self.dataloader(a_train, u_train, x_train, 
                                       batch_size=batch_size, shuffle=shuffle)
        ############# The training process
        for epoch in trange(epochs):
            loss_train_sum, loss_data_sum, loss_pde_sum = 0., 0., 0.
            for a, u, x in train_loader:
                lossClass = LossClass(self)
                ############# Calculate losses
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                #
                loss_pde = lossClass.Loss_pde(a)
                loss_data = lossClass.Loss_data(x, a, u)
                loss_train = w_data * loss_data + w_pde * loss_pde
                #
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                self.iter += 1
                #
                loss_train_sum += loss_train
                loss_data_sum += loss_data
                loss_pde_sum += loss_pde
            ###################### The testing loss and error
            a, u, x = a_test.to(self.device), u_test.to(self.device), x_test.to(self.device)
            lossClass = LossClass(self)
            try: # when no gradients are required 
                with torch.no_grad():
                    loss_test = lossClass.Loss_data(x, a, u)
                    error_test = lossClass.Error(x, a, u)
            except: # when gradients are required
                loss_test = lossClass.Loss_data(x, a, u)
                error_test = lossClass.Error(x, a, u)
            #
            self.callBack(loss_train_sum/len(train_loader), loss_data_sum/len(train_loader), 
                          loss_pde_sum/len(train_loader), loss_test, error_test, self.t_start)
            #
            if isinstance(error_test, list):
                error_test = sum(error_test)/len(error_test)
            if error_test.item()<self.best_err_test:
                self.best_err_test = error_test.item()
                self.saveModel(kwrds['save_path'], 'model_dgno_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type=='Plateau':
                self.scheduler.step(error_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss:{loss_train_sum.item()/len(train_loader):.4f}, loss_pde:{loss_pde_sum.item()/len(train_loader):.4f}, loss_data:{loss_data_sum.item()/len(train_loader):.4f}')
                for para in self.optimizer.param_groups:
                    print(f"                l2_test:{error_test.item():.4f}, lr:{para['lr']}")
        ########################
        self.saveModel(kwrds['save_path'], name='model_dgno_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_dgno')
        print(f'The total training time is {time.time()-self.t_start:.4f}')

    def train_index(self, LossClass:Module.LossClass, 
                    a_train, a_test,
                    w_pde:int=1., w_data:int=1.,
                    batch_size:int=100, epochs:int=1,
                    epoch_show:int=10, shuffle=True, **kwrds):
        '''Train the model
        '''
        ############# The training and testing data
        train_loader = self.indexloader(a_train.shape[0],
                                        batch_size=batch_size, shuffle=shuffle)
        test_loader = self.indexloader(a_test.shape[0],
                                        batch_size=batch_size, shuffle=False)
        ############# The training process
        for epoch in trange(epochs):
            loss_train_sum, loss_data_sum, loss_pde_sum = 0., 0., 0.
            for index in train_loader:
                lossClass = LossClass(self)
                ############# Calculate losses
                loss_pde = lossClass.Loss_pde(index, 'train')
                loss_data = lossClass.Loss_data(index, 'train')
                loss_train = w_data * loss_data + w_pde * loss_pde
                #
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                self.iter += 1
                #
                loss_train_sum += loss_train
                loss_data_sum += loss_data
                loss_pde_sum += loss_pde
            ###################### The testing loss and error
            lossClass = LossClass(self)
            #
            loss_test_sum = 0.
            for idx in test_loader:
                try: # when no gradients are required 
                    with torch.no_grad():
                        loss_test = lossClass.Loss_data(idx, 'test')
                        error_test = lossClass.Error()
                except: # when gradients are required
                    loss_test = lossClass.Loss_data(idx, 'test')
                    error_test = lossClass.Error()
                loss_test_sum += loss_test
            #
            self.callBack(loss_train_sum/len(train_loader), loss_data_sum/len(train_loader), 
                          loss_pde_sum/len(train_loader), loss_test_sum/len(test_loader), 
                          error_test, self.t_start)
            #
            if isinstance(error_test, list):
                error_test = sum(error_test)/len(error_test)
            if error_test.item()<self.best_err_test:
                self.best_err_test = error_test.item()
                self.saveModel(kwrds['save_path'], 'model_dgno_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type=='Plateau':
                self.scheduler.step(error_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss_train:{loss_train_sum.item()/len(train_loader):.4f}, loss_pde:{loss_pde_sum.item()/len(train_loader):.4f}, loss_data:{loss_data_sum.item()/len(train_loader):.4f}')
                for para in self.optimizer.param_groups:
                    print(f"                l2_test:{error_test.item():.4f}, lr:{para['lr']}")
        ########################
        self.saveModel(kwrds['save_path'], name='model_dgno_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_dgno')
        print(f'The total training time is {time.time()-self.t_start:.4f}')