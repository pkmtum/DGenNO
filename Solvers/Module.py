# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:12:22 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:12:22 
#  */
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

#############################
class MyDataset(Dataset):

    def __init__(self, x:torch.tensor, y:torch.tensor, z:torch.tensor=None):
        '''
        Input:
            x: size(N,?)
            y: size(N,?)
            z: size(N,?)
        '''
        self.x = x 
        self.y = y 
        self.z = z
    
    def __getitem__(self, index):
        if self.z is not None:
            return self.x[index], self.y[index], self.z[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

#############################
class MyIndex(Dataset):

    def __init__(self, index:torch.tensor):
        '''
        Input:
            index: size(N,)
        '''
        self.index = index
    
    def __getitem__(self, idx):
        return self.index[idx]

    def __len__(self):
        return self.index.shape[0]
    
###############################
class Solver():

    def __init__(self, dtype=torch.float32):
        '''Neural Operator-based PDE solver
        '''
        self.dtype = dtype

    def datasplit(self, x, y, test_rate_or_size:float=0.2):
        ''' Split the data into training set and testing set
        Input:
            x: size(?,d)
            y: size(?,d)
            test_rate_or_size: the rate/size of testing set
        Output:
            x_train, x_test, y_train, y_test
        '''
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_rate_or_size)
        #
        print(f'The train_data size: {x_train.shape}, {y_train.shape}')
        print(f'The test_data size: {x_test.shape}, {y_test.shape}')
        #
        return x_train, x_test, y_train, y_test
        
    def dataloader(self, x:torch.tensor, y:torch.tensor, 
                   z:torch.tensor=None, batch_size:int=100, 
                   shuffle=True):
        '''Prepare the data_loader for training
        Input:
            x: size(N,?)
            y: size(N,?)
            z: size(N,?)
            batch_size: int
        Output:
            train_loader
        '''
        return DataLoader(MyDataset(x,y,z), 
                          batch_size= batch_size, 
                          shuffle=shuffle)

    def indexloader(self, N:int, batch_size:int=100, shuffle=True):
        '''Prepare the index_loader for training
        Input: 
            index: int
        '''
        index = torch.tensor([i for i in range(N)], dtype=torch.int32)
        return DataLoader(MyIndex(index),
                          batch_size=batch_size,
                          shuffle=shuffle)

################
class LossClass(object):

    def __init__(self, solver:Solver, **kwrds):
        self.solver = solver

    def Loss_beta(self):
        '''The loss of beta model
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)

    def Loss_pde(self):
        '''The loss of pde
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)
    
    def Loss_data(self):
        '''The loss of boundary conditions
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)
    
    def Error(self):
        '''The errors
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)

