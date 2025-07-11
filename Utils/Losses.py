# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:13:02 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:13:02 
#  */
import torch 

class MyError(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(MyError, self).__init__()
        '''Relative/Absolute Lp Error
        Input:
            d: dimension of problem
            p: norm order
        '''
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = 1e-8

    def LP_abs(self, y_pred:torch.tensor, y_true:torch.tensor):
        ''' Absolute Error
        Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        '''
        assert y_true.shape==y_pred.shape
        batch_size = y_true.shape[0]
        #Assume uniform mesh
        h = 1.0 / (y_true.shape[1] - 1.0)
        #
        total_norm = (h**(self.d/self.p))*torch.norm(y_true.reshape(batch_size,-1) 
                                                      - y_pred.reshape(batch_size,-1), self.p, 1)
        #
        if self.reduction:
            if self.size_average:
                return torch.mean(total_norm)
            else:
                return torch.sum(total_norm)

        return total_norm

    def Lp_rel(self, y_pred:torch.tensor, y_true:torch.tensor):
        ''' Relative Error
        Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        '''
        assert y_true.shape==y_pred.shape
        batch_size = y_true.shape[0]
        #
        diff_norms = torch.norm(y_true.reshape(batch_size,-1) - y_pred.reshape(batch_size,-1), self.p, 1)
        y_norms = torch.norm(y_true.reshape(batch_size,-1), self.p, 1) + self.eps
        #
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

class MyLoss(object):

    def __init__(self, size_average=True, reduction=True):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.size_average = size_average
        self.eps = 1e-6

    def mse_org(self, y_pred:torch.tensor, y_true:torch.tensor):
        ''' The mse loss w/o relative 
        Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        '''
        assert y_true.shape==y_pred.shape
        batch_size = y_pred.shape[0]
        #
        diff_norm = torch.norm(y_true.reshape(batch_size,-1) - y_pred.reshape(batch_size,-1), 2, 1)
        #
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norm)
            else:
                return torch.sum(diff_norm)
        
        return diff_norm

    def mse_rel(self, y_pred:torch.tensor, y_true:torch.tensor):
        ''' The mse loss w relative
         Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        '''
        assert y_true.shape==y_pred.shape
        batch_size = y_pred.shape[0]
        #
        diff_norms = torch.norm(y_true.reshape(batch_size,-1) - y_pred.reshape(batch_size,-1), 2, 1)
        y_norms = torch.norm(y_true.reshape(batch_size,-1), 2, 1) + self.eps
        #
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        
        return diff_norms/y_norms