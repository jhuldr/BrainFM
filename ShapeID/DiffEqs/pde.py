# ported from https://github.com/pvigier/perlin-numpy

import math

import numpy as np 

import torch
import torch.nn as nn




def gradient_f(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Upper-boundaries: Backward Difference
    Non-boundaries & Upper-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    device = X.device
    dim = len(X.size()) - 1 if batched else len(X.size())
    #print(batched)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        dX = torch.zeros(X.size(), dtype = torch.float, device = device)
        X = X.permute(1, 0) if batched else X
        dX = dX.permute(1, 0) if batched else dX
        dX[-1] = X[-1] - X[-2] # Backward Difference
        dX[:-1] = X[1:] - X[:-1] # Forward Difference

        dX = dX.permute(1, 0) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = torch.zeros(X.size() + tuple([2]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 0) if batched else X
        dX = dX.permute(1, 2, 3, 0) if batched else dX # put batch to last dim
        dX[-1, :, 0] = X[-1, :] - X[-2, :] # Backward Difference
        dX[:-1, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX = dX.permute(3, 0, 1, 2) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = torch.zeros(X.size() + tuple([3]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 3, 0) if batched else X
        dX = dX.permute(1, 2, 3, 4, 0) if batched else dX
        dX[-1, :, :, 0] = X[-1, :, :] - X[-2, :, :] # Backward Difference
        dX[:-1, :, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, :, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, :, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2] # Backward Difference
        dX[:, :, :-1, 2] = X[:, :, 1:] - X[:, :, :-1] # Forward Difference

        dX = dX.permute(4, 0, 1, 2, 3) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


def gradient_b(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Non-boundaries & Upper-boundaries: Backward Difference
    Lower-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    device = X.device
    dim = len(X.size()) - 1 if batched else len(X.size())
    #print(batched)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        dX = torch.zeros(X.size(), dtype = torch.float, device = device)
        X = X.permute(1, 0) if batched else X
        dX = dX.permute(1, 0) if batched else dX
        dX[1:] = X[1:] - X[:-1] # Backward Difference
        dX[0] = X[1] - X[0] # Forward Difference

        dX = dX.permute(1, 0) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = torch.zeros(X.size() + tuple([2]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 0) if batched else X
        dX = dX.permute(1, 2, 3, 0) if batched else dX # put batch to last dim
        dX[1:, :, 0] = X[1:, :] - X[:-1, :] # Backward Difference
        dX[0, :, 0] = X[1] - X[0] # Forward Difference

        dX[:, 1:, 1] = X[:, 1:] - X[:, :-1] # Backward Difference
        dX[:, 0, 1] = X[:, 1] - X[:, 0] # Forward Difference

        dX = dX.permute(3, 0, 1, 2) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = torch.zeros(X.size() + tuple([3]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 3, 0) if batched else X
        dX = dX.permute(1, 2, 3, 4, 0) if batched else dX
        dX[1:, :, :, 0] = X[1:, :, :] - X[:-1, :, :] # Backward Difference
        dX[0, :, :, 0] = X[1] - X[0] # Forward Difference

        dX[:, 1:, :, 1] = X[:, 1:] - X[:, :-1] # Backward Difference
        dX[:, 0, :, 1] = X[:, 1] - X[:, 0] # Forward Difference

        dX[:, :, 1:, 2] = X[:, :, 1:] - X[:, :, :-1] # Backward Difference
        dX[:, :, 0, 2] = X[:, :, 1] - X[:, :, 0] # Forward Difference

        dX = dX.permute(4, 0, 1, 2, 3) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX
  

def gradient_c(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Non-boundaries: Central Difference
    Upper-boundaries: Backward Difference
    Lower-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    device = X.device
    dim = len(X.size()) - 1 if batched else len(X.size())
    #print(X.size())
    #print(batched)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        dX = torch.zeros(X.size(), dtype = torch.float, device = device)
        X = X.permute(1, 0) if batched else X
        dX = dX.permute(1, 0) if batched else dX
        dX[1:-1] = (X[2:] - X[:-2]) / 2 # Central Difference
        dX[0] = X[1] - X[0] # Forward Difference
        dX[-1] = X[-1] - X[-2] # Backward Difference

        dX = dX.permute(1, 0) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = torch.zeros(X.size() + tuple([2]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 0) if batched else X
        dX = dX.permute(1, 2, 3, 0) if batched else dX # put batch to last dim
        dX[1:-1, :, 0] = (X[2:, :] - X[:-2, :]) / 2
        dX[0, :, 0] = X[1] - X[0]
        dX[-1, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, 1] = (X[:, 2:] - X[:, :-2]) / 2
        dX[:, 0, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, 1] = X[:, -1] - X[:, -2]

        dX = dX.permute(3, 0, 1, 2) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = torch.zeros(X.size() + tuple([3]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 3, 0) if batched else X
        dX = dX.permute(1, 2, 3, 4, 0) if batched else dX
        dX[1:-1, :, :, 0] = (X[2:, :, :] - X[:-2, :, :]) / 2
        dX[0, :, :, 0] = X[1] - X[0]
        dX[-1, :, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, :, 1] = (X[:, 2:, :] - X[:, :-2, :]) / 2
        dX[:, 0, :, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, :, 1] = X[:, -1] - X[:, -2]
        dX[:, :, 1:-1, 2] = (X[:, :, 2:] - X[:, :, :-2]) / 2
        dX[:, :, 0, 2] = X[:, :, 1] - X[:, :, 0]
        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2]

        dX = dX.permute(4, 0, 1, 2, 3) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


def gradient_c_numpy(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a Numpy array "X" in each direction
    Non-boundaries: Central Difference
    Upper-boundaries: Backward Difference
    Lower-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    dim = len(X.shape) - 1 if batched else len(X.shape)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        X = np.transpose(X, (1, 0)) if batched else X
        dX = np.zeros(X.shapee).astype(float)
        dX[1:-1] = (X[2:] - X[:-2]) / 2 # Central Difference
        dX[0] = X[1] - X[0] # Forward Difference
        dX[-1] = X[-1] - X[-2] # Backward Difference

        dX = np.transpose(X, (1, 0)) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = np.zeros(X.shape + tuple([2])).astype(float)
        X = np.transpose(X, (1, 2, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 0)) if batched else dX # put batch to last dim
        dX[1:-1, :, 0] = (X[2:, :] - X[:-2, :]) / 2
        dX[0, :, 0] = X[1] - X[0]
        dX[-1, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, 1] = (X[:, 2:] - X[:, :-2]) / 2
        dX[:, 0, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, 1] = X[:, -1] - X[:, -2]

        dX = np.transpose(dX, (3, 0, 1, 2)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = np.zeros(X.shape + tuple([3])).astype(float)
        X = np.transpose(X, (1, 2, 3, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 4, 0)) if batched else dX # put batch to last dim
        dX[1:-1, :, :, 0] = (X[2:, :, :] - X[:-2, :, :]) / 2
        dX[0, :, :, 0] = X[1] - X[0]
        dX[-1, :, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, :, 1] = (X[:, 2:, :] - X[:, :-2, :]) / 2
        dX[:, 0, :, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, :, 1] = X[:, -1] - X[:, -2]
        dX[:, :, 1:-1, 2] = (X[:, :, 2:] - X[:, :, :-2]) / 2
        dX[:, :, 0, 2] = X[:, :, 1] - X[:, :, 0]
        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2]

        dX = np.transpose(dX, (4, 0, 1, 2, 3)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


def gradient_f_numpy(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Upper-boundaries: Backward Difference
    Non-boundaries & Upper-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    dim = len(X.shape) - 1 if batched else len(X.shape)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        X = np.transpose(X, (1, 0)) if batched else X
        dX = np.zeros(X.shapee).astype(float)
        dX[-1] = X[-1] - X[-2] # Backward Difference
        dX[:-1] = X[1:] - X[:-1] # Forward Difference

        dX = np.transpose(X, (1, 0)) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = np.zeros(X.shape + tuple([2])).astype(float)
        X = np.transpose(X, (1, 2, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 0)) if batched else dX # put batch to last dim
        dX[-1, :, 0] = X[-1, :] - X[-2, :] # Backward Difference
        dX[:-1, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX = np.transpose(dX, (3, 0, 1, 2)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = np.zeros(X.shape + tuple([3])).astype(float)
        X = np.transpose(X, (1, 2, 3, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 4, 0)) if batched else dX # put batch to last dim
        dX[-1, :, :, 0] = X[-1, :, :] - X[-2, :, :] # Backward Difference
        dX[:-1, :, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, :, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, :, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2] # Backward Difference
        dX[:, :, :-1, 2] = X[:, :, 1:] - X[:, :, :-1] # Forward Difference

        dX = np.transpose(dX, (4, 0, 1, 2, 3)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


class Upwind(object):
    '''
    Backward if > 0, forward if <= 0
    '''
    def __init__(self, U, data_spacing = [1., 1, 1.], batched = True):
        self.U = U # (s, r, c)
        self.batched = batched
        self.data_spacing = data_spacing
        self.dim = len(self.U.size()) - 1 if batched else len(self.U.size())
        self.I = torch.ones(self.U.size(), dtype = torch.float, device = U.device)

    def dX(self, FGx):
        dXf = gradient_f(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 0]
        dXb = gradient_b(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 0]
        Xflag = (FGx > 0).float()
        return dXf * (self.I - Xflag) + dXb * Xflag

    def dY(self, FGy):
        dYf = gradient_f(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 1]
        dYb = gradient_b(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 1]
        Yflag = (FGy > 0).float()
        return dYf * (self.I - Yflag) + dYb * Yflag

    def dZ(self, FGz):
        dZf = gradient_f(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 2]
        dZb = gradient_b(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 2]
        Zflag = (FGz > 0).float()
        return dZf * (self.I - Zflag) + dZb * Zflag
    
    
class AdvDiffPartial(nn.Module):
    def __init__(self, data_spacing, device):
        super(AdvDiffPartial, self).__init__()
        self.dimension = len(data_spacing)  # (slc, row, col)
        self.device = device
        self.data_spacing = data_spacing

    @property
    def Grad_Ds(self):
        return {
            'constant': self.Grad_constantD,
            'scalar': self.Grad_scalarD,
            'diag': self.Grad_diagD,
            'full': self.Grad_fullD,
            'full_dual': self.Grad_fullD,
            'full_spectral':self.Grad_fullD,
            'full_cholesky': self.Grad_fullD,
            'full_symmetric': self.Grad_fullD
        }
    @property
    def Grad_Vs(self):
        return {
            'constant': self.Grad_constantV,
            'scalar': self.Grad_scalarV,
            'vector': self.Grad_vectorV, # For general V w/o div-free TODO self.Grad_vectorV
            'vector_div_free': self.Grad_div_free_vectorV,
            'vector_div_free_clebsch': self.Grad_div_free_vectorV,
            'vector_div_free_stream': self.Grad_div_free_vectorV,
            'vector_div_free_stream_gauge': self.Grad_div_free_vectorV, 
        }

    def Grad_constantD(self, C, Dlst):
        if self.dimension == 1:
            return Dlst['D'] * (self.ddXc(C))
        elif self.dimension == 2:
            return Dlst['D'] * (self.ddXc(C) + self.ddYc(C))
        elif self.dimension == 3:
            return Dlst['D'] * (self.ddXc(C) + self.ddYc(C) + self.ddZc(C))

    def Grad_constant_tensorD(self, C, Dlst):
        if self.dimension == 1:
            raise NotImplementedError
        elif self.dimension == 2:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                Dlst['Dxy'] * self.dXb(dC_f[..., 1]) + Dlst['Dxy'] * self.dYb(dC_f[..., 0]) +\
                        Dlst['Dyy'] * self.dYb(dC_f[..., 1])  
        elif self.dimension == 3:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return Dlst['Dxx'] * self.dXb(dC_f[..., 0]) + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) + Dlst['Dzz'] * self.dZb(dC_f[..., 2]) + \
                            Dlst['Dxy'] * (self.dXb(dC_f[..., 1]) + self.dYb(dC_f[..., 0])) + \
                                Dlst['Dyz'] * (self.dYb(dC_f[..., 2]) + self.dZb(dC_f[..., 1])) + \
                                    Dlst['Dxz'] * (self.dZb(dC_f[..., 0]) + self.dXb(dC_f[..., 2]))
        
    def Grad_scalarD(self, C, Dlst): # batch_C: (batch_size, (slc), row, col)
        # Expanded version: \nabla (D \nabla C) => \nabla D \cdot \nabla C (part (a)) + D \Delta C (part (b)) # 
        # NOTE: Work better than Central Differences !!! # 
        # Nested Forward-Backward Difference Scheme in part (b)#
        if self.dimension == 1:
            dC = gradient_c(C, batched = True, delta_lst = self.data_spacing)
            return gradient_c(Dlst['D'], batched = True, delta_lst = self.data_spacing) * dC + \
                Dlst['D'] * gradient_c(dC, batched = True, delta_lst = self.data_spacing)
        else: # (dimension = 2 or 3)
            dC_c = gradient_c(C, batched = True, delta_lst = self.data_spacing)
            dC_f = gradient_f(C, batched = True, delta_lst = self.data_spacing)
            dD_c = gradient_c(Dlst['D'], batched = True, delta_lst = self.data_spacing)
            out = (dD_c * dC_c).sum(-1)
            for dim in range(dC_f.size(-1)):
                out += Dlst['D'] * gradient_b(dC_f[..., dim], batched = True, delta_lst = self.data_spacing)[..., dim]
            return out

    def Grad_diagD(self, C, Dlst):
        # Expanded version #
        if self.dimension == 1:
            raise NotImplementedError('diag_D is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return self.dXc(Dlst['Dxx']) * dC_c[..., 0] + Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                self.dYc(Dlst['Dyy']) * dC_c[..., 1] + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) 
        elif self.dimension == 3:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return self.dXc(Dlst['Dxx']) * dC_c[..., 0] + Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                self.dYc(Dlst['Dyy']) * dC_c[..., 1] + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) +\
                    self.dZc(Dlst['Dzz']) * dC_c[..., 2] + Dlst['Dzz'] * self.dZb(dC_f[..., 2])

    def Grad_fullD(self, C, Dlst):
        # Expanded version #
        '''https://github.com/uncbiag/PIANOinD/blob/master/Doc/PIANOinD.pdf'''
        if self.dimension == 1:
            raise NotImplementedError('full_D is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return self.dXc(Dlst['Dxx']) * dC_c[..., 0] + Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                self.dXc(Dlst['Dxy']) * dC_c[..., 1] + Dlst['Dxy'] * self.dXb(dC_f[..., 1]) +\
                    self.dYc(Dlst['Dxy']) * dC_c[..., 0] + Dlst['Dxy'] * self.dYb(dC_f[..., 0]) +\
                        self.dYc(Dlst['Dyy']) * dC_c[..., 1] + Dlst['Dyy'] * self.dYb(dC_f[..., 1])  
        elif self.dimension == 3:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return (self.dXc(Dlst['Dxx']) + self.dYc(Dlst['Dxy']) + self.dZc(Dlst['Dxz'])) * dC_c[..., 0] + \
                (self.dXc(Dlst['Dxy']) + self.dYc(Dlst['Dyy']) + self.dZc(Dlst['Dyz'])) * dC_c[..., 1] + \
                    (self.dXc(Dlst['Dxz']) + self.dYc(Dlst['Dyz']) + self.dZc(Dlst['Dzz'])) * dC_c[..., 2] + \
                        Dlst['Dxx'] * self.dXb(dC_f[..., 0]) + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) + Dlst['Dzz'] * self.dZb(dC_f[..., 2]) + \
                            Dlst['Dxy'] * (self.dXb(dC_f[..., 1]) + self.dYb(dC_f[..., 0])) + \
                                Dlst['Dyz'] * (self.dYb(dC_f[..., 2]) + self.dZb(dC_f[..., 1])) + \
                                    Dlst['Dxz'] * (self.dZb(dC_f[..., 0]) + self.dXb(dC_f[..., 2]))

    def Grad_constantV(self, C, Vlst):
        if len(Vlst['V'].size()) == 1:
            if self.dimension == 1:
                return - Vlst['V'] * self.dXb(C) if Vlst['V'] > 0 else - Vlst['V'] * self.dXf(C)
            elif self.dimension == 2:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C)) if Vlst['V'] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C))
            elif self.dimension == 3:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C) + self.dZb(C)) if Vlst['V'] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C) + self.dZf(C))
        else:
            if self.dimension == 1:
                return - Vlst['V'] * self.dXb(C) if Vlst['V'][0, 0] > 0 else - Vlst['V'] * self.dXf(C)
            elif self.dimension == 2:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C)) if Vlst['V'][0, 0, 0] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C))
            elif self.dimension == 3:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C) + self.dZb(C)) if Vlst['V'][0, 0, 0, 0] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C) + self.dZf(C))
    
    def Grad_constant_vectorV(self, C, Vlst):
        if self.dimension == 1:
            raise NotImplementedError
        elif self.dimension == 2:
            out_x = - Vlst['Vx'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vx'][0, 0, 0] > 0 else - Vlst['Vx'] * (self.dXf(C) + self.dYf(C))
            out_y = - Vlst['Vy'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vy'][0, 0, 0] > 0 else - Vlst['Vy'] * (self.dXf(C) + self.dYf(C))
            return out_x + out_y
        elif self.dimension == 3:
            out_x = - Vlst['Vx'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vx'][0, 0, 0] > 0 else - Vlst['Vx'] * (self.dXf(C) + self.dYf(C))
            out_y = - Vlst['Vy'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vy'][0, 0, 0] > 0 else - Vlst['Vy'] * (self.dXf(C) + self.dYf(C))
            out_z = - Vlst['Vz'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vz'][0, 0, 0] > 0 else - Vlst['Vz'] * (self.dXf(C) + self.dYf(C))
            return out_x + out_y + out_z
    
    def Grad_SimscalarV(self, C, Vlst):
        V = Vlst['V']
        Upwind_C = Upwind(C, self.data_spacing)
        if self.dimension == 1:
            C_x = Upwind_C.dX(V)
            return - V * C_x
        if self.dimension == 2:
            C_x, C_y = Upwind_C.dX(V), Upwind_C.dY(V)
            return - V * (C_x + C_y)
        if self.dimension == 3:
            C_x, C_y, C_z = Upwind_C.dX(V), Upwind_C.dY(V), Upwind_C.dZ(V)
            return - V * (C_x + C_y + C_z)

    def Grad_scalarV(self, C, Vlst):
        V = Vlst['V']
        Upwind_C = Upwind(C, self.data_spacing)
        dV = gradient_c(V, batched = True, delta_lst = self.data_spacing)
        if self.dimension == 1:
            C_x = Upwind_C.dX(V)
            return - V * C_x - C * dV
        elif self.dimension == 2:
            C_x, C_y = Upwind_C.dX(V), Upwind_C.dY(V)
            return - V * (C_x + C_y) - C * dV.sum(-1)
        elif self.dimension == 3:
            C_x, C_y, C_z = Upwind_C.dX(V), Upwind_C.dY(V), Upwind_C.dZ(V)
            return - V * (C_x + C_y + C_z) - C * dV.sum(-1)

    def Grad_div_free_vectorV(self, C, Vlst):
        ''' For divergence-free-by-definition velocity'''
        if self.dimension == 1:
            raise NotImplementedError('clebschVector is not supported for 1D version of velocity')
        Upwind_C = Upwind(C, self.data_spacing) 
        C_x, C_y = Upwind_C.dX(Vlst['Vx']), Upwind_C.dY(Vlst['Vy'])
        if self.dimension == 2:
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y)
        elif self.dimension == 3:
            C_z = Upwind_C.dZ(Vlst['Vz'])
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y + Vlst['Vz'] * C_z)
            
    def Grad_vectorV(self, C, Vlst):
        ''' For general velocity'''
        if self.dimension == 1:
            raise NotImplementedError('vector is not supported for 1D version of velocity')
        Upwind_C = Upwind(C, self.data_spacing) 
        C_x, C_y = Upwind_C.dX(Vlst['Vx']), Upwind_C.dY(Vlst['Vy'])
        Vx_x = self.dXc(Vlst['Vx'])
        Vy_y = self.dYc(Vlst['Vy']) 
        if self.dimension == 2:
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y) - C * (Vx_x + Vy_y)
        if self.dimension == 3:
            C_z = Upwind_C.dZ(Vlst['Vz'])
            Vz_z = self.dZc(Vlst['Vz'])
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y + Vlst['Vz'] * C_z) - C * (Vx_x + Vy_y + Vz_z)
    
    ################# Utilities #################
    def db(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)
    def df(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)
    def dc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)
    def dXb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 0]
    def dXf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0]
    def dXc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 0]
    def dYb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 1]
    def dYf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1]
    def dYc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 1]
    def dZb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 2]
    def dZf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2]
    def dZc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 2]
    def ddXc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0], 
        batched = True, delta_lst = self.data_spacing)[..., 0]
    def ddYc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1], 
        batched = True, delta_lst = self.data_spacing)[..., 1]
    def ddZc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2], 
        batched = True, delta_lst = self.data_spacing)[..., 2]
    


class AdvDiffPDE(nn.Module):
    '''
    Plain advection-diffusion PDE solver for pre-set V_lst and D_lst (1D, 2D, 3D) for forward time series simulation
    '''
    def __init__(self, data_spacing, perf_pattern, D_type='scalar', V_type='vector', BC=None, dt=0.1, V_dict={}, D_dict={}, stochastic=False, device='cpu'):
        super(AdvDiffPDE, self).__init__() 
        self.BC = BC
        self.dt = dt
        self.dimension = len(data_spacing)
        self.perf_pattern = perf_pattern
        self.partials = AdvDiffPartial(data_spacing, device)
        self.D_type, self.V_type = D_type, V_type
        self.stochastic = stochastic 
        self.V_dict, self.D_dict = V_dict, D_dict
        self.Sigma, self.Sigma_V, self.Sigma_D = 0., 0., 0. # Only for initialization # 
        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise ValueError('Unsupported dimension: %d' % self.dimension)
                 
    @property
    def set_BC(self):
    # NOTE For bondary condition of mass concentration #
        '''X: (n_batch, spatial_shape)'''
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = 1
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X 
        
    def forward(self, t, batch_C):
        '''
        t: (batch_size,)
        batch_C: (batch_size, (slc,) row, col)
        ''' 
        batch_size  = batch_C.size(0)
        batch_C = self.set_BC(batch_C)
        if 'diff' not in self.perf_pattern:
            out = self.partials.Grad_Vs[self.V_type](batch_C, self.V_dict) 
            if self.stochastic:  
                out = out + self.Sigma * math.sqrt(self.dt) * torch.randn_like(batch_C).to(batch_C)
        elif 'adv' not in self.perf_pattern:
            out = self.partials.Grad_Ds[self.D_type](batch_C, self.D_dict)
            if self.stochastic:  
                out = out + self.Sigma * math.sqrt(self.dt) * torch.randn_like(batch_C).to(batch_C)
        else:
            if self.stochastic:  
                out_D = self.partials.Grad_Ds[self.D_type](batch_C, self.D_dict)
                out_V = self.partials.Grad_Vs[self.V_type](batch_C, self.V_dict) 
                out = out_D + out_V + self.Sigma * math.sqrt(self.dt) * torch.randn_like(batch_C).to(batch_C) 
            else:
                out_V = self.partials.Grad_Vs[self.V_type](batch_C, self.V_dict)  
                out_D = self.partials.Grad_Ds[self.D_type](batch_C, self.D_dict)
                out = out_V + out_D
        return out

        

