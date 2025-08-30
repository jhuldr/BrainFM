# ported from https://github.com/pvigier/perlin-numpy


import torch
import numpy as np
import matplotlib.pyplot as plt



def center_crop(img, win_size = [220, 220, 220]):
    # center crop
    if len(img.shape) == 4: 
        img = torch.permute(img, (3, 0, 1, 2)) # (move last dim to first)
        img = img[None]
        permuted = True
    else: 
        assert len(img.shape) == 3
        img = img[None, None]
        permuted = False

    orig_shp = img.shape[2:] # (1, d, s, r, c)
    if win_size is None:
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp
    elif orig_shp[0] > win_size[0] or orig_shp[1] > win_size[1] or orig_shp[2] > win_size[2]:
        crop_start = [ max((orig_shp[i] - win_size[i]), 0) // 2 for i in range(3) ]
        crop_img = img[ :, :, crop_start[0] : crop_start[0] + win_size[0], 
                   crop_start[1] : crop_start[1] + win_size[1], 
                   crop_start[2] : crop_start[2] + win_size[2]]
        if permuted:
            return torch.permute(crop_img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return crop_img, crop_start, orig_shp
    else:
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp



def V_plot(Vx, Vy, save_path):
    # Meshgrid 
    X,Y = np.meshgrid(np.arange(0, Vx.shape[0], 1), np.arange(0, Vx.shape[1], 1)) 
    # Assign vector directions 
    Ex = Vx 
    Ey = Vy 

    # Depict illustration 
    plt.figure() 
    plt.streamplot(X,Y,Ex,Ey, density=1.4, linewidth=None, color='orange')  
    plt.axis('off')
    plt.savefig(save_path)
    #plt.show()

def stream_2D(Phi, batched = False, delta_lst = [1., 1.]):
    '''
    input: Phi as a scalar field in 2D grid: (r, c) or (n_batch, r, c)
    output: curl of Phi (divergence-free by definition)
    ''' 
    dD = gradient_c(Phi, batched = batched, delta_lst = delta_lst) 
    Vx = - dD[..., 1]
    Vy = dD[..., 0]
    return Vx, Vy


def stream_3D(Phi_a, Phi_b, Phi_c, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: (batch, s, r, c)
    '''
    device = Phi_a.device
    dDa = gradient_c(Phi_a, batched = batched, delta_lst = delta_lst)
    dDb = gradient_c(Phi_b, batched = batched, delta_lst = delta_lst)
    dDc = gradient_c(Phi_c, batched = batched, delta_lst = delta_lst)
    Va_x, Va_y, Va_z = dDa[..., 0], dDa[..., 1], dDa[..., 2]
    Vb_x, Vb_y, Vb_z = dDb[..., 0], dDb[..., 1], dDb[..., 2]
    Vc_x, Vc_y, Vc_z = dDc[..., 0], dDc[..., 1], dDc[..., 2]
    Vx = Vc_y - Vb_z
    Vy = Va_z - Vc_x
    Vz = Vb_x - Va_y
    return Vx, Vy, Vz



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


