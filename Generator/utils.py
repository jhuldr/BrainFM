import os
import numpy as np
import nibabel as nib

import torch
from torch.nn.functional import conv3d
from torch.utils.data import Dataset

from scipy.io.matlab import loadmat


import time, datetime

from ShapeID.DiffEqs.adjoint import odeint_adjoint as odeint
from ShapeID.perlin3d import generate_velocity_3d , generate_shape_3d


class ConcatDataset(Dataset):
    def __init__(self,dataset_list, probs=None):
        self.datasets = dataset_list
        self.probs = probs if probs else [1/len(self.datasets)] * len(self.datasets)

    def __getitem__(self, i):
        chosen_dataset = np.random.choice(self.datasets, 1, p=self.probs)[0]
        i = i % len(chosen_dataset)
        return chosen_dataset[i]

    def __len__(self):
        return  max(len(d) for d in self.datasets)



# Prepare generator
def resolution_sampler(low_res_only = False):
    
    if low_res_only:
        r = (np.random.rand() * 0.5) + 0.5 # in [0.5, 1]
    else:
        r = np.random.rand() # in [0, 1]

    if r < 0.25: # 1mm isotropic
        resolution = np.array([1.0, 1.0, 1.0])
        thickness = np.array([1.0, 1.0, 1.0])
    elif r < 0.5: # clinical (low-res in one dimension)
        resolution = np.array([1.0, 1.0, 1.0])
        thickness = np.array([1.0, 1.0, 1.0])
        idx = np.random.randint(3)
        resolution[idx] = 2.5 + 6 * np.random.rand()
        thickness[idx] = np.min([resolution[idx], 4.0 + 2.0 * np.random.rand()])
    elif r < 0.75:  # low-field: stock sequences (always axial)
        resolution = np.array([1.3, 1.3, 4.8]) + 0.4 * np.random.rand(3)
        thickness = resolution.copy()
    else: # low-field: isotropic-ish (also good for scouts)
        resolution = 2.0 + 3.0 * np.random.rand(3)
        thickness = resolution.copy()

    return resolution, thickness


#####################################
############ Utility Func ###########
#####################################


def binarize(p, thres):
    # TODO: what is the optimal thresholding strategy?
    thres = thres * p.max()

    bin = p.clone()
    bin[p < thres] = 0.
    bin[p >= thres] = 1. 
    return bin

def make_gaussian_kernel(sigma, device):

    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2*sl+1, dtype=torch.float, device=device)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel

def gaussian_blur_3d(input, stds, device):
    blurred = input[None, None, :, :, :]
    if stds[0]>0:
        kx = make_gaussian_kernel(stds[0], device=device)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1]>0:
        ky = make_gaussian_kernel(stds[1], device=device)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2]>0:
        kz = make_gaussian_kernel(stds[2], device=device)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return torch.squeeze(blurred)



#####################################
######### Deformation Func ##########
#####################################

def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A


def fast_3D_interp_torch(X, II, JJ, KK, mode='linear', default_value_linear=0.0):

    if II is None: 
        return X
 
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None] 
        Y = X[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = (IIv - fx)[..., None]
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = (JJv - fy)[..., None]
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = (KKv - fz)[..., None]
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None] 
        
        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        Y = torch.zeros([*II.shape, X.shape[3]], device=X.device) 
        Y[ok] = c.float()
        Y[~ok] = default_value_linear   

        if Y.shape[-1]==1:
            Y = Y[...,0] 
    else:
        raise Exception('mode must be linear or nearest')
    
    return Y



def myzoom_torch(X, factor, aff=None):

    if len(X.shape)==3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], dtype=torch.float, device=X.device)[:newsize[0]]
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], dtype=torch.float, device=X.device)[:newsize[1]]
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], dtype=torch.float, device=X.device)[:newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = (vx - fx) 
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = (vy - fy) 
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = (vz - fz) 
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], dtype=torch.float, device=X.device) 

    tmp1 = torch.zeros([newsize[0], X.shape[1], X.shape[2], X.shape[3]], dtype=torch.float, device=X.device)
    for i in range(newsize[0]):
        tmp1[i, :, :] = wfx[i] * X[fx[i], :, :] +  wcx[i] * X[cx[i], :, :]
    tmp2 = torch.zeros([newsize[0], newsize[1], X.shape[2], X.shape[3]], dtype=torch.float, device=X.device)
    for j in range(newsize[1]):
        tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
    for k in range(newsize[2]):
        Y[:, :, k] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    if aff is not None:
        aff_new = aff.copy() 
        aff_new[:-1] = aff_new[:-1] / factor
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (0.5 - 0.5 / (factor * np.ones(3)))
        return Y, aff_new
    else:
        return Y
    



#####################################
############ Reading Func ###########
#####################################

def read_image(file_name):
    img = nib.load(file_name) 
    aff = img.affine
    res = np.sqrt(np.sum(abs(aff[:-1, :-1]), axis=0)) 
    return img, aff, res

def deform_image(I, deform_dict, device, default_value_linear_mode=None, deform_mode = 'linear'):
    if I is None:
        return I
    
    [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

    if not isinstance(I, torch.Tensor):
        I = torch.squeeze(torch.tensor(I.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=device))
    else:
        I = torch.squeeze(I[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=device)
    I = torch.nan_to_num(I) 
    
    if default_value_linear_mode is not None:
        if default_value_linear_mode == 'max':
            default_value_linear = torch.max(I)
        else:
            raise ValueError('Not support default_value_linear_mode:', default_value_linear_mode)
    else:
        default_value_linear = 0.
    Idef = fast_3D_interp_torch(I, xx2, yy2, zz2, deform_mode, default_value_linear) 
    
    return Idef


def read_and_deform(file_name, dtype, deform_dict, device, mask, default_value_linear_mode=None, deform_mode = 'linear', mean = 0., scale = 1.):
    [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

    try:
        Iimg = nib.load(file_name)  
    except:
        Iimg = nib.load(file_name + '.gz')
    res = np.sqrt(np.sum(abs(Iimg.affine[:-1, :-1]), axis=0))
    I = torch.squeeze(torch.tensor(Iimg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=dtype, device=device))
    I = torch.nan_to_num(I) 

    I -= mean
    I /= scale

    if mask is not None:
        I[mask == 0] = 0

    if default_value_linear_mode is not None:
        if default_value_linear_mode == 'max':
            default_value_linear = torch.max(I)
        else:
            raise ValueError('Not support default_value_linear_mode:', default_value_linear_mode)
    else:
        default_value_linear = 0.
    Idef = fast_3D_interp_torch(I, xx2, yy2, zz2, deform_mode, default_value_linear) 
    return Idef, res


def read_and_deform_image(exist_keys, task_name, file_name, setups, deform_dict, device, mask, **kwargs):
    Idef, _ = read_and_deform(file_name, torch.float, deform_dict, device, mask) 
    Idef -= torch.min(Idef)
    Idef /= torch.max(Idef)
    if setups['flip']: 
        Idef = torch.flip(Idef, [0]) 
    update_dict = {task_name: Idef[None]}
    
    if os.path.isfile(file_name[:-4] + '.defacingmask.nii'):
        Idef_DM, _ = read_and_deform(file_name[:-4] + '.defacingmask.nii', torch.float, deform_dict, device, mask) 
        Idef_DM = torch.clamp(Idef_DM, min = 0.) 
        Idef_DM /= torch.max(Idef_DM) 
        if setups['flip']: 
            Idef = torch.flip(Idef_DM, [0]) 
        update_dict.update({task_name + '_DM': Idef_DM[None]}) 
    #if not 'brain_mask' in exist_keys:
    #    mask = torch.ones_like(Idef)
    #    mask[Idef <= 0.] = 0.  
    #    update_dict.update({'brain_mask': mask[None]}) 
    return update_dict

def read_and_deform_CT(exist_keys, task_name, file_name, setups, deform_dict, device, mask, **kwargs):
    Idef, _ = read_and_deform(file_name, torch.float, deform_dict, device, mask, scale = 1000)
    #Idef = torch.clamp(Idef, min = 0., max = 80.) # No clamping for inference/GT
    #Idef /= torch.max(Idef)
    if setups['flip']: 
        Idef = torch.flip(Idef, [0]) 
    update_dict = {'CT': Idef[None]}
    
    if os.path.isfile(file_name[:-4] + '.defacingmask.nii'):
        Idef_DM, _ = read_and_deform(file_name[:-4] + '.defacingmask.nii', torch.float, deform_dict, device, mask) 
        Idef_DM = torch.clamp(Idef_DM, min = 0.) 
        Idef_DM /= torch.max(Idef_DM) 
        if setups['flip']: 
            Idef = torch.flip(Idef_DM, [0]) 
        update_dict.update({task_name + '_DM': Idef_DM[None]}) 
    #if not 'brain_mask' in exist_keys:
    #    mask = torch.ones_like(Idef)
    #    mask[Idef <= 0.] = 0. 
    #    update_dict.update({'brain_mask': mask[None]}) 
    return update_dict

def read_and_deform_distance(exist_keys, task_name, file_names, setups, deform_dict, device, mask, cfg, **kwargs):
    [lp_dist_map, lw_dist_map, rp_dist_map, rw_dist_map] = file_names
    
    
    lp, _ = read_and_deform(lp_dist_map, torch.float, deform_dict, device, mask, default_value_linear_mode = 'max', mean = 128., scale = 20) 
    lw, _ = read_and_deform(lw_dist_map, torch.float, deform_dict, device, mask, default_value_linear_mode = 'max', mean = 128., scale = 20) 

    if mask is not None: # left_hemis_only
        Idef = torch.stack([lp, lw], dim = 0)  
    else: 
        rp, _ = read_and_deform(rp_dist_map, torch.float, deform_dict, device, mask, default_value_linear_mode = 'max', mean = 128., scale = 20) 
        rw, _ = read_and_deform(rw_dist_map, torch.float, deform_dict, device, mask, default_value_linear_mode = 'max', mean = 128., scale = 20) 

        if setups['flip']: 
            aux = torch.flip(lp, [0])
            lp = torch.flip(rp, [0])
            rp = aux
            aux = torch.flip(lw, [0])
            lw = torch.flip(rw, [0])
            rw = aux

        Idef = torch.stack([lp, lw, rp, rw], dim = 0)

    Idef /= deform_dict['scaling_factor_distances']
    Idef = torch.clamp(Idef, min=-cfg.max_surf_distance, max=cfg.max_surf_distance)
    
    return {'distance':  Idef} 
    
def read_and_deform_segmentation(exist_keys, task_name, file_name, setups, deform_dict, device, mask, cfg, onehotmatrix, lut, vflip, **kwargs):
    [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

    Simg = nib.load(file_name)
    S = torch.squeeze(torch.tensor(Simg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(int), dtype=torch.int, device=device))

    if mask is not None:
        S[mask == 0] = 0

    Sdef = fast_3D_interp_torch(S,  xx2, yy2, zz2, 'nearest')
    if cfg.generator.deform_one_hots:
        Sonehot = onehotmatrix[lut[S.long()]]
        Sdef_OneHot = fast_3D_interp_torch(Sonehot, xx2, yy2, zz2)
    else:
        Sdef_OneHot = onehotmatrix[lut[Sdef.long()]]
        
    if setups['flip']: 
        #Sdef = torch.flip(Sdef, [0])   
        Sdef_OneHot = torch.flip(Sdef_OneHot, [0])[:, :, :, vflip]

    # prepare for input
    Sdef_OneHot = Sdef_OneHot.permute([3, 0, 1, 2])

    #update_dict = {'label': Sdef[None], 'segmentation': Sdef_OneHot}
    update_dict = {'segmentation': Sdef_OneHot}

    #if not 'brain_mask' in exist_keys:
    #    mask = torch.ones_like(Sdef)
    #    mask[Sdef <= 0.] = 0. 
    #    update_dict.update({'brain_mask': mask[None]}) 
    return update_dict



def read_and_deform_pathology(exist_keys, task_name, file_name, setups, deform_dict, device, mask = None, 
                              augment = False, pde_func = None, t = None, 
                              shape_gen_args = None, thres = 0., **kwargs):
    # NOTE does not support left_hemis for now

    [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

    if file_name is None:
        return {'pathology': torch.zeros(xx2.shape)[None].to(device), 'pathology_prob': torch.zeros(xx2.shape)[None].to(device)}
    
    if file_name == 'random_shape': # generate random shape
        percentile = np.random.uniform(shape_gen_args.mask_percentile_min, shape_gen_args.mask_percentile_max)
        _, Pdef = generate_shape_3d(xx2.shape, shape_gen_args.perlin_res, percentile, device)   
    else: # read from existing shape
        Pdef, _ = read_and_deform(file_name, torch.float, deform_dict, device)  

    if augment:
        Pdef = augment_pathology(Pdef, pde_func, t, shape_gen_args, device) 

    #if setups['flip']: # flipping should happen after P has been encoded
    #    Pdef = torch.flip(Pdef, [0]) 

    P = binarize(Pdef, thres)
    if P.mean() <= shape_gen_args.pathol_tol:
        return {'pathology': torch.zeros(xx2.shape)[None].to(device), 'pathology_prob': torch.zeros(xx2.shape)[None].to(device)}
    #print('process', P.mean(), shape_gen_args.pathol_tol)

    return {'pathology': P[None], 'pathology_prob': Pdef[None]}


def read_and_deform_registration(exist_keys, task_name, file_names, setups, deform_dict, device, mask, **kwargs):
    [mni_reg_x, mni_reg_y, mni_reg_z] = file_names
    regx, _ = read_and_deform(mni_reg_x, torch.float, deform_dict, device, mask, scale = 10000) 
    regy, _ = read_and_deform(mni_reg_y, torch.float, deform_dict, device, mask, scale = 10000) 
    regz, _ = read_and_deform(mni_reg_z, torch.float, deform_dict, device, mask, scale = 10000)  

    if setups['flip']: 
        regx = -torch.flip(regx, [0]) # NOTE: careful with switching sign
        regy = torch.flip(regy, [0])
        regz = torch.flip(regz, [0])

    Idef = torch.stack([regx, regy, regz], dim = 0) 
    
    return {'registration':  Idef}

def read_and_deform_bias_field(exist_keys, task_name, file_name, setups, deform_dict, device, mask, **kwargs):
    Idef, _ = read_and_deform(file_name, torch.float, deform_dict, mask, device)
    if setups['flip']: 
        Idef = torch.flip(Idef, [0]) 
    return {'bias_field': Idef[None]}

def read_and_deform_surface(exist_keys, task_name, file_name, setups, deform_dict, device, mask, size):
    Fneg, A, c2 = deform_dict['Fneg'], deform_dict['A'], deform_dict['c2']
    # NOTE does not support left_hemis for now

    mat = loadmat(file_name.split('.nii')[0] + '.mat')

    Vlw = torch.tensor(mat['Vlw'], dtype=torch.float, device=device)
    Flw = torch.tensor(mat['Flw'], dtype=torch.int, device=device)
    Vrw = torch.tensor(mat['Vrw'], dtype=torch.float, device=device)
    Frw = torch.tensor(mat['Frw'], dtype=torch.int, device=device)
    Vlp = torch.tensor(mat['Vlp'], dtype=torch.float, device=device)
    Flp = torch.tensor(mat['Flp'], dtype=torch.int, device=device)
    Vrp = torch.tensor(mat['Vrp'], dtype=torch.float, device=device)
    Frp = torch.tensor(mat['Frp'], dtype=torch.int, device=device)

    Ainv = torch.inverse(A)
    Vlw -= c2[None, :]
    Vlw = Vlw @ torch.transpose(Ainv, 0, 1)
    Vlw += fast_3D_interp_torch(Fneg, Vlw[:, 0] + c2[0], Vlw[:, 1]+c2[1], Vlw[:, 2] + c2[2])
    Vlw += c2[None, :]
    Vrw -= c2[None, :]
    Vrw = Vrw @ torch.transpose(Ainv, 0, 1)
    Vrw += fast_3D_interp_torch(Fneg, Vrw[:, 0] + c2[0], Vrw[:, 1]+c2[1], Vrw[:, 2] + c2[2])
    Vrw += c2[None, :]
    Vlp -= c2[None, :]
    Vlp = Vlp @ torch.transpose(Ainv, 0, 1)
    Vlp += fast_3D_interp_torch(Fneg, Vlp[:, 0] + c2[0], Vlp[:, 1] + c2[1], Vlp[:, 2] + c2[2])
    Vlp += c2[None, :]
    Vrp -= c2[None, :]
    Vrp = Vrp @ torch.transpose(Ainv, 0, 1)
    Vrp += fast_3D_interp_torch(Fneg, Vrp[:, 0] + c2[0], Vrp[:, 1] + c2[1], Vrp[:, 2] + c2[2])
    Vrp += c2[None, :]

    if setups['flip']: 
        Vlw[:, 0] = size[0] - 1 - Vlw[:, 0]
        Vrw[:, 0] = size[0] - 1 - Vrw[:, 0]
        Vlp[:, 0] = size[0] - 1 - Vlp[:, 0]
        Vrp[:, 0] = size[0] - 1 - Vrp[:, 0]
        Vlw, Vrw = Vrw, Vlw
        Vlp, Vrp = Vrp, Vlp
        Flw, Frw = Frw, Flw
        Flp, Frp = Frp, Flp

    print(Vlw.shape) # 131148
    print(Vlp.shape) # 131148

    print(Vrw.shape) # 131720
    print(Vrp.shape) # 131720

    print(Flw.shape) # 262292
    print(Flp.shape) # 262292

    print(Frw.shape) # 263436
    print(Frp.shape) # 263436
    #return torch.stack([Vlw, Flw, Vrw, Frw, Vlp, Flp, Vrp, Frp])
    return {'Vlw': Vlw, 'Flw': Flw, 'Vrw': Vrw, 'Frw': Frw, 'Vlp': Vlp, 'Flp': Flp, 'Vrp': Vrp, 'Frp': Frp}
    

#####################################
#########  Pathology Shape  #########
#####################################


def augment_pathology(Pprob, pde_func, t, shape_gen_args, device):
    Pprob = torch.squeeze(Pprob) 

    nt = np.random.randint(1, shape_gen_args.max_nt+1)
    if nt <= 1:  
        return Pprob

    pde_func.V_dict = generate_velocity_3d(Pprob.shape, shape_gen_args.perlin_res, shape_gen_args.V_multiplier, device)

    #start_time = time.time()
    Pprob = odeint(pde_func, Pprob[None], t[:nt], 
                    shape_gen_args.dt, 
                    method = shape_gen_args.integ_method)[-1, 0] # (last_t, n_batch=1, s, r, c)
    # total_time = time.time() - start_time
    #total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #print('Time {} for {} time points'.format(total_time_str, nt)) 

    
    return Pprob


#####################################
######### Augmentation Func #########
#####################################
    

def add_gamma_transform(I, aux_dict, cfg, device, **kwargs):
    gamma = torch.tensor(np.exp(cfg.gamma_std * np.random.randn(1)[0]), dtype=float, device=device)
    I_gamma = 300.0 * (I / 300.0) ** gamma
    #aux_dict.update({'gamma': gamma}) # uncomment if you want to save gamma for later use
    return I_gamma, aux_dict

def add_bias_field(I, aux_dict, cfg, input_mode, setups, size, device, **kwargs):
    if input_mode == 'CT':
        aux_dict.update({'high_res': I})
        return I, aux_dict
    
    bf_scale = cfg.bf_scale_min + np.random.rand(1) * (cfg.bf_scale_max - cfg.bf_scale_min)
    size_BF_small = np.round(bf_scale * np.array(size)).astype(int).tolist()
    if setups['photo_mode']:
        size_BF_small[1] = np.round(size[1]/setups['spac']).astype(int)
    BFsmall = torch.tensor(cfg.bf_std_min + (cfg.bf_std_max - cfg.bf_std_min) * np.random.rand(1), dtype=torch.float, device=device) * \
        torch.randn(size_BF_small, dtype=torch.float, device=device)
    BFlog = myzoom_torch(BFsmall, np.array(size) / size_BF_small)
    BF = torch.exp(BFlog)
    I_bf = I * BF
    aux_dict.update({'BFlog': BFlog, 'high_res': I_bf})
    return I_bf, aux_dict

def resample_resolution(I, aux_dict, setups, res, size, device, **kwargs):
    stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) /np.pi * setups['thickness'] / res
    stds[setups['thickness']<=res] = 0.0 # no blur if thickness is equal to the resolution of the training data
    I_blur = gaussian_blur_3d(I, stds, device)
    new_size = (np.array(size) * res / setups['resolution']).astype(int)

    factors = np.array(new_size) / np.array(size)
    delta = (1.0 - factors) / (2.0 * factors)
    vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
    vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
    vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
    II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
    II = torch.tensor(II, dtype=torch.float, device=device)
    JJ = torch.tensor(JJ, dtype=torch.float, device=device)
    KK = torch.tensor(KK, dtype=torch.float, device=device)

    I_small = fast_3D_interp_torch(I_blur, II, JJ, KK) 
    aux_dict.update({'factors': factors})
    return I_small, aux_dict


def resample_resolution_photo(I, aux_dict, setups, res, size, device, **kwargs):
    stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) /np.pi * setups['thickness'] / res
    stds[setups['thickness']<=res] = 0.0 # no blur if thickness is equal to the resolution of the training data
    I_blur = gaussian_blur_3d(I, stds, device)
    new_size = (np.array(size) * res / setups['resolution']).astype(int)

    factors = np.array(new_size) / np.array(size)
    delta = (1.0 - factors) / (2.0 * factors)
    vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
    vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
    vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
    II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
    II = torch.tensor(II, dtype=torch.float, device=device)
    JJ = torch.tensor(JJ, dtype=torch.float, device=device)
    KK = torch.tensor(KK, dtype=torch.float, device=device)

    I_small = fast_3D_interp_torch(I_blur, II, JJ, KK) 
    aux_dict.update({'factors': factors})
    return I_small, aux_dict


def add_noise(I, aux_dict, cfg, device, **kwargs):
    noise_std = torch.tensor(cfg.noise_std_min + (cfg.noise_std_max - cfg.noise_std_min) * np.random.rand(1), dtype=torch.float, device=device)
    I_noisy = I + noise_std * torch.randn(I.shape, dtype=torch.float, device=device)
    I_noisy[I_noisy < 0] = 0
    #aux_dict.update({'noise_std': noise_std}) # uncomment if you want to save noise_std for later use
    return I_noisy, aux_dict
    
    
#####################################
#####################################


# map SynthSeg right to left labels for contrast synthesis
right_to_left_dict = {
    41: 2,
    42: 3,
    43: 4,
    44: 5,
    46: 7,
    47: 8,
    49: 10,
    50: 11,
    51: 12,
    52: 13,
    53: 17,
    54: 18,
    58: 26,
    60: 28
}

# based on merged left & right SynthSeg labels
ct_brightness_group = {
    'darker': [4, 5, 14, 15, 24, 31, 72], # ventricles, CSF
    'dark': [2, 7, 16, 77, 30], # white matter
    'bright': [3, 8, 17, 18, 28, 10, 11, 12, 13, 26], # grey matter (cortex, hippocampus, amggdala, ventral DC), thalamus, ganglia (nucleus (putamen, pallidus, accumbens), caudate)
    'brighter': [], # skull, pineal gland, choroid plexus
}
