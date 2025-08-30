"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os 
import shutil 
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk 

import torch 

 

'''if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size'''


def make_dir(dir_name, parents = True, exist_ok = True, reset = False):
    if reset and os.path.isdir(dir_name):
        shutil.rmtree(dir_name) 
    dir_name = Path(dir_name)
    dir_name.mkdir(parents=parents, exist_ok=exist_ok) 
    return dir_name


def read_image(img_path, save_path = None):
    img = nib.load(img_path)
    nda = img.get_fdata()
    affine = img.affine
    if save_path: 
        ni_img = nib.Nifti1Image(nda, affine) 
        nib.save(ni_img, save_path)
    return np.squeeze(nda), affine

def save_image(nda, affine, save_path):
    ni_img = nib.Nifti1Image(nda, affine) 
    nib.save(ni_img, save_path)
    return save_path

def img2nda(img_path, save_path = None):
    img = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(img)
    if save_path:
        np.save(save_path, nda)
    return nda, img.GetOrigin(), img.GetSpacing(), img.GetDirection()

def to3d(img_path, save_path = None):
    nda, o, s, d = img2nda(img_path)
    save_path = img_path if save_path is None else save_path
    if len(o) > 3:
        nda2img(nda, o[:3], s[:3], d[:3] + d[4:7] + d[8:11], save_path)
    return save_path

def nda2img(nda, origin = None, spacing = None, direction = None, save_path = None, isVector = None):
    if type(nda) == torch.Tensor:
        nda = nda.cpu().detach().numpy()
    nda = np.squeeze(np.array(nda)) 
    isVector = isVector if isVector else len(nda.shape) > 3
    img = sitk.GetImageFromArray(nda, isVector = isVector)
    if origin:
        img.SetOrigin(origin)
    if spacing:
        img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    if save_path:
        sitk.WriteImage(img, save_path)
    return img
  


def cropping(img_path, tol = 0, crop_range_lst = None, spare = 0, save_path = None):

    img = sitk.ReadImage(img_path)
    orig_nda = sitk.GetArrayFromImage(img)
    if len(orig_nda.shape) > 3: # 4D data: last axis (t=0) as time dimension
        nda = orig_nda[..., 0]
    else:
        nda = np.copy(orig_nda)
    
    if crop_range_lst is None:
        # Mask of non-black pixels (assuming image has a single channel).
        mask = nda > tol
        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)
        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        # add sparing gap if needed
        x0 = x0 - spare if x0 > spare else x0
        y0 = y0 - spare if y0 > spare else y0
        z0 = z0 - spare if z0 > spare else z0
        x1 = x1 + spare if x1 < orig_nda.shape[0] - spare else x1
        y1 = y1 + spare if y1 < orig_nda.shape[1] - spare else y1
        z1 = z1 + spare if z1 < orig_nda.shape[2] - spare else z1
        
        # Check the the bounding box #
        #print('    Cropping Slice [%d, %d)' % (x0, x1))
        #print('    Cropping Row [%d, %d)' % (y0, y1))
        #print('    Cropping Column [%d, %d)' % (z0, z1))

    else:
        [[x0, y0, z0], [x1, y1, z1]] = crop_range_lst


    cropped_nda = orig_nda[x0 : x1, y0 : y1, z0 : z1]
    new_origin = [img.GetOrigin()[0] + img.GetSpacing()[0] * z0,\
        img.GetOrigin()[1] + img.GetSpacing()[1] * y0,\
            img.GetOrigin()[2] + img.GetSpacing()[2] * x0]  # numpy reverse to sitk'''
    cropped_img = sitk.GetImageFromArray(cropped_nda, isVector = len(orig_nda.shape) > 3)
    cropped_img.SetOrigin(new_origin)
    #cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    if save_path:
        sitk.WriteImage(cropped_img, save_path)

    return cropped_img, [[x0, y0, z0], [x1, y1, z1]], new_origin




def crop_and_pad(orig_nda, crop_idx = [], tol = 1e-7, pad_size = [224, 224, 224], to_print = True):
    if len(crop_idx) < 2:
        [[x0, y0, z0], [x1, y1, z1]] = crop(orig_nda, to_print = to_print)
    else:
        [[x0, y0, z0], [x1, y1, z1]] = crop_idx
    nda = orig_nda[x0:x1, y0:y1, z0:z1]
    nda = pad(nda, pad_size, to_print = to_print)
    return nda, [[x0, y0, z0], [x1, y1, z1]]


def crop(orig_nda, tol = 1e-7, to_print = True):  

    if len(orig_nda.shape) > 3: # 4D data: last axis (t=0) as time dimension
        nda = orig_nda[..., 0]
    else:
        nda = np.copy(orig_nda) 
        
    # Mask of non-black pixels (assuming image has a single channel).
    mask = nda > tol

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    
    # Bounding box of non-black pixels.
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    
    if to_print:
        # Check the the bounding box #
        print('    Cropping Slice [%d, %d)' % (x0, x1))
        print('    Cropping Row [%d, %d)' % (y0, y1))
        print('    Cropping Column [%d, %d)' % (z0, z1))

    return [[x0, y0, z0], [x1, y1, z1]]

def pad(orig_nda, pad_size = [224, 224, 224], to_print = True):
    orig_shape = orig_nda.shape
    to_pad_start = [int((pad_size[i] - orig_shape[i])/2) for i in range(3)]

    if to_print:
        print('     orig shape:', orig_shape)
        print('     pad  start:', to_pad_start)

    new_nda = np.zeros(pad_size)
    new_nda[to_pad_start[0]:to_pad_start[0]+orig_shape[0],
            to_pad_start[1]:to_pad_start[1]+orig_shape[1],
            to_pad_start[2]:to_pad_start[2]+orig_shape[2]] = orig_nda
    
    return new_nda
 
    
#########################################
#########################################


def viewVolume(x, aff=None, prefix='', postfix='', names=[], ext='.nii.gz', save_dir='/tmp'):

    if aff is None:
        aff = np.eye(4)
    else:
        if type(aff) == torch.Tensor:
            aff = aff.cpu().detach().numpy()

    if type(x) is dict:
        names = list(x.keys())
        x = [x[k] for k in x]

    if type(x) is not list:
        x = [x]

    #cmd = 'source /usr/local/freesurfer/nmr-dev-env-bash && freeview '

    for n in range(len(x)):
        vol = x[n]
        if vol is not None:
            if type(vol) == torch.Tensor:
                vol = vol.cpu().detach().numpy()
            vol = np.squeeze(np.array(vol))
            try:
                save_path = os.path.join(save_dir, prefix + names[n] + postfix + ext)
            except:
                save_path = os.path.join(save_dir, prefix + str(n) + postfix + ext)
            MRIwrite(vol, aff, save_path)
            #cmd = cmd + ' ' + save_path

    #os.system(cmd + ' &')
    return save_path

###############################3

def MRIwrite(volume, aff, filename, dtype=None):

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if aff is None:
        aff = np.eye(4)
    header = nib.Nifti1Header()
    nifty = nib.Nifti1Image(volume, aff, header)

    nib.save(nifty, filename)

###############################

def MRIread(filename, dtype=None, im_only=False):
    # dtype example: 'int', 'float'
    assert filename.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % filename

    x = nib.load(filename)
    volume = x.get_fdata()
    aff = x.affine

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff

##############
 