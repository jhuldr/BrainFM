# ported from https://github.com/pvigier/perlin-numpy

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from misc import stream_3D, V_plot, center_crop
from utils.misc import viewVolume, make_dir, read_image


#from ShapeID.DiffEqs.odeint import odeint
from ShapeID.DiffEqs.adjoint import odeint_adjoint as odeint
from ShapeID.DiffEqs.pde import AdvDiffPDE

from perlin3d import *




if __name__ == '__main__': 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    percentile = 80
    

    #image, mask_image = generate_perlin_noise_3d([128, 128, 128], [2, 2, 2], tileable=(True, False, False), percentile = percentile)
    #viewVolume(image, names = ['image'], save_dir = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/ShapeID/out/3d')
    #viewVolume(mask_image, names = ['mask_image'], save_dir = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/ShapeID/out/3d') 


    #mask_image, aff = read_image('/autofs/space/yogurt_001/users/pl629/data/adni/pathology_probability/subject_193441.nii.gz')
    mask_image, aff = read_image('/autofs/space/yogurt_001/users/pl629/data/isles2022/pathology_probability/sub-strokecase0127.nii.gz')
    mask_image, _, _ = center_crop(torch.from_numpy(mask_image), win_size = [128, 128, 128])
    mask_image = mask_image[0, 0].numpy()

    shape = mask_image.shape

    curl_a, _ = generate_perlin_noise_3d(shape, [2, 2, 2], tileable=(True, False, False), percentile = percentile) 
    curl_b, _ = generate_perlin_noise_3d(shape, [2, 2, 2], tileable=(True, False, False), percentile = percentile) 
    curl_c, _ = generate_perlin_noise_3d(shape, [2, 2, 2], tileable=(True, False, False), percentile = percentile) 
    dx, dy, dz = stream_3D(torch.from_numpy(curl_a), torch.from_numpy(curl_b), torch.from_numpy(curl_c)) 


    dt = 0.1
    nt = 10
    integ_method = 'dopri5' # choices=['dopri5', 'adams', 'rk4', 'euler'] 
    t = torch.from_numpy(np.arange(nt) * dt).to(device)
    thres = 0.5

    initial = torch.from_numpy(mask_image)[None] # (batch=1, h, w)
    Vx, Vy, Vz = dx * 500, dy * 500, dz * 500
    print(abs(Vx).mean(), abs(Vy).mean(), abs(Vz).mean())

    forward_pde = AdvDiffPDE(data_spacing=[1., 1., 1.], 
                             perf_pattern='adv', 
                             V_type='vector_div_free', 
                             V_dict={'Vx': Vx, 'Vy': Vy, 'Vz': Vz},
                             BC='neumann', 
                             dt=dt, 
                             device=device
                             )

    
    start_time = time.time()
    noise_progression = odeint(forward_pde, 
                               initial, 
                               t, dt, method = integ_method
                               )[:, 0] # (nt, n_batch, h, w)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str)) 
    
    noise_progression = noise_progression[::2]
    noise_progression = noise_progression.numpy()
    make_dir('out/3d/progression')


    for i, noise_t in enumerate(noise_progression):
        noise_t[noise_t > 1] = 1
        noise_t[noise_t <= thres] = 0
        print(i, noise_t.mean())
        viewVolume(noise_t, names = ['noise_%s' % i], save_dir = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/ShapeID/out/3d/progression')
        
        noise_t[noise_t > 0.] = 1
        viewVolume(noise_t, names = ['noise_%s_mask' % i], save_dir = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/ShapeID/out/3d/progression')
