# ported from https://github.com/pvigier/perlin-numpy

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from misc import stream_2D, V_plot
from utils.misc import viewVolume, make_dir

from perlin2d import *


#from ShapeID.DiffEqs.odeint import odeint
from ShapeID.DiffEqs.adjoint import odeint_adjoint as odeint
from ShapeID.DiffEqs.pde import AdvDiffPDE


 

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    
    image, mask_image = generate_perlin_noise_2d([256, 256], [2, 2], percentile = 80)  
    plt.imshow(image, cmap='gray') #, interpolation='lanczos')
    plt.axis('off')
    plt.savefig('out/2d/image.png') 
    plt.imshow(mask_image, cmap='gray') #, interpolation='lanczos')
    plt.axis('off')
    plt.savefig('out/2d/mask_image.png') 



    curl, mask_curl = generate_perlin_noise_2d([256, 256], [2, 2], percentile = 80)
    plt.imshow(curl, cmap='gray') #, interpolation='lanczos')
    plt.axis('off')
    plt.savefig('out/2d/curl.png') 
    plt.imshow(mask_curl, cmap='gray') #, interpolation='lanczos')
    plt.axis('off')
    plt.savefig('out/2d/mask_curl.png') 
     

    dx, dy = stream_2D(torch.from_numpy(curl))
    V_plot(dx.numpy(), dy.numpy(), 'out/2d/V.png')

    plt.imshow(mask_image, cmap='gray') #, interpolation='lanczos')
    plt.axis('off')
    plt.savefig('out/2d/image_with_v.png')  
    #plt.close()


    dt = 0.15
    nt = 21
    integ_method = 'dopri5' # choices=['dopri5', 'adams', 'rk4', 'euler'] 
    t = torch.from_numpy(np.arange(nt) * dt).to(device)
    thres = 0.9

    initial = torch.from_numpy(mask_image)
    Vx, Vy = dx * 1000, dy * 1000

    forward_pde = AdvDiffPDE(data_spacing=[1., 1.], 
                             perf_pattern='adv', 
                             V_type='vector_div_free', 
                             V_dict={'Vx': Vx, 'Vy': Vy},
                             BC='neumann', 
                             dt=dt, 
                             device=device
                             )
    
  
    start_time = time.time()
    noise_progression = odeint(forward_pde, 
                               initial.unsqueeze(0), 
                               t, dt, method = integ_method
                               )[:, 0]
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str)) 
    noise_progression = noise_progression[::2]


    noise_progression = noise_progression.numpy()
    make_dir('out/2d/progression')

    for i, noise_t in enumerate(noise_progression): 
        print(i, noise_t.mean())

        noise_t[noise_t > thres] = 1
        noise_t[noise_t <= thres] = 0

        #fig = plt.figure()
        plt.imshow(noise_t, cmap='gray') #, interpolation='lanczos')
        plt.savefig('out/2d/progression/%d.png' % i) 
        #plt.close() 

    viewVolume(noise_progression, names = ['noise_progression'], save_dir = 'out/2d/progression')