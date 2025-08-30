import os, sys, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict 
import random

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset 


from .utils import *
from .constants import n_pathology, pathology_paths, pathology_prob_paths, \
    n_neutral_labels_brainseg_with_extracerebral, label_list_segmentation_brainseg_with_extracerebral, \
    label_list_segmentation_brainseg_left, augmentation_funcs, processing_funcs
import utils.interpol as interpol

from utils.misc import viewVolume


from ShapeID.DiffEqs.pde import AdvDiffPDE
 


class BaseGen(Dataset):
    """
    BaseGen dataset
    """ 
    def __init__(self, gen_args, device='cpu'):

        self.gen_args = gen_args 
        self.split = gen_args.split 

        self.synth_args = self.gen_args.generator
        self.shape_gen_args = gen_args.pathology_shape_generator
        self.real_image_args = gen_args.real_image_generator
        self.synth_image_args = gen_args.synth_image_generator 
        self.augmentation_steps = vars(gen_args.augmentation_steps)
        self.input_prob = vars(gen_args.modality_probs)
        self.device = device

        self.prepare_tasks()
        self.prepare_paths()
        self.prepare_grid()
        self.prepare_one_hot()


    def __len__(self):
        return sum([len(self.names[i]) for i in range(len(self.names))])


    def idx_to_path(self, idx):
        cnt = 0
        for i, l in enumerate(self.datasets_len):
            if idx >= cnt and idx < cnt + l:
                dataset_name = self.datasets[i]
                age = self.ages[i][os.path.basename(self.names[i][idx - cnt]).split('.T1w')[0]] if len(self.ages) > 0 else None
                return dataset_name, vars(self.input_prob[dataset_name]), self.names[i][idx - cnt], age
            else:
                cnt += l


    def prepare_paths(self):

        # Collect list of available images, per dataset
        if len(self.gen_args.dataset_names) < 1:
            datasets = [] 
            g = glob.glob(os.path.join(self.gen_args.data_root, '*' + 'T1w.nii'))
            for i in range(len(g)):
                filename = os.path.basename(g[i])
                dataset = filename[:filename.find('.')]
                found = False
                for d in datasets:
                    if dataset == d:
                        found = True
                if found is False:
                    datasets.append(dataset)
            print('Found ' + str(len(datasets)) + ' datasets with ' + str(len(g)) + ' scans in total')
        else:
            datasets = self.gen_args.dataset_names
        print('Dataset list', datasets)


        names = [] 
        if 'age' in self.tasks: 
            self.split = self.split + '_age'
        if self.gen_args.split_root is not None:
            split_file = open(os.path.join(self.gen_args.split_root, self.split + '.txt'), 'r')
            split_names = []
            for subj in split_file.readlines():
                split_names.append(subj.strip())  

            for i in range(len(datasets)):
                names.append([name for name in split_names if os.path.basename(name).startswith(datasets[i])]) 
        #else:
        #    for i in range(len(datasets)):
        #        names.append(glob.glob(os.path.join(self.gen_args.data_root, datasets[i] + '.*' + 'T1w.nii')))

        # read brain age
        ages = []
        if 'age' in self.tasks: 
            age_file = open(os.path.join(self.gen_args.split_root, 'participants_age.txt'), 'r') 
            subj_name_age = [] 
            for line in age_file.readlines(): # 'subj age\n' 
                subj_name_age.append(line.strip().split(' '))
            for i in range(len(datasets)):
                ages.append({})
                for [name, age] in subj_name_age:
                    if name.startswith(datasets[i]):
                        ages[-1][name] = float(age)
            print('Age info', self.split, len(ages[0].items()), min(ages[0].values()), max(ages[0].values()))
            
        self.ages = ages
        self.names = names
        self.datasets = datasets
        self.datasets_num = len(datasets)
        self.datasets_len = [len(self.names[i]) for i in range(len(self.names))]
        print('Num of data', sum([len(self.names[i]) for i in range(len(self.names))]))

        self.pathology_type = None #setup_dict['pathology_type']
        

    def prepare_tasks(self):
        self.tasks = [key for (key, value) in vars(self.gen_args.task).items() if value]
        if 'bias_field' in self.tasks and 'segmentation' not in self.tasks:
            # add segmentation mask for computing bias_field_soft_mask
            self.tasks += ['segmentation']
        if 'pathology' in self.tasks and self.synth_args.augment_pathology and self.synth_args.random_shape_prob < 1.: 
            self.t = torch.from_numpy(np.arange(self.shape_gen_args.max_nt) * self.shape_gen_args.dt).to(self.device)
            with torch.no_grad():
                self.adv_pde = AdvDiffPDE(data_spacing=[1., 1., 1.], 
                                        perf_pattern='adv', 
                                        V_type='vector_div_free', 
                                        V_dict={},
                                        BC=self.shape_gen_args.bc, 
                                        dt=self.shape_gen_args.dt, 
                                        device=self.device
                                        )
        else:
            self.t, self.adv_pde = None, None
        for task_name in self.tasks: 
            if task_name not in processing_funcs.keys(): 
                print('Warning: Function for task "%s" not found' % task_name)


    def prepare_grid(self): 
        self.size = self.synth_args.size

        # Get resolution of training data
        #aff = nib.load(os.path.join(self.modalities['Gen'], self.names[0])).affine
        #self.res_training_data = np.sqrt(np.sum(abs(aff[:-1, :-1]), axis=0))

        self.res_training_data = np.array([1.0, 1.0, 1.0])

        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
        self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
        self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float, device=self.device)
        self.xc = self.xx - self.c[0]
        self.yc = self.yy - self.c[1]
        self.zc = self.zz - self.c[2]
        return
    
    def prepare_one_hot(self): 
        if self.synth_args.left_hemis_only:
            n_labels = len(label_list_segmentation_brainseg_left)
            label_list_segmentation = label_list_segmentation_brainseg_left
        else:
            # Matrix for one-hot encoding (includes a lookup-table)
            n_labels = len(label_list_segmentation_brainseg_with_extracerebral)
            label_list_segmentation = label_list_segmentation_brainseg_with_extracerebral

        self.lut = torch.zeros(10000, dtype=torch.long, device=self.device)
        for l in range(n_labels):
            self.lut[label_list_segmentation[l]] = l
        self.onehotmatrix = torch.eye(n_labels, dtype=torch.float, device=self.device)
        
        # useless for left_hemis_only
        nlat = int((n_labels - n_neutral_labels_brainseg_with_extracerebral) / 2.0)
        self.vflip = np.concatenate([np.array(range(n_neutral_labels_brainseg_with_extracerebral)),
                                np.array(range(n_neutral_labels_brainseg_with_extracerebral + nlat, n_labels)),
                                np.array(range(n_neutral_labels_brainseg_with_extracerebral, n_neutral_labels_brainseg_with_extracerebral + nlat))])
        return

    
    def random_affine_transform(self, shp):
        rotations = (2 * self.synth_args.max_rotation * np.random.rand(3) - self.synth_args.max_rotation) / 180.0 * np.pi
        shears = (2 * self.synth_args.max_shear * np.random.rand(3) - self.synth_args.max_shear)
        scalings = 1 + (2 * self.synth_args.max_scaling * np.random.rand(3) - self.synth_args.max_scaling)
        scaling_factor_distances = np.prod(scalings) ** .33333333333 
        A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=self.device)

        # sample center
        if self.synth_args.random_shift:
            max_shift = (torch.tensor(np.array(shp[0:3]) - self.size, dtype=torch.float, device=self.device)) / 2
            max_shift[max_shift < 0] = 0
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device) + (2 * (max_shift * torch.rand(3, dtype=float, device=self.device)) - max_shift)
        else:
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device)
        return scaling_factor_distances, A, c2

    def random_nonlinear_transform(self, photo_mode, spac):
        nonlin_scale = self.synth_args.nonlin_scale_min + np.random.rand(1) * (self.synth_args.nonlin_scale_max - self.synth_args.nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = self.synth_args.nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float, device=self.device)
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if 'surface' in self.tasks: # TODO need to integrate the non-linear deformation fields for inverse
            steplength = 1.0 / (2.0 ** self.synth_args.n_steps_svf_integration)
            Fsvf = F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf += fast_3D_interp_torch(Fsvf, self.xx + Fsvf[:, :, :, 0], self.yy + Fsvf[:, :, :, 1], self.zz + Fsvf[:, :, :, 2], 'linear')
            Fsvf_neg = -F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf_neg += fast_3D_interp_torch(Fsvf_neg, self.xx + Fsvf_neg[:, :, :, 0], self.yy + Fsvf_neg[:, :, :, 1], self.zz + Fsvf_neg[:, :, :, 2], 'linear')
            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None
        return F, Fneg
    
    def generate_deformation(self, setups, shp):

        # generate affine deformation
        scaling_factor_distances, A, c2 = self.random_affine_transform(shp)
        
        # generate nonlinear deformation 
        if self.synth_args.nonlinear_transform:
            F, Fneg = self.random_nonlinear_transform(setups['photo_mode'], setups['spac']) 
        else:
            F, Fneg = None, None

        # deform the image grid 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_grid(shp, A, c2, F)  

        return {'scaling_factor_distances': scaling_factor_distances, 
                'A': A, 
                'c2': c2, 
                'F': F, 
                'Fneg': Fneg, 
                'grid': [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], 
                }


    def get_left_hemis_mask(self, grid): 
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = grid

        if self.synth_args.left_hemis_only: 
            S, aff, res = read_image(self.modalities['segmentation']) # read seg map
            S = torch.squeeze(torch.from_numpy(S.get_fdata()[x1:x2, y1:y2, z1:z2].astype(int))).to(self.device)
            S = self.lut[S.int()] # mask out non-left labels
            X, aff, res = read_image(self.modalities['registration'][0]) # read_mni_coord_X
            X = torch.squeeze(torch.from_numpy(X.get_fdata()[x1:x2, y1:y2, z1:z2])).to(self.device)
            self.hemis_mask = ((S > 0) & (X < 0)).int()
        else:
            self.hemis_mask = None
    
    def deform_grid(self, shp, A, c2, F): 
        if F is not None:
            # deform the images (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = self.xc + F[:, :, :, 0]
            yy1 = self.yc + F[:, :, :, 1]
            zz1 = self.zc + F[:, :, :, 2]
        else:
            xx1 = self.xc
            yy1 = self.yc
            zz1 = self.zc
 
        xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
        yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
        zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]  
        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1+torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2


    def augment_sample(self, name, I_def, setups, deform_dict, res, target, pathol_direction = None, input_mode = 'synth'):

        sample = {}
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

        if not isinstance(I_def, torch.Tensor):
            I_def = torch.squeeze(torch.tensor(I_def.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
            if self.hemis_mask is not None:
                I_def[self.hemis_mask == 0] = 0
            # Deform grid
            I_def = fast_3D_interp_torch(I_def, xx2, yy2, zz2, 'linear')

        if input_mode == 'CT':
            I_def = torch.clamp(I_def, min = 0., max = 80.)

        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor) and target['pathology'].sum() > 0:
            I_def = self.encode_pathology(I_def, target['pathology'], target['pathology_prob'], pathol_direction)
            I_def[I_def < 0.] = 0.
        else: 
            target['pathology'] = 0.
            target['pathology_prob'] = 0.  

        # Augment sample
        aux_dict = {}
        augmentation_steps = self.augmentation_steps['synth'] if input_mode == 'synth' else self.augmentation_steps['real']
        for func_name in augmentation_steps:
            I_def, aux_dict = augmentation_funcs[func_name](I = I_def, aux_dict = aux_dict, cfg = self.gen_args.generator, 
                                                         input_mode = input_mode, setups = setups, size = self.size, res = res, device = self.device)


        # Back to original resolution 
        if self.synth_args.bspline_zooming:
            I_def = interpol.resize(I_def, shape=self.size, anchor='edge', interpolation=3, bound='dct2', prefilter=True) 
        else:
            I_def = myzoom_torch(I_def, 1 / aux_dict['factors']) 
            
        maxi = torch.max(I_def)
        I_final = I_def / maxi

        if 'super_resolution' in self.tasks: 
            SRresidual = aux_dict['high_res'] / maxi - I_final
            sample.update({'high_res_residual': torch.flip(SRresidual, [0])[None] if setups['flip'] else SRresidual[None]})


        sample.update({'input': torch.flip(I_final, [0])[None] if setups['flip'] else I_final[None]})
        if 'bias_field' in self.tasks and input_mode != 'CT':
            sample.update({'bias_field_log': torch.flip(aux_dict['BFlog'], [0])[None] if setups['flip'] else aux_dict['BFlog'][None]})

        return sample 
    

    def generate_sample(self, name, G, setups, deform_dict, res, target):  
        
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

        # Generate contrasts
        mus, sigmas = self.get_contrast(setups['photo_mode'])

        G = torch.squeeze(torch.tensor(G.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
        #G[G > 255] = 0 # kill extracerebral regions
        G[G == 77] = 2 # merge WM lesion to white matter region
        if self.hemis_mask is not None:
            G[self.hemis_mask == 0] = 0
        Gr = torch.round(G).long()
        
        SYN = mus[Gr] + sigmas[Gr] * torch.randn(Gr.shape, dtype=torch.float, device=self.device)
        SYN[SYN < 0] = 0
        #SYN /= mus[2] # normalize by WM
        #SYN = gaussian_blur_3d(SYN, 0.5*np.ones(3), self.device) # cosmetic

        SYN = fast_3D_interp_torch(SYN, xx2, yy2, zz2) 

        # Make random linear combinations
        if np.random.rand() < self.gen_args.mix_synth_prob: 
            v = torch.rand(4)
            v[2] = 0 if 'T2' not in self.modalities else v[2]
            v[3] = 0 if 'FLAIR' not in self.modalities else v[3]
            v /= torch.sum(v) 
            SYN = v[0] * SYN + v[1] * target['T1'][0]
            if 'T2' in self.modalities:
                SYN += v[2] * target['T2'][0]
            if 'FLAIR' in self.modalities:
                SYN += v[3] * target['FLAIR'][0] 
            
        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor) and target['pathology'].sum() > 0:
            SYN_cerebral = SYN.clone()
            SYN_cerebral[Gr == 0] = 0
            SYN_cerebral = fast_3D_interp_torch(SYN_cerebral, xx2, yy2, zz2)[None]

            wm_mask = (Gr==2) | (Gr==41)
            wm_mean = (SYN * wm_mask).sum() / wm_mask.sum()  
            gm_mask = (Gr!=0) & (Gr!=2) & (Gr!=41)
            gm_mean = (SYN * gm_mask).sum() / gm_mask.sum()

            target['pathology'][SYN_cerebral == 0] = 0
            target['pathology_prob'][SYN_cerebral == 0] = 0 
            # determine to be T1-resembled or T2-resembled
            #if pathol_direction: lesion should be brigher than WM.mean() 
            # pathol_direction: +1: T2-like; -1: T1-like
            pathol_direction = self.get_pathology_direction('synth', gm_mean > wm_mean)
        else:
            pathol_direction = None
            target['pathology'] = 0.
            target['pathology_prob'] = 0. 
            
        SYN[SYN < 0.] = 0.
        return target['pathology'], target['pathology_prob'], self.augment_sample(name, SYN, setups, deform_dict, res, target, pathol_direction = pathol_direction)
    
    def get_pathology_direction(self, input_mode, pathol_direction = None):  
        #if np.random.rand() < 0.1: # in some (rare) cases, randomly pick the direction
        #    return random.choice([True, False])
        
        if pathol_direction is not None: # for synth image
            return pathol_direction
        
        if input_mode in ['T1', 'CT']:
            return False
        
        if input_mode in ['T2', 'FLAIR']:
            return True
        
        return random.choice([True, False])


    def get_contrast(self, photo_mode):
        # Sample Gaussian image
        mus = 25 + 200 * torch.rand(256, dtype=torch.float, device=self.device)
        sigmas = 5 + 20 * torch.rand(256, dtype=torch.float, device=self.device)

        if np.random.rand() < self.synth_args.ct_prob:
            darker = 25 + 10 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['darker']:
                mus[l] = darker
            dark = 90 + 20 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['dark']:
                mus[l] = dark
            bright = 110 + 20 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['bright']:
                mus[l] = bright
            brighter = 150 + 50 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['brighter']:
                mus[l] = brighter
                
        if photo_mode or np.random.rand(1)<0.5: # set the background to zero every once in a while (or always in photo mode)
            mus[0] = 0

        #  partial volume
        # 1 = lesion, 2 = WM, 3 = GM, 4 = CSF
        v = 0.02 * torch.arange(50).to(self.device)
        mus[100:150] = mus[1] * (1 - v) + mus[2] * v
        mus[150:200] = mus[2] * (1 - v) + mus[3] * v
        mus[200:250] = mus[3] * (1 - v) + mus[4] * v
        mus[250] = mus[4]
        sigmas[100:150] = torch.sqrt(sigmas[1]**2 * (1 - v) + sigmas[2]**2 * v)
        sigmas[150:200] = torch.sqrt(sigmas[2]**2 * (1 - v) + sigmas[3]**2 * v)
        sigmas[200:250] = torch.sqrt(sigmas[3]**2 * (1 - v) + sigmas[4]**2 * v)
        sigmas[250] = sigmas[4]

        return mus, sigmas
    
    def get_setup_params(self): 

        if self.synth_args.left_hemis_only:
            hemis = 'left'
        else:
            hemis = 'both' 

        if self.synth_args.low_res_only:
            photo_mode = False
        elif self.synth_args.left_hemis_only:
            photo_mode = True
        else:
            photo_mode = np.random.rand() < self.synth_args.photo_prob
            
        pathol_mode = np.random.rand() < self.synth_args.pathology_prob
        pathol_random_shape = np.random.rand() < self.synth_args.random_shape_prob
        spac = 2.5 + 10 * np.random.rand() if photo_mode else None  
        flip = np.random.randn() < self.synth_args.flip_prob if not self.synth_args.left_hemis_only else False
        
        if photo_mode: 
            resolution = np.array([self.res_training_data[0], spac, self.res_training_data[2]])
            thickness = np.array([self.res_training_data[0], 0.1, self.res_training_data[2]])
        else:
            resolution, thickness = resolution_sampler(self.synth_args.low_res_only)
        return {'resolution': resolution, 'thickness': thickness, 
                'photo_mode': photo_mode, 'pathol_mode': pathol_mode, 
                'pathol_random_shape': pathol_random_shape,
                'spac': spac, 'flip': flip, 'hemis': hemis}
    
    
    def encode_pathology(self, I, P, Pprob, pathol_direction = None):


        if pathol_direction is None: # True: T2/FLAIR-resembled, False: T1-resembled
            pathol_direction = random.choice([True, False])

        P, Pprob = torch.squeeze(P), torch.squeeze(Pprob)
        I_mu = (I * P).sum() / P.sum()

        p_mask = torch.round(P).long()
        #pth_mus = I_mu/4 + I_mu/2 * torch.rand(10000, dtype=torch.float, device=self.device)
        pth_mus = 3*I_mu/4 + I_mu/4 * torch.rand(10000, dtype=torch.float, device=self.device) # enforce the pathology pattern harder!
        pth_mus = pth_mus if pathol_direction else -pth_mus 
        pth_sigmas = I_mu/4 * torch.rand(10000, dtype=torch.float, device=self.device)
        I += Pprob * (pth_mus[p_mask] + pth_sigmas[p_mask] * torch.randn(p_mask.shape, dtype=torch.float, device=self.device))
        I[I < 0] = 0

        #print('encode', P.shape, P.mean()) 
        #print('pre', I_mu) 
        #I_mu = (I * P).sum() / P.sum()
        #print('post', I_mu)

        return I
    
    def get_info(self, t1):

        t1dm = t1[:-7] + 'T1w.defacingmask.nii'
        t2 = t1[:-7] + 'T2w.nii'
        t2dm = t1[:-7] + 'T2w.defacingmask.nii'
        flair = t1[:-7] + 'FLAIR.nii'
        flairdm = t1[:-7] + 'FLAIR.defacingmask.nii'
        ct = t1[:-7] + 'CT.nii'
        ctdm = t1[:-7] + 'CT.defacingmask.nii'
        generation_labels = t1[:-7] + 'generation_labels.nii' 
        segmentation_labels = t1[:-7] + self.gen_args.segment_prefix + '.nii'
        #brain_dist_map = t1[:-7] + 'brain_dist_map.nii'
        lp_dist_map = t1[:-7] + 'lp_dist_map.nii'
        rp_dist_map = t1[:-7] + 'rp_dist_map.nii'
        lw_dist_map = t1[:-7] + 'lw_dist_map.nii'
        rw_dist_map = t1[:-7] + 'rw_dist_map.nii'
        mni_reg_x = t1[:-7] + 'mni_reg.x.nii'
        mni_reg_y = t1[:-7] + 'mni_reg.y.nii'
        mni_reg_z = t1[:-7] + 'mni_reg.z.nii'


        self.modalities = {'T1': t1, 'Gen': generation_labels, 'segmentation': segmentation_labels,   
                           'distance': [lp_dist_map, lw_dist_map, rp_dist_map, rw_dist_map],
                           'registration': [mni_reg_x, mni_reg_y, mni_reg_z]}

        if os.path.isfile(t1dm):
            self.modalities.update({'T1_DM': t1dm}) 
        if os.path.isfile(t2):
            self.modalities.update({'T2': t2}) 
        if os.path.isfile(t2dm):
            self.modalities.update({'T2_DM': t2dm}) 
        if os.path.isfile(flair):
            self.modalities.update({'FLAIR': flair}) 
        if os.path.isfile(flairdm):  
            self.modalities.update({'FLAIR_DM': flairdm}) 
        if os.path.isfile(ct): 
            self.modalities.update({'CT': ct}) 
        if os.path.isfile(ctdm): 
            self.modalities.update({'CT_DM': ctdm}) 

        return self.modalities


    def read_input(self, idx):
        """
        determine input type according to prob (in generator/constants.py)
        Logic: if np.random.rand() < real_image_prob and is real_image_exist --> input real images; otherwise, synthesize images. 
        """
        dataset_name, input_prob, t1_path, age = self.idx_to_path(idx)
        case_name = os.path.basename(t1_path).split('.T1w.nii')[0]
        self.modalities = self.get_info(t1_path)

        prob = np.random.rand() 
        if prob < input_prob['T1'] and 'T1' in self.modalities:
            input_mode = 'T1'
            img, aff, res = read_image(self.modalities['T1']) 
        elif prob < input_prob['T2'] and 'T2' in self.modalities:
            input_mode = 'T2'
            img, aff, res = read_image(self.modalities['T2']) 
        elif prob < input_prob['FLAIR'] and 'FLAIR' in self.modalities:
            input_mode = 'FLAIR'
            img, aff, res = read_image(self.modalities['FLAIR']) 
        elif prob < input_prob['CT'] and 'CT' in self.modalities:
            input_mode = 'CT'
            img, aff, res = read_image(self.modalities['CT']) 
        else:
            input_mode = 'synth' 
            img, aff, res = read_image(self.modalities['Gen']) 

        return dataset_name, case_name, input_mode, img, aff, res, age
    

    def read_and_deform_target(self, idx, exist_keys, task_name, input_mode, setups, deform_dict, linear_weights = None):
        current_target = {}
        p_prob_path, augment, thres = None, False, 0.1

        if task_name == 'pathology':
            # NOTE: for now - encode pathology only for healthy cases
            # TODO: what to do if the case has pathology itself? -- inconsistency between encoded pathol and the output
            if self.pathology_type is None: # healthy
                if setups['pathol_mode']: # and input_mode == 'synth':
                    if setups['pathol_random_shape']:
                        p_prob_path = 'random_shape'
                        augment, thres = False, self.shape_gen_args.pathol_thres 
                    else:
                        p_prob_path = random.choice(pathology_prob_paths)
                        augment, thres = self.synth_args.augment_pathology, self.shape_gen_args.pathol_thres 
            else: 
                pass
                #p_prob_path = self.modalities['pathology_prob'] 

            current_target = processing_funcs[task_name](exist_keys, task_name, p_prob_path, setups, deform_dict, self.device,
                                                         mask = self.hemis_mask,
                                                         augment = augment, 
                                                         pde_func = self.adv_pde, 
                                                         t = self.t, 
                                                         shape_gen_args = self.shape_gen_args, 
                                                         thres = thres
                                                         )
            
        else:
            if task_name in self.modalities:
                current_target = processing_funcs[task_name](exist_keys, task_name, self.modalities[task_name], 
                                                            setups, deform_dict, self.device, 
                                                            mask = self.hemis_mask,
                                                            cfg = self.gen_args, 
                                                            onehotmatrix = self.onehotmatrix, 
                                                            lut = self.lut, vflip = self.vflip
                                                            )
            else:
                current_target = {task_name: 0.}
        return current_target
    
        
    def update_gen_args(self, new_args):
        for key, value in vars(new_args).items():
            vars(self.gen_args.generator)[key] = value 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        # read input: real or synthesized image, according to customized prob
        dataset_name, case_name, input_mode, img, aff, res, age = self.read_input(idx)

        # generate random values
        setups = self.get_setup_params()

        # sample random deformation
        deform_dict = self.generate_deformation(setups, img.shape)

        # get left_hemis_mask if needed
        self.get_left_hemis_mask(deform_dict['grid'])

        # read and deform target according to the assigned tasks
        target = defaultdict(lambda: None)
        target['name'] = case_name
        target.update(self.read_and_deform_target(idx, target.keys(), 'T1', input_mode, setups, deform_dict))
        target.update(self.read_and_deform_target(idx, target.keys(), 'T2', input_mode, setups, deform_dict)) 
        target.update(self.read_and_deform_target(idx, target.keys(), 'FLAIR', input_mode, setups, deform_dict))
        for task_name in self.tasks:
            if task_name in processing_funcs.keys() and task_name not in ['T1', 'T2', 'FLAIR']: 
                target.update(self.read_and_deform_target(idx, target.keys(), task_name, input_mode, setups, deform_dict))
        

        # process or generate input sample
        if input_mode == 'synth':
            self.update_gen_args(self.synth_image_args) # severe noise injection for real images
            target['pathology'], target['pathology_prob'], sample = \
                self.generate_sample(case_name, img, setups, deform_dict, res, target)  
        else:
            self.update_gen_args(self.real_image_args) # milder noise injection for real images
            sample = self.augment_sample(case_name, img, setups, deform_dict, res, target,  
                                        pathol_direction = self.get_pathology_direction(input_mode),input_mode = input_mode)

        if setups['flip'] and isinstance(target['pathology'], torch.Tensor): # flipping should happen after P has been encoded
            target['pathology'], target['pathology_prob'] = torch.flip(target['pathology'], [1]), torch.flip(target['pathology_prob'], [1]) 
        
        if age is not None:
            target['age'] = age

        return self.datasets_num, dataset_name, input_mode, target, sample




# An example of customized dataset from BaseSynth
class BrainIDGen(BaseGen):
    """
    BrainIDGen dataset
    BrainIDGen enables intra-subject augmentation, i.e., each subject will have multiple augmentations
    """
    def __init__(self, gen_args, device='cpu'):  
        super(BrainIDGen, self).__init__(gen_args, device)

        self.all_samples = gen_args.generator.all_samples 
        self.mild_samples = gen_args.generator.mild_samples 
        self.mild_generator_args = gen_args.mild_generator
        self.severe_generator_args = gen_args.severe_generator
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        # read input: real or synthesized image, according to customized prob 
        dataset_name, case_name, input_mode, img, aff, res, age = self.read_input(idx)

        # generate random values
        setups = self.get_setup_params()

        # sample random deformation
        deform_dict = self.generate_deformation(setups, img.shape) 

        # get left_hemis_mask if needed
        self.get_left_hemis_mask(deform_dict['grid'])

        # read and deform target according to the assigned tasks
        target = defaultdict(lambda: 1.)
        target['name'] = case_name
        target.update(self.read_and_deform_target(idx, target.keys(), 'T1', input_mode, setups, deform_dict))
        target.update(self.read_and_deform_target(idx, target.keys(), 'T2', input_mode, setups, deform_dict)) 
        target.update(self.read_and_deform_target(idx, target.keys(), 'FLAIR', input_mode, setups, deform_dict))
        for task_name in self.tasks:
            if task_name in processing_funcs.keys() and task_name not in ['T1', 'T2', 'FLAIR']: 
                target.update(self.read_and_deform_target(idx, target.keys(), task_name, input_mode, setups, deform_dict)) 

        # process or generate intra-subject input samples 
        samples = []
        for i_sample in range(self.all_samples):
            if i_sample < self.mild_samples:  
                self.update_gen_args(self.mild_generator_args)
                if input_mode == 'synth':
                    self.update_gen_args(self.synth_image_args)
                    target['pathology'], target['pathology_prob'], sample = \
                        self.generate_sample(case_name, img, setups, deform_dict, res, target) 
                else:
                    self.update_gen_args(self.real_image_args)
                    sample = self.augment_sample(case_name, img, setups, deform_dict, res, target,  
                                                 pathol_direction = self.get_pathology_direction(input_mode),input_mode = input_mode)
            else: 
                self.update_gen_args(self.severe_generator_args)
                if input_mode == 'synth':
                    self.update_gen_args(self.synth_image_args)
                    target['pathology'], target['pathology_prob'], sample = \
                        self.generate_sample(case_name, img, setups, deform_dict, res, target)  
                else:
                    self.update_gen_args(self.real_image_args) 
                    sample = self.augment_sample(case_name, img, setups, deform_dict, res, target, 
                                                 pathol_direction = self.get_pathology_direction(input_mode),input_mode = input_mode)

            samples.append(sample) 
        
        if setups['flip'] and isinstance(target['pathology'], torch.Tensor): # flipping should happen after P has been encoded
            target['pathology'], target['pathology_prob'] = torch.flip(target['pathology'], [1]), torch.flip(target['pathology_prob'], [1]) 
 
        if age is not None:
            target['age'] = age
        return self.datasets_num, dataset_name, input_mode, target, samples