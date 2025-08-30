###############################
####  Brani-ID Inference  #####
###############################

import os, sys, warnings, shutil, glob, time, datetime
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict 

import torch
import numpy as np

from utils.misc import make_dir, viewVolume, MRIread
import utils.test_utils as utils 
from Generator.utils import fast_3D_interp_torch 

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


#################################
### For hemisphere prediction ###
#################################
label_list_left_segmentation = [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 17, 31, 34, 36, 38, 40, 42]
lut = torch.zeros(10000, dtype=torch.long, device=device)
for l in range(len(label_list_left_segmentation)):
    lut[label_list_left_segmentation[l]] = l

# get left hemis mask
#S = read_brainseg.nii
#S = lut[X.astype(np.int)]
#X = read_mni_coord_X
#M = (S > 0) & (X < 0) 

# apply hemis mask for all image I
#I[M==0] = 0 
#################################
#################################


def prepare_paths(data_root, split_txt):

    # Collect list of available images, per dataset
    datasets = [] 
    g = glob.glob(os.path.join(data_root, '*' + 'T1w.nii'))
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
    print('Dataset list', datasets)
    names = []

    split_file = open(split_txt, 'r')
    split_names = []
    for subj in split_file.readlines():
        split_names.append(subj.strip())  

    for i in range(len(datasets)):
        names.append([name for name in split_names if os.path.basename(name).startswith(datasets[i])])  
        
    datasets_num = len(datasets)
    datasets_len = [len(names[i]) for i in range(len(names))]
    print('Num of testing data', sum([len(names[i]) for i in range(len(names))]))

    return names, datasets


def get_info(t1):
 
    t2 = t1[:-7] + 'T2w.nii' 
    flair = t1[:-7] + 'FLAIR.nii' 
    ct = t1[:-7] + 'CT.nii' 
    cerebral_labels = t1[:-7] + 'brainseg.nii' 
    segmentation_labels = t1[:-7] + 'brainseg_with_extracerebral.nii'
    brain_dist_map = t1[:-7] + 'brain_dist_map.nii'
    lp_dist_map = t1[:-7] + 'lp_dist_map.nii'
    rp_dist_map = t1[:-7] + 'rp_dist_map.nii'
    lw_dist_map = t1[:-7] + 'lw_dist_map.nii'
    rw_dist_map = t1[:-7] + 'rw_dist_map.nii'
    mni_reg_x = t1[:-7] + 'mni_reg.x.nii'
    mni_reg_y = t1[:-7] + 'mni_reg.y.nii'
    mni_reg_z = t1[:-7] + 'mni_reg.z.nii'

    modalities = {'T1': t1}
    if os.path.isfile(t2):
        modalities.update({'T2': t2})  
    if os.path.isfile(flair):
        modalities.update({'FLAIR': flair})  
    if os.path.isfile(ct): 
        modalities.update({'CT': ct})  

    aux = {'label': segmentation_labels, 'cerebral_label': cerebral_labels, 'distance': brain_dist_map,
           'regx': mni_reg_x, 'regy': mni_reg_y, 'regz': mni_reg_z, 
            'lp': lp_dist_map, 'lw': lw_dist_map, 'rp': rp_dist_map, 'rw': rw_dist_map}

    return modalities, aux


#################################


gen_cfg = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/test/demo_test.yaml'
gen_hemis_cfg = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/test/demo_test_hemis.yaml'
model_cfg = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/trainer/test/demo_test.yaml'

#win_size = [192, 192, 192] 
win_size = [160, 160, 160] 
mask_output = False 


exclude_keys = ['segmentation']
data_root = '/autofs/vast/lemon/data_curated/brain_mris_QCed'
split_txt = '/autofs/vast/lemon/temp_stuff/peirong/train_test_split/test.txt'
names, datasets = prepare_paths(data_root, split_txt)


max_num_test_dataset = None #1
max_num_per_dataset  = None #5

zero_crop = False

main_save_dir = make_dir('/autofs/space/yogurt_002/users/pl629/results/MTBrainID/test/', reset = False)

models = [
    #('test_reggr', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/wosr_reggrad/l6_16/1025-1744/ckp/checkpoint_latest.pth'),
    #('test_lowres', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/wosr_reggrad_lowres/l6_16/1025-1746/ckp/checkpoint_latest.pth'),
    ('test_sr', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/sr/l6_16/0926-2035/ckp/checkpoint_latest.pth'),
    #('test_sr_lowres', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/sr_lowres/l6_16/0926-2025/ckp/checkpoint_latest.pth'),
    #('test_hemis', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/hemis/l6_16/0806-1008/ckp/checkpoint_latest.pth'),

    #('comp_synth', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/synth/l6_16/0924-0929/ckp/checkpoint_latest.pth'),
    #('comp_dist', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/dist/l6_16/0806-1024/ckp/checkpoint_latest.pth'),
    #('comp_reg', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/reg/l6_16/0806-1029/ckp/checkpoint_latest.pth'),
    #('comp_bf', '/autofs/vast/lemon/temp_stuff/peirong/results/MTBrainID/bf/l6_16/0806-1026/ckp/checkpoint_latest.pth'),
]

#spacing = [1.5, 1.5, 5] # [1, 1, 1], [1.5, 1.5, 5], [3, 3, 3], None 
#add_bf = False
setups = [
    #([1, 1, 1], False), 
    #([1, 1, 1], True),
    ([1.5, 1.5, 5], False), 
    #([1.5, 1.5, 5], True), 
]



all_start_time = time.time()
for postfix, ckp_path in models:

    for spacing, add_bf in setups:
        curr_postfix = postfix + '_BF' if add_bf else postfix
        curr_postfix += '_%s-%s-%s' % (str(spacing[0]), str(spacing[1]), str(spacing[2])) if spacing is not None else '_1-1-1' 
        save_dir = make_dir(os.path.join(main_save_dir, curr_postfix), reset = True)
        print('\nSave at: %s\n' % save_dir)

        curr_gen_cfg = gen_hemis_cfg if 'hemis' in postfix else gen_cfg


        for i, curr_dataset in enumerate(names):
            curr_dataset.sort()
            print('Dataset: %s (%d/%d) -- %d total cases' % (datasets[i], i+1, len(datasets), len(curr_dataset))) 

            #'''
            if max_num_test_dataset is not None and i >= max_num_test_dataset:
                break

            start_time = time.time()
            for j, t1_name in enumerate(curr_dataset):

                if max_num_per_dataset is not None and j >= max_num_per_dataset:
                    break

                subj_name = os.path.basename(t1_name).split('.T1w')[0]
                subj_dir = make_dir(os.path.join(save_dir, subj_name))
                print('Now testing: %s (%d/%d)' % (t1_name, j+1, len(curr_dataset)))

                modalities, aux = get_info(t1_name)

                S_cerebral = torch.squeeze(utils.prepare_image(aux['cerebral_label'], win_size = win_size, zero_crop = zero_crop, spacing = spacing, rescale = False, im_only = True, device = device)) # read seg map
                
                if 'hemis' in postfix:  
                    S = utils.prepare_image(aux['cerebral_label'], win_size = win_size, zero_crop = zero_crop, spacing = spacing, rescale = False, im_only = True, device = device) # read seg map
                    S = lut[S.int()] # mask out non-left labels
                    X = utils.prepare_image(aux['regx'], win_size = win_size, zero_crop = zero_crop, spacing = spacing, rescale = False, im_only = True, device = device) # read_mni_coord_X
                    hemis_mask = (S > 0) & (X < 0).int()  # (1, 1, s, r, c)
                    viewVolume(hemis_mask, names = ['.'.join(os.path.basename(aux['label']).split('.')[:2]) + '.hemis_mask'], save_dir = subj_dir)
                else:
                    hemis_mask = None

                # save all GT
                for mod in modalities.keys():
                    final, orig, high_res, bf, _, _, _ = utils.prepare_image(modalities[mod], win_size = win_size, zero_crop = zero_crop, spacing = spacing, add_bf = add_bf, is_CT = 'CT' in mod, rescale = False, hemis_mask = hemis_mask, im_only = False, device = device)
                    viewVolume(orig, names = [os.path.basename(modalities[mod])[:-4]], save_dir = subj_dir)
                    viewVolume(final, names = [os.path.basename(modalities[mod])[:-4] + '.input'], save_dir = subj_dir)
                    viewVolume(high_res, names = [os.path.basename(modalities[mod])[:-4] + '.high_res'], save_dir = subj_dir)
                    if bf is not None:
                        viewVolume(bf, names = [os.path.basename(modalities[mod])[:-4] + '.bias_field'], save_dir = subj_dir)
                for mod in aux.keys():
                    im = utils.prepare_image(aux[mod], win_size = win_size, zero_crop = zero_crop, is_label = 'label' in mod, rescale = False, hemis_mask = hemis_mask, im_only = True, device = device)
                    viewVolume(im, names = [os.path.basename(aux[mod])[:-4]], save_dir = subj_dir)

                # testing
                for mod in modalities.keys():
                    test_dir = make_dir(os.path.join(subj_dir, 'input_' + mod))
                    im = utils.prepare_image(os.path.join(subj_dir, os.path.basename(modalities[mod])[:-4] + '.input.nii.gz'), win_size = win_size, zero_crop = zero_crop, is_CT = 'CT' in mod, hemis_mask = hemis_mask, im_only = True, device = device)
                    outs = utils.evaluate_image(im, ckp_path = ckp_path, feature_only = False, device = device, gen_cfg = curr_gen_cfg, model_cfg = model_cfg)

                    if mask_output:
                        mask = im.clone()
                        mask[im != 0.] = 1. 

                    for k, v in outs.items(): 
                        if 'feat' not in k and k not in exclude_keys: 
                            viewVolume(v * mask if mask_output else v, names = [ 'out_' + k], save_dir = test_dir)
                    
                    print(S_cerebral.shape, outs['regx'].shape)
                    deformed_atlas = utils.get_deformed_atlas(S_cerebral, torch.squeeze(outs['regx']), torch.squeeze(outs['regy']), torch.squeeze(outs['regz'])) 
                    viewVolume(deformed_atlas * mask if mask_output else deformed_atlas, names = [ 'out_deformed_atlas'], save_dir = test_dir)
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Testing time for {}: {}'.format(total_time_str, datasets[i]))
            
        all_total_time = time.time() - all_start_time
        all_total_time_str = str(datetime.timedelta(seconds=int(all_total_time)))
print('Total testing time: {}'.format(total_time_str))
#'''