###############################
####  Synthetic Data Demo  ####
###############################


import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import torch

import utils.misc as utils 
 

from Generator import build_datasets 



# default & gpu cfg # 
default_gen_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/default.yaml' 
demo_gen_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/test/demo_synth.yaml'


def map_back_orig(img, idx, shp):
    if idx is None or shp is None:
        return img
    if len(img.shape) == 3:
        img = img[None, None]
    elif len(img.shape) == 4:
        img = img[None]
    return img[:, :, idx[0]:idx[0] + shp[0], idx[1]:idx[1] + shp[1], idx[2]:idx[2] + shp[2]]


def generate(args):

    _, gen_args, _ = args
 
    if gen_args.device_generator:
        device = gen_args.device_generator
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'  
    print('device: %s' % device) 

    print('out_dir:', gen_args.out_dir)

    # ============ preparing data ... ============ 
    dataset_dict = build_datasets(gen_args, device = gen_args.device_generator if gen_args.device_generator is not None else device) 
    dataset = dataset_dict[gen_args.dataset_names[0]] 

    tasks = [key for (key, value) in vars(gen_args.task).items() if value]
       
    print("Start generating")
    start_time = time.time()


    dataset.mild_samples = gen_args.mild_samples
    dataset.all_samples = gen_args.all_samples 
    for itr in range(min(gen_args.test_itr_limit, len(dataset.names))):
        
        subj_name = os.path.basename(dataset.names[itr]).split('.nii')[0]

        save_dir = utils.make_dir(os.path.join(gen_args.out_dir, subj_name))

        print('Processing image (%d/%d): %s' % (itr, len(dataset), dataset.names[itr]))

        for i_deform in range(gen_args.num_deformations):
            def_save_dir = utils.make_dir(os.path.join(save_dir, 'deform-%s' % i_deform))

            (_, subjects, samples) = dataset.__getitem__(itr)
                
            if 'aff' in subjects:
                aff = subjects['aff']
                shp = subjects['shp']
                loc_idx = subjects['loc_idx']
            else:
                aff = torch.eye((4))
                shp = loc_idx = None
            
            print('num samples:', len(samples))
            print('     deform:', i_deform)
            
            #print(subjects.keys())

            if 'T1' in subjects:
                utils.viewVolume(subjects['T1'], aff, names = ['T1'], save_dir = def_save_dir)
            if 'T2' in subjects:
                utils.viewVolume(subjects['T2'], aff, names = ['T2'], save_dir = def_save_dir)
            if 'FLAIR' in subjects:
                utils.viewVolume(subjects['FLAIR'], aff, names = ['FLAIR'], save_dir = def_save_dir)
            if 'CT' in subjects:
                utils.viewVolume(subjects['CT'], aff, names = ['CT'], save_dir = def_save_dir) 
            if 'pathology' in tasks:
                utils.viewVolume(subjects['pathology'], aff, names = ['pathology'], save_dir = def_save_dir)
            if 'segmentation' in tasks:
                utils.viewVolume(subjects['segmentation']['label'], aff, names = ['label'], save_dir = def_save_dir)

            for i_sample, sample in enumerate(samples):
                print('         sample:', i_sample)
                sample_save_dir = utils.make_dir(os.path.join(def_save_dir, 'sample-%s' % i_sample))

                #print(sample.keys())
                
                if 'input' in sample:
                    utils.viewVolume(map_back_orig(sample['input'], loc_idx, shp), aff, names = ['input'], save_dir = sample_save_dir)
                if 'super_resolution' in tasks:
                    utils.viewVolume(map_back_orig(sample['orig'], loc_idx, shp), aff, names = ['high_reso'], save_dir = sample_save_dir)
                if 'bias_field' in tasks:
                    utils.viewVolume(map_back_orig(torch.exp(sample['bias_field_log']), loc_idx, shp), aff, names = ['bias_field'], save_dir = sample_save_dir)

 
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Generation time {}'.format(total_time_str))


#####################################################################################


if __name__ == '__main__': 
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, demo_gen_cfg_file])
    utils.launch_job(submit_cfg = None, gen_cfg = gen_args, train_cfg = None, func = generate)