import os, glob

from .utils import *

augmentation_funcs = {
    'gamma': add_gamma_transform,
    'bias_field': add_bias_field,
    'resample': resample_resolution,
    'noise': add_noise,
}

processing_funcs = {
    'T1': read_and_deform_image,
    'T2': read_and_deform_image,
    'FLAIR': read_and_deform_image,
    'CT': read_and_deform_CT,
    'segmentation': read_and_deform_segmentation,
    'surface': read_and_deform_surface,
    'distance': read_and_deform_distance,
    'bias_field': read_and_deform_bias_field,
    'registration': read_and_deform_registration,
    'pathology': read_and_deform_pathology, 
}


dataset_setups = { 

    'ADHD': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/adhd200_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': None,

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'HCP': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/hcp_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'T2'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': None,
                'CT': None,

                # processed ground truths 
                'surface': None, #'surfaces', 
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'AIBL': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/aibl_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'T2', 'FLAIR'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': None,

                # processed ground truths 
                'surface': None, #'surfaces', 
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'OASIS': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/oasis3',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'CT'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': 'CT',

                # processed ground truths 
                'surface': None, #'surfaces', 
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'ADNI': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/adni_crop',
        'pathology_type': None, #'wmh',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': 'Dmaps', 
                'DmapsBag': 'DmapsBag', 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': None,

                # processed ground truths
                'surface': 'surfaces',  
                'distance': 'Dmaps',  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'ADNI3': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/adni3_crop',
        'pathology_type': None, # 'wmh',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'FLAIR'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': 'FLAIR',
                'CT': None,

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'ATLAS': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/atlas_crop',
        'pathology_type': 'stroke',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': None,

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'ISLES': { 
        'root': '/autofs/space/yogurt_001/users/pl629/data/isles2022_crop',
        'pathology_type': 'stroke',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['FLAIR'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': None, 
                'T2': None, 
                'FLAIR': 'FLAIR',
                'CT': None,

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_field': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },
}


all_dataset_names = dataset_setups.keys()


# get all pathologies
pathology_paths = []
pathology_prob_paths = []
for name, dict in dataset_setups.items():
    # TODO: select what kind of shapes?
    if dict['paths']['pathology'] is not None and dict['pathology_type'] is not None and dict['pathology_type'] == 'stroke':   
        pathology_paths += glob.glob(os.path.join(dict['root'], dict['paths']['pathology'], '*.nii.gz')) \
                                        + glob.glob(os.path.join(dict['root'], dict['paths']['pathology'], '*.nii'))
        pathology_prob_paths += glob.glob(os.path.join(dict['root'], dict['paths']['pathology_prob'], '*.nii.gz')) \
                                        + glob.glob(os.path.join(dict['root'], dict['paths']['pathology_prob'], '*.nii'))
n_pathology = len(pathology_paths)


# with csf # NOTE old version (FreeSurfer standard), non-vast
label_list_segmentation = [0,14,15,16,24,77,85,   2, 3, 4, 7, 8, 10,11,12,13,17,18,26,28,   41,42,43,46,47,49,50,51,52,53,54,58,60] # 33
n_neutral_labels = 7


## NEW VAST synth
label_list_segmentation_brainseg_with_extracerebral =  [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46,
                                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51, 53, 55,
                                   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50, 52, 54, 56] 
n_neutral_labels_brainseg_with_extracerebral = 20

label_list_segmentation_brainseg_left = [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 17, 31, 34, 36, 38, 40, 42]

