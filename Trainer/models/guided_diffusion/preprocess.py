import os
import nibabel

import numpy as np
from misc import crop_and_pad, viewVolume, MRIread, make_dir


test_flag = False

directory = '/autofs/space/yogurt_004/users/pl629/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training'
new_directory = make_dir('/autofs/space/yogurt_004/users/pl629/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training_CropPad')

database = []
mask_vis = []
pad_size = [224, 224, 224]

for root, dirs, files in os.walk(directory):
    dirs_sorted = sorted(dirs)
    for dir_id in dirs_sorted:
        datapoint = dict()
        sli_dict = dict()
        for ro, di, fi in os.walk(root + "/" + str(dir_id)):
            fi_sorted = sorted(fi)

            assert os.path.isfile(os.path.join(root, dir_id, dir_id + '-t1n.nii.gz'))
            
            new_dir = make_dir(os.path.join(new_directory, dir_id))
            print('Create new case dir:', new_dir)
            '''try:
                to_crop, aff = MRIread(os.path.join(root, dir_id, dir_id + '-t1n.nii.gz'), im_only=False, dtype='float')
                _, crop_idx = crop_and_pad(to_crop, pad_size = pad_size, to_print = True)
                print('-- crop_idx:', crop_idx)
            except:
                raise NotImplementedError 


            for f in fi_sorted:
                seqtype = f.split("-")[-1].split(".")[0]
                datapoint[seqtype] = os.path.join(root, dir_id, f)
                print('-- current filename:', f)
                print('-- current seqtype:', seqtype)
                print('-- to save in new_dir:', os.path.join(new_dir, f.split('.')[0] + '.nii.gz'))

                curr_nda, _ = MRIread(os.path.join(root, dir_id, f), im_only=False, dtype='float')
                new_curr_nda, _ = crop_and_pad(curr_nda, crop_idx, pad_size = pad_size, to_print = False)
                viewVolume(new_curr_nda, aff, names = [f.split('.')[0]], save_dir = new_dir)'''

            
            nda, aff = MRIread(os.path.join(new_dir, dir_id + '-t1n.nii.gz'), im_only=False, dtype='float')
            mask, _ = MRIread(os.path.join(new_dir, dir_id + '-mask-healthy.nii.gz'), im_only=False, dtype='float')
            viewVolume(nda * (1 - mask), aff, names = [dir_id + '-t1n-healthyvoided'], save_dir = new_dir)

            database.append(datapoint) 

        #exit()

    break

print('Total num of cases:',  len(database))