###############################
#######  Brani-ID Demo  #######
###############################

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import utils.test_utils as utils 
from utils.misc import viewVolume, make_dir, MRIread

os.system('export FREESURFER_HOME=/usr/local/freesurfer/7.4.1/')
os.system('source $FREESURFER_HOME/SetUpFreeSurfer.sh')

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

model_cfg = 'cfgs/trainer/test/demo_test.yaml'


gen_cfg = 'cfgs/generator/test/demo_test.yaml' # regular
ckp_path = 'ckp/brainfm_pretrained.pth' # regular


data_dir = 'test_data_folder'
img_paths = os.listdir(data_dir)
img_paths = [os.path.join(data_dir, name) for name in img_paths]

main_save_dir = make_dir('outs/test_results', reset = False) 


#win_size = [160, 160, 160]
win_size = None #[192, 192, 192] 
zero_crop_first = True
spacing = None # [1.5, 1.5, 5], None == [1, 1, 1]
add_bf = False




def test(img_path, ckp_path, main_save_dir, win_size = None):

    save_dir = make_dir(os.path.join(make_dir(main_save_dir), os.path.basename(img_path).split('.')[0]), reset = True)

    ### Image Prepration ###
    im, orig, high_res, bf, aff, crop_start, orig_shp = utils.prepare_image(img_path, win_size = win_size, zero_crop_first = zero_crop_first, spacing = None, im_only = False, add_bf = False, device = device) 
    viewVolume(im, aff, names = ['input'], save_dir = save_dir)
    print('input shape:', im.shape[2:])

    ### Inference ###
    outs = utils.evaluate_image(im, ckp_path = ckp_path, feature_only = False, device = device, gen_cfg = gen_cfg, model_cfg = model_cfg)

    mask = im.clone()
    mask[im != 0.] = 1. 

    for k, v in outs.items(): 
        if 'feat' not in k and 'segmentation' not in k:
            viewVolume(v * mask, aff, names = [ 'out_' + k], save_dir = save_dir)
            #viewVolume(v, aff, names = [ 'out_' + k], save_dir = save_dir)

    deformed_atlas = utils.get_deformed_atlas(torch.squeeze(mask),
                                    torch.squeeze(outs['regx']), torch.squeeze(outs['regy']), torch.squeeze(outs['regz'])) 
    viewVolume(deformed_atlas * mask, aff, names = [ 'out_deformed_atlas'], save_dir = save_dir) 


def test_tile(img_path, ckp_path, main_save_dir, stride = [40, 40, 40], win_size = [160, 160, 160]):

    save_dir = make_dir(os.path.join(main_save_dir, os.path.basename(img_path).split('.')[0]), reset = True)

    ### Image Prepration ###
    full_im, orig, high_res, bf, aff, crop_start, orig_shp = utils.prepare_image(img_path, win_size = win_size, zero_crop = zero_crop_first, spacing = spacing, im_only = False, add_bf = add_bf, device = device) 
    viewVolume(full_im, aff, names = ['input_full'], save_dir = save_dir) 
    print('full input shape:', full_im.shape[2:])

    im_list, cnt = utils.tiling(full_im, stride = stride, win_size = win_size)

    print(len(im_list))
    #return

    for i, (im, loc_range) in enumerate(im_list):

        print('Patch #%d/%d' % (i+1, len(im_list)))
        viewVolume(im, aff, names = ['input_%d' % (i+1)], save_dir = save_dir)

        ### Inference ###
        outs = utils.evaluate_image(im, ckp_path = ckp_path, feature_only = False, device = device, gen_cfg = gen_cfg, model_cfg = model_cfg)

        mask = im.clone()
        mask[im != 0.] = 1.

        #feats = outs['feat'][-1]
        #print(feats.size())

        #if not os.path.isfile(os.path.join(save_dir, 'synthsr.nii.gz')):
        #    os.system('mri_synthsr' + ' --i ' + os.path.join(save_dir, 'input.nii.gz') + ' --o ' + os.path.join(save_dir, 'synthsr.nii.gz' + ' --threads 8'))

        for k, v in outs.items():
            print(k)
            if 'feat' not in k and 'segmentation' not in k: 
                viewVolume(v * mask, aff, names = [ 'out_%s_%d' % (k, i+1)], save_dir = save_dir) if len(im_list) > 1 else viewVolume(v * mask, aff, names = [ 'out_%s' % (k)], save_dir = save_dir)

        deformed_atlas = utils.get_deformed_atlas(torch.squeeze(mask),
                                        torch.squeeze(outs['regx']), torch.squeeze(outs['regy']), torch.squeeze(outs['regz'])) 
        viewVolume(deformed_atlas * mask, aff, names = [ 'out_deformed_atlas_%d' % (i+1)], save_dir = save_dir) if len(im_list) > 1 else viewVolume(deformed_atlas * mask, aff, names = [ 'out_deformed_atlas'], save_dir = save_dir)


    # Tiling
    if len(im_list) > 1:
        outs['deformed_atlas'] = None
        for k, _ in outs.items():
            print(k)
            if 'feat' not in k and 'segmentation' not in k:
                full_out = torch.zeros_like(torch.squeeze(full_im))
                for i, (_, loc_range) in enumerate(im_list):
                    im = utils.read_image(os.path.join(save_dir, 'out_%s_%d.nii.gz' % (k, i+1)), is_label = 'label' in k, device = full_out.device)
                    (x_start, x_end), (y_start, y_end), (z_start, z_end) = loc_range
                    full_out[x_start : x_end, y_start : y_end, z_start : z_end] += im
                full_out /= cnt
                viewVolume(full_out, aff, names = [ 'out_%s_tiled' % k], save_dir = save_dir)


############################################################

for img_path in img_paths:
    #test(img_path, ckp_path, main_save_dir, win_size = None) # [160, 160, 160]
    test_tile(img_path, ckp_path, main_save_dir, stride = [80, 80, 80], win_size = [160, 160, 160])


    