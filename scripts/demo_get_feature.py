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


gen_cfg = '/autofs/vast/lemon/temp_stuff/brainfm/cfg/full_gen.yaml'
model_cfg = '/autofs/vast/lemon/temp_stuff/brainfm/cfg/train.yaml'

ckp_path = '/autofs/vast/lemon/temp_stuff/brainfm/ckp/regular.pth' 





def get_brainid_feat(im, feat_only = True, win_size = None): 
    outs = utils.evaluate_image(im, ckp_path = ckp_path, feature_only = False, device = device, gen_cfg = gen_cfg, model_cfg = model_cfg)

    if feat_only: # return a 64-dimensional tensor with size (1, 64, x, y, z)
        return outs['feat'][-1] 
    else: # return a dictionary containing all outputs, print out keys to see its elements
        return outs




#####################################################################################
#####################################################################################


if __name__ == '__main__': 

    ### Demo Example Usage ###

    save_dir = make_dir('/autofs/space/yogurt_002/users/pl629/results/tmp', reset = False)
    img_path = '/autofs/vast/lemon/temp_stuff/ana/atlas_test_processed/sub-013.nii'

    ### Read Image ###
    im, im_aff = MRIread(img_path)
    im = torch.from_numpy(im)[None, None].to(device).float() # (1, 1, x, y, z)


    feats = get_brainid_feat(im, True, win_size = None)
    print(feats.shape)
