#!/bin/bash

#SBATCH --job-name=test_sr # test_lr
#SBATCH --gpus=1
#SBATCH --partition=rtx8000 # rtx8000, lcnrtx, dgx-a100, lcnv100, rtx6000

#SBATCH --mail-type=FAIL
#SBATCH --account=mlsclemon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24	
#SBATCH --mem=256G
#SBATCH --time=6-23:59:59
#SBATCH --output=/autofs/vast/lemon/temp_stuff/peirong/logs/%j.log # Standard output and error log 


# exp-specific cfg #
#exp_cfg_file='/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/test/feat.yaml'


date;hostname;pwd
python /autofs/space/yogurt_003/users/pl629/code/MTBrainID/scripts/test.py #$exp_cfg_file 
date
