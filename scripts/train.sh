#!/bin/bash

#SBATCH --job-name=lr # reggr hemis lowres age_pool sr_lowres synth
#SBATCH --gpus=1
#SBATCH --partition=lcnrtx # lcna100, lcnrtx, lcna40, rtx8000, dgx-a100, lcnv100, rtx6000

#SBATCH --mail-type=FAIL
#SBATCH --account=lcnlemon #lcnrtx, lcnlemon, mlsclemon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G # 128G
#SBATCH --time=29-23:59:59
#SBATCH --output=/autofs/vast/lemon/temp_stuff/peirong/logs/%j.log # Standard output and error log 


# exp-specific cfg #
gen_cfg_file=/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/train/brain_id_lowres.yaml
train_cfg_file=/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/trainer/train/joint_lowres.yaml


date;hostname;pwd
python /autofs/space/yogurt_003/users/pl629/code/MTBrainID/scripts/train.py $gen_cfg_file $train_cfg_file 
date