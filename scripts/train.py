
import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import yaml
import json
import random
import time
from argparse import Namespace
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader 

from utils.checkpoint import load_checkpoint 
import utils.logging as logging
import utils.misc as utils 
 
from Generator import build_datasets
from Trainer.visualizer import TaskVisualizer, FeatVisualizer
from Trainer.models import build_model, build_optimizer, build_schedulers
from Trainer.engine import train_one_epoch
 


logger = logging.get_logger(__name__)


# default & gpu cfg #
submit_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/submit.yaml'

default_gen_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/default.yaml'

default_train_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/trainer/default_train.yaml'
default_val_file = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/trainer/default_val.yaml'

gen_cfg_dir = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/generator/train'
train_cfg_dir = '/autofs/space/yogurt_003/users/pl629/code/MTBrainID/cfgs/trainer/train'


def get_params_groups(model):
    all = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        all.append(param)
    return [{'params': all}]


def train(args):

    """
    args: list of configs
    """

    submit_args, gen_args, train_args = args

    utils.init_distributed_mode(submit_args)
    if torch.cuda.is_available():
        if submit_args.num_gpus > torch.cuda.device_count():
            submit_args.num_gpus = torch.cuda.device_count()
        assert (
            submit_args.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        submit_args.num_gpus = 0

    if train_args.debug:
        submit_args.num_workers = 0
 
    output_dir = utils.make_dir(train_args.out_dir)
    cfg_dir = utils.make_dir(os.path.join(output_dir, "cfg")) 
    plt_dir = utils.make_dir(os.path.join(output_dir, "plt")) 
    vis_train_dir = utils.make_dir(os.path.join(output_dir, "vis-train")) 
    ckp_output_dir = utils.make_dir(os.path.join(output_dir, "ckp")) 
    #ckp_epoch_dir = utils.make_dir(os.path.join(ckp_output_dir, "epoch")) 

    yaml.dump(
        vars(submit_args),
        open(cfg_dir / 'config_submit.yaml', 'w'), allow_unicode=True)
    yaml.dump(
        vars(gen_args),
        open(cfg_dir / 'config_generator.yaml', 'w'), allow_unicode=True)
    yaml.dump(
        vars(train_args),
        open(cfg_dir / 'config_trainer.yaml', 'w'), allow_unicode=True)

    # ============ setup logging  ... ============
    logging.setup_logging(output_dir)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(submit_args)).items())))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(gen_args)).items())))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(train_args)).items())))
    log_path = os.path.join(output_dir, 'log.txt')

    if submit_args.device is not None: # assign to specified device
        device = submit_args.device 
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'  
    logger.info('device: %s' % device)

    # fix the seed for reproducibility
    #seed = submit_args.seed + utils.get_rank()
    seed = int(time.time())

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # ============ preparing data ... ============
    dataset_dict = build_datasets(gen_args, device = gen_args.device_generator if gen_args.device_generator is not None else device) 
    data_loader_dict = {}
    data_total = 0
    for name in dataset_dict.keys():
        if submit_args.num_gpus>1:
            sampler_train = utils.DistributedWeightedSampler(dataset_dict[name])
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_dict[name])

        data_loader_dict[name] = DataLoader(
            dataset_dict[name],
            batch_sampler=torch.utils.data.BatchSampler(sampler_train, train_args.batch_size, drop_last=True),
            #collate_fn=utils.collate_fn, # apply custom data cooker if needed
            num_workers=submit_args.num_workers)
        data_total += len(data_loader_dict[name])
        logger.info('Dataset: {}'.format(name))
    logger.info('Num of total training data: {}'.format(data_total))

    visualizers = {'result': TaskVisualizer(gen_args, train_args)}
    if train_args.visualizer.feat_vis:
        visualizers['feature'] = FeatVisualizer(gen_args, train_args) 

    # ============ building model ... ============
    gen_args, train_args, model, processors, criterion, postprocessor = build_model(gen_args, train_args, device = device) # train: True; test: False

    model_without_ddp = model
    # Use multi-process data parallel model in the multi-gpu setting
    if submit_args.num_gpus > 1:
        logger.info('currect device: %s' % str(torch.cuda.current_device()))
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device, 
            find_unused_parameters=True
        )
        model_without_ddp = model.module # unwarp the model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Num of trainable model params: {}'.format(n_parameters))


    # ============ preparing optimizer ... ============
    scaler = torch.cuda.amp.GradScaler()
    param_dicts = get_params_groups(model_without_ddp)
    optimizer = build_optimizer(train_args, param_dicts)

    # ============ init schedulers ... ============ 
    lr_scheduler, wd_scheduler = build_schedulers(train_args, data_total, train_args.lr, train_args.min_lr)
    logger.info(f"Optimizer and schedulers ready.")


    best_val_stats = None 
    train_args.start_epoch = 0  
    # Load weights if provided
    if train_args.resume or train_args.eval_only:
        if train_args.ckp_path:
            ckp_path = train_args.ckp_path
        else:
            ckp_path = sorted(glob.glob(ckp_output_dir + '/*.pth'))

        train_args.start_epoch, best_val_stats = load_checkpoint(ckp_path, [model_without_ddp], optimizer, ['model'], exclude_key = 'supervised_seg') 
        logger.info(f"Resume epoch: {train_args.start_epoch}")
    else:
        logger.info('Starting from scratch')
    if train_args.reset_epoch:
        train_args.start_epoch = 0
    logger.info(f"Start epoch: {train_args.start_epoch}")

    # ============ start training ... ============

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(train_args.start_epoch, train_args.n_epochs):

        if os.path.isfile(os.path.join(ckp_output_dir,'checkpoint_latest.pth')):
            os.rename(os.path.join(ckp_output_dir,'checkpoint_latest.pth'), os.path.join(ckp_output_dir,'checkpoint_latest_bk.pth'))

        checkpoint_paths = [ckp_output_dir / 'checkpoint_latest.pth']
        
        # ============ save model ... ============
        #checkpoint_paths.append(ckp_epoch_dir / f"checkpoint_epoch_{epoch}.pth")

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'submit_args': submit_args,
                'gen_args': gen_args,
                'train_args': train_args,
                'best_val_stats': best_val_stats
            }, checkpoint_path)

        # ============ training one epoch ... ============ 
        if submit_args.num_gpus > 1:
            sampler_train.set_epoch(epoch)
        log_stats = train_one_epoch(epoch, gen_args, train_args, model_without_ddp, processors, criterion, data_loader_dict,
                                    scaler, optimizer, lr_scheduler, wd_scheduler, postprocessor, visualizers, vis_train_dir, device) 
        
        # ============ writing logs ... ============
        if utils.is_main_process():
            with (Path(output_dir) / "log.txt").open("a") as f: 
                f.write('epoch %s - ' % str(epoch).zfill(5))
                f.write(json.dumps(log_stats) + "\n")  
        
        # ============  plot training losses ... ============
        if os.path.isfile(log_path):
            sum_losses = [0.] * (epoch + 1)
            for loss_name in criterion.loss_names:
                curr_epoches, curr_losses = utils.read_log(log_path, 'loss_' + loss_name)
                sum_losses = [sum_losses[i] + curr_losses[i] for i in range(len(curr_losses))]
                utils.plot_loss(curr_losses, os.path.join(utils.make_dir(plt_dir), 'loss_%s.png' % loss_name))
            utils.plot_loss(sum_losses, os.path.join(utils.make_dir(plt_dir), 'loss_all.png'))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str)) 


#####################################################################################

if __name__ == '__main__': 
    submit_args = utils.preprocess_cfg([submit_cfg_file])
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, sys.argv[1]], cfg_dir = gen_cfg_dir)
    train_args = utils.preprocess_cfg([default_train_cfg_file, default_val_file, sys.argv[2]], cfg_dir = train_cfg_dir)
    utils.launch_job(submit_args, gen_args, train_args, train)