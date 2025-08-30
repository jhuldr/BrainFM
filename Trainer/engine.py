
"""
Train and eval functions
"""
import os, random
import math 
import time

import torch
import numpy as np

import utils.misc as utils
import utils.logging as logging


logger = logging.get_logger(__name__)




def make_results(target, samples, outputs, out_dir): 
    case_names = target['name']
    results = outputs
    case_out_dir = utils.make_dir(os.path.join(out_dir, case_names[0], 'results'))

    if 'aff' in target:
        aff = target['aff'][0]
    else:
        aff = None

    if 'label' in target:
        utils.viewVolume(target['label'], aff = aff, names = ['label'], prefix = 'gt_', save_dir = case_out_dir) 
    if 'image' in target:
        utils.viewVolume(target['image'], aff = aff, names = ['image'], prefix = 'gt_', save_dir = case_out_dir)  
    if 'image_orig' in target:
        utils.viewVolume(target['image_orig'], aff = aff, names = ['image_orig'], prefix = 'gt_', save_dir = case_out_dir)   
    
    for i_sample, sample in enumerate(samples):

        if 'bias_field_log' in sample:
            utils.viewVolume(torch.exp(sample['bias_field_log']), aff = aff, names = ['bflog'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(torch.exp(outputs[i_sample]['bias_field_log']), aff = aff, names = ['bflog'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

        if 'input' in sample:
            utils.viewVolume(sample['input'], aff = aff, names = ['input'], prefix = '', postfix = '_#%d' % i_sample, save_dir = case_out_dir)
        
        if 'orig' in sample:
            utils.viewVolume(sample['orig'], aff = aff, names = ['orig'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

        if 'source' in sample:
            utils.viewVolume(sample['source'], aff = aff, names = ['source'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(sample['target'], aff = aff, names = ['target'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(outputs[i_sample]['tgt_def'], aff = aff, names = ['source'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(outputs[i_sample]['src_def'], aff = aff, names = ['target'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

        if 'label' in outputs[i_sample]:
            utils.viewVolume(outputs[i_sample]['label'], aff = aff, names = ['label'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir)

        if 'image' in outputs[i_sample]:
            utils.viewVolume(outputs[i_sample]['image'], aff = aff, names = ['image'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

    return results



def train_one_epoch(epoch, gen_args, train_args, model, processors, criterion, data_loader_dict, 
                            scaler, optimizer, lr_scheduler, wd_scheduler, 
                            postprocessor, visualizers, output_dir, device = 'cpu'):
    
    model.train()
    criterion.train()

    seed = int(time.time()) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    random.seed(seed)

    metric_logger = utils.MetricLogger(
        train_args.log_itr,
        delimiter="  ",
        debug=train_args.debug)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, train_args.n_epochs)

    max_len = max([len(v) for v in data_loader_dict.values()])
    probs = probs if gen_args.dataset_probs else [1/len(data_loader_dict)] * len(data_loader_dict)

    for itr, (dataset_num, curr_dataset, input_mode, target, samples) in enumerate(metric_logger.log_every(data_loader_dict, max_len, probs, epoch, header=header, train_limit=train_args.train_itr_limit)): 
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # update weight decay and learning rate according to their schedule
            curr_itr = max_len * epoch + itr  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups): 
                param_group["lr"] = lr_scheduler[curr_itr]
                param_group["weight_decay"] = wd_scheduler[curr_itr]

            samples = utils.nested_dict_to_device(samples, device)
            target = utils.nested_dict_to_device(target, device)

            cond = []
            if train_args.condition is not None:
                for i in range(len(samples)):
                    curr_cond = None
                    if 'mask' in train_args.condition:
                        samples[i]['input'] *= 1 - target['pathology'] # mask out anomaly # (b, 1, s, r, c)
                        curr_cond = target['pathology'].to(samples[0]['input'].dtype)
                    if 'flip' in train_args.condition:
                        samples[i]['input_flip'] = torch.flip(samples[i]['input'], dims = [2])
                        curr_cond = torch.concat([samples[i]['input_flip'], curr_cond], dim = 1) if curr_cond is not None else samples[i]['input_flip']
                    cond.append(curr_cond)
            
            outputs, _ = model(samples, cond = cond)
            for processor in processors:
                outputs = processor(outputs, target, curr_dataset)
            
            loss_dict = criterion(outputs, target, samples) 

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            try:
                loss_value = losses_reduced_scaled.item()
            except:
                logger.info('This iteration does not have any loss applicable, skipping')
                torch.cuda.empty_cache() 
                continue
            
            if not math.isfinite(loss_value):
                #logger.info(f"Loss is {loss_value}, stopping training")
                logger.info(f"Loss is {loss_value}, skipping this iteration")
                logger.info(loss_dict_reduced)
                logger.info(f"Case is {curr_dataset} - {target['name']}, skipping this iteration")
                #sys.exit(1)
                torch.cuda.empty_cache() 
                continue

        #losses.backward() # old
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        if train_args.clip_max_norm > 0:
            utils.clip_gradients(model, train_args.clip_max_norm)
        utils.cancel_gradients_last_layer(epoch, model, train_args.freeze_last_layer)
        #optimizer.step() # old
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        if utils.get_world_size() > 1:
            torch.cuda.synchronize()
        metric_logger.update(loss = loss_value,
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled
                            )
        metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.update(wd = optimizer.param_groups[0]["weight_decay"]) 

        if train_args.debug or (itr % train_args.vis_itr < dataset_num) and visualizers is not None and utils.is_main_process(): 
            vis_itr = itr - itr % train_args.vis_itr
            epoch_vis_dir = utils.make_dir(os.path.join(output_dir, str(epoch), str(vis_itr), curr_dataset + '-' + input_mode)) if epoch is not None else output_dir

            if postprocessor is not None:
                outputs, samples, target = postprocessor(gen_args, train_args, outputs, samples, target = target, feats = None, tasks = gen_args.tasks)
            if train_args.visualizer.make_results:  
                make_results(target, samples, outputs, out_dir = epoch_vis_dir)

            visualizers['result'].visualize_all(target, samples, outputs, epoch_vis_dir, 
                                                output_names = train_args.output_names + train_args.aux_output_names, target_names = train_args.target_names) 
            #if 'feature' in visualizers:
            #    visualizers['feature'].visualize_all_multi(target, samples, outputs, epoch_vis_dir)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger)) 

    if train_args.debug:
        exit()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  


 

def train_one_epoch_twostage(epoch, gen_args, train_args, pathol_model, task_model, pathol_processors, task_processors, 
                            criterion, data_loader_dict, scaler, optimizer, lr_scheduler, wd_scheduler, 
                            postprocessor, visualizers, output_dir, device = 'cpu'):
    
    pathol_model.train()
    task_model.train()
    criterion.train()
    
    seed = int(time.time()) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    random.seed(seed)

    metric_logger = utils.MetricLogger(
        train_args.log_itr,
        delimiter="  ",
        debug=train_args.debug)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, train_args.n_epochs)

    max_len = max([len(v) for v in data_loader_dict.values()])
    probs = probs if gen_args.dataset_probs else [1/len(data_loader_dict)] * len(data_loader_dict)

    for itr, (dataset_num, curr_dataset, input_mode, target, samples) in enumerate(metric_logger.log_every(data_loader_dict, max_len, probs, epoch, header=header, train_limit=train_args.train_itr_limit)): 
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # update weight decay and learning rate according to their schedule
            curr_itr = max_len * epoch + itr  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups): 
                param_group["lr"] = lr_scheduler[curr_itr]
                param_group["weight_decay"] = wd_scheduler[curr_itr]

            samples = utils.nested_dict_to_device(samples, device)
            target = utils.nested_dict_to_device(target, device)

            # stage-0: pathology segmentation prediction
            outputs_pathol, _ = pathol_model(samples)
            for processor in pathol_processors:
                outputs_pathol = processor(outputs_pathol, target, curr_dataset)
                
            # stage-1: pathology-mask-conditioned inpainting tasks prediction
            cond = []
            for i in range(len(samples)):
                samples[i]['input_masked'] = samples[i]['input'] * (1 - outputs_pathol[i]['pathology']) # mask out anomaly # (b, 1, s, r, c)
                curr_cond = target['pathology'].to(samples[0]['input'].dtype)
                if 'flip' in train_args.condition:
                    samples[i]['input_flip'] = torch.flip(samples[i]['input'], dims = [2])
                    curr_cond = torch.concat([samples[i]['input_flip'], curr_cond], dim = 1)
                cond.append(curr_cond)

            outputs_task, _ = task_model(samples, input_name = 'input_masked', cond = cond)
            for processor in task_processors:
                outputs_task = processor(outputs_task, target, curr_dataset)

            outputs = utils.merge_list_of_dict(outputs_task, outputs_pathol) 
            loss_dict = criterion(outputs, target, samples) 

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            try:
                loss_value = losses_reduced_scaled.item()
            except:
                logger.info('This iteration does not have any loss applicable, skipping')
                torch.cuda.empty_cache() 
                continue
            if not math.isfinite(loss_value):
                #logger.info(f"Loss is {loss_value}, stopping training")
                logger.info(f"Loss is {loss_value}, skipping this iteration")
                logger.info(loss_dict_reduced)
                logger.info(f"Case is {curr_dataset} - {target['name']}, skipping this iteration")
                #sys.exit(1)
                torch.cuda.empty_cache() 
                continue


        #losses.backward() # old
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        if train_args.clip_max_norm > 0:
            utils.clip_gradients(pathol_model, train_args.clip_max_norm)
            utils.clip_gradients(task_model, train_args.clip_max_norm)
        #optimizer.step() # old
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        if utils.get_world_size() > 1:
            torch.cuda.synchronize()
        metric_logger.update(loss = loss_value,
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled
                            )
        metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.update(wd = optimizer.param_groups[0]["weight_decay"]) 

        if train_args.debug or (itr % train_args.vis_itr < dataset_num) and visualizers is not None and utils.is_main_process():  
            vis_itr = itr - itr % train_args.vis_itr
            epoch_vis_dir = utils.make_dir(os.path.join(output_dir, str(epoch), str(vis_itr), curr_dataset + '-' + input_mode)) if epoch is not None else output_dir

            if postprocessor is not None:
                outputs, samples, target = postprocessor(gen_args, train_args, outputs, samples, target = target, feats = None, tasks = gen_args.tasks) 

            visualizers['result'].visualize_all(target, samples, outputs, epoch_vis_dir, 
                                                output_names = train_args.output_names + train_args.aux_output_names, target_names = train_args.target_names) 
            #if 'feature' in visualizers:
            #    visualizers['feature'].visualize_all_multi(target, samples, outputs, epoch_vis_dir)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger)) 

    if train_args.debug:
        exit()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  

