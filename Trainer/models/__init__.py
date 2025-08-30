

"""
Submodule interface.
"""
import torch

from .backbone import build_backbone
from .criterion import *
from .evaluator import Evaluator
from .head import get_head
from .joiner import get_processors, get_joiner
import utils.misc as utils


#########################################

# some constants
label_list_segmentation_brainseg_left = [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 17, 31, 34, 36, 38, 40, 42]
n_labels_brainseg_left = len(label_list_segmentation_brainseg_left)

label_list_segmentation_brainseg_with_extracerebral =  [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46,
                                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51, 53, 55,
                                   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50, 52, 54, 56] 
n_neutral_labels_brainseg_with_extracerebral = 20
n_labels_brainseg_with_extracerebral = len(label_list_segmentation_brainseg_with_extracerebral)
nlat = int((n_labels_brainseg_with_extracerebral - n_neutral_labels_brainseg_with_extracerebral) / 2.0)
vflip = np.concatenate([np.array(range(n_neutral_labels_brainseg_with_extracerebral)),
                        np.array(range(n_neutral_labels_brainseg_with_extracerebral + nlat, n_labels_brainseg_with_extracerebral)),
                        np.array(range(n_neutral_labels_brainseg_with_extracerebral, n_neutral_labels_brainseg_with_extracerebral + nlat))]) 

                        
############################################
############# helper functions #############
############################################

def process_args(gen_args, train_args, task):
    """
    task options: feat-anat, feat-seg, feat-anat-seg, anat, seg, reg, sr, bf
    """
    gen_args.tasks = [key for (key, value) in vars(task).items() if value]

    gen_args.generator.size = gen_args.generator.size # update real sample size (if sample_size is downsampled)
    train_args.size = gen_args.generator.size

    if gen_args.generator.left_hemis_only:
        gen_args.label_list_segmentation = label_list_segmentation_brainseg_left
        gen_args.n_labels = n_labels_brainseg_left
    else:
        gen_args.label_list_segmentation = label_list_segmentation_brainseg_with_extracerebral 
        gen_args.n_labels = n_labels_brainseg_with_extracerebral

    train_args.out_channels = {}
    train_args.output_names = []
    train_args.aux_output_names = []
    train_args.target_names = []
    if not 'contrastive' in gen_args.tasks: 
        if 'T1' in gen_args.tasks:  
            train_args.out_channels['T1'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['T1']
            train_args.target_names += ['T1']
            if train_args.losses.uncertainty is not None:
                train_args.aux_output_names += ['T1_sigma']
        if 'T2' in gen_args.tasks:  
            train_args.out_channels['T2'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['T2']
            train_args.target_names += ['T2']
            if train_args.losses.uncertainty is not None:
                train_args.aux_output_names += ['T2_sigma']
        if 'FLAIR' in gen_args.tasks:  
            train_args.out_channels['FLAIR'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['FLAIR']
            train_args.target_names += ['FLAIR']
            if train_args.losses.uncertainty is not None:
                train_args.aux_output_names += ['FLAIR_sigma']
        if 'CT' in gen_args.tasks:  
            train_args.out_channels['CT'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['CT']
            train_args.target_names += ['CT']
            if train_args.losses.uncertainty is not None: # TODO
                train_args.aux_output_names += ['CT_sigma']
        if 'bias_field' in gen_args.tasks:  
            train_args.out_channels['bias_field_log'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['bias_field']
            train_args.target_names += ['bias_field']
        if 'segmentation' in gen_args.tasks:  
            train_args.out_channels['segmentation'] = gen_args.n_labels
            train_args.output_names += ['label']
            train_args.target_names += ['label']
        if 'distance' in gen_args.tasks:  
            if gen_args.generator.left_hemis_only:
                train_args.out_channels['distance'] = 2
                train_args.output_names += ['distance', 'lp', 'lw']
                train_args.target_names += ['distance', 'lp', 'lw']
            else:
                train_args.out_channels['distance'] = 4
                train_args.output_names += ['distance', 'lp', 'lw', 'rp', 'rw']
                train_args.target_names += ['distance', 'lp', 'lw', 'rp', 'rw']
        if 'registration' in gen_args.tasks:  
            train_args.out_channels['registration'] = 3
            train_args.output_names += ['registration', 'regx', 'regy', 'regz']
            train_args.target_names += ['registration', 'regx', 'regy', 'regz']
        if 'surface' in gen_args.tasks:  
            train_args.out_channels['surface'] = 8
            train_args.output_names += ['surface']
            train_args.target_names += ['surface']
        if 'super_resolution' in gen_args.tasks:
            train_args.out_channels['high_res_residual'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['high_res', 'high_res_residual']
            train_args.target_names += ['high_res', 'high_res_residual']
        if 'pathology' in gen_args.tasks:
            train_args.out_channels['pathology'] = 1 
            train_args.output_names += ['pathology']
            train_args.target_names += ['pathology']

        if 'age' in gen_args.tasks:
            train_args.out_channels['age'] = -1  

        if train_args.losses.implicit_pathol: # TODO
            train_args.output_names += ['implicit_pathol_orig']
            train_args.output_names += ['implicit_pathol_pred']
            
        #assert len(train_args.output_names) > 0

    return gen_args, train_args

############################################
################ CRITERIONS ################
############################################

def get_evaluator(args, task, device):
    """
    task options: sr, seg, anat, reg
    """
    metric_names = []
    if 'T1' in task or 'T2' in task or 'FLAIR' in task or 'CT' in task:
        metric_names += ['feat_ssim', 'feat_ms_ssim', 'feat_l1']
    else:
        if 'T1' in task: # TODO
            metric_names += ['recon_l1', 'recon_psnr', 'recon_ssim', 'recon_ms_ssim']
        if 'super_resolution' in task:
            metric_names += ['sr_l1', 'sr_psnr', 'sr_ssim', 'sr_ms_ssim']
        if 'bias_field' in task: 
            metric_names += ['bf_normalized_l2', 'bf_corrected_l1']
        if 'segmentation' in task:
            metric_names += ['seg_dice']
        if 'pathology' in task:
            metric_names += ['pathol_dice']
        
    assert len(metric_names) > 0

    evaluator = Evaluator(
        args = args,
        metric_names = metric_names, 
        device = device,
        )
        
    return evaluator



def get_criterion(gen_args, train_args, tasks, device, exclude_keys = []):
    """
    task options: sr, seg, anat, reg
    """
    loss_names = []
    weight_dict = {}

    if 'contrastive' in tasks: 
        loss_names += ['contrastive']
        weight_dict['loss_contrastive'] = train_args.weights.contrastive
        return SetCriterion(
            gen_args = gen_args,
            train_args = train_args,
            weight_dict = weight_dict,
            loss_names = loss_names, 
            device = device,
            )
    
 
    for task in tasks: 

        if 'T1' in task or 'T2' in task or 'FLAIR' in task or 'CT' in task: 
            name = task

            loss_names += [name]
            weight_dict.update({'loss_%s' % name: train_args.weights.image})
            if train_args.losses.image_grad:
                loss_names += ['%s_grad' % name]
                weight_dict['loss_%s_grad' % name] = train_args.weights.image_grad 

        if 'segmentation' in task:
            loss_names += ['seg_ce', 'seg_dice']
            weight_dict.update( {
                'loss_seg_ce': train_args.weights.seg_ce,
                'loss_seg_dice': train_args.weights.seg_dice,
            } )
        
        if 'bias_field' in task:
            loss_names += ['bias_field_log']
            weight_dict.update( {
                'loss_bias_field_log': train_args.weights.bias_field_log, 
            } )
        
        if 'super_resolution' in task:
            loss_names += ['SR']
            weight_dict.update( {
                'loss_SR': train_args.weights.image, 
            } )
            if train_args.losses.image_grad:
                loss_names += ['SR_grad']
                weight_dict['loss_SR_grad'] = train_args.weights.image_grad 

        if 'distance' in task:
            loss_names += ['distance']
            weight_dict.update( {
                'loss_distance': train_args.weights.distance, 
            } )  

        if 'registration' in task:
            loss_names += ['registration'] 
            weight_dict.update( {
                'loss_registration': train_args.weights.registration, 
            } ) 
            if train_args.losses.registration_grad:
                loss_names += ['registration_grad']
                weight_dict['loss_registration_grad'] = train_args.weights.registration_grad
            if train_args.losses.registration_smooth:
                loss_names += ['registration_smooth']
                weight_dict['loss_registration_smooth'] = train_args.weights.registration_smooth
            if train_args.losses.registration_hessian:
                loss_names += ['registration_hessian']
                weight_dict['loss_registration_hessian'] = train_args.weights.registration_hessian

        if 'surface' in task:
            loss_names += ['surface']
            weight_dict['loss_surface'] = train_args.weights.surface

        if 'age' in task:
            loss_names += ['age']
            weight_dict['loss_age'] = train_args.weights.age

        if 'pathology' in task and 'pathology' not in exclude_keys:
            loss_names += ['pathol_ce', 'pathol_dice']
            weight_dict.update( {
                'loss_pathol_ce': train_args.weights.pathol_ce,
                'loss_pathol_dice': train_args.weights.pathol_dice,
            } )

    if train_args.losses.implicit_pathol: 
        loss_names += ['implicit_pathol_ce', 'implicit_pathol_dice']
        weight_dict.update( {
            'loss_implicit_pathol_ce': train_args.weights.implicit_pathol_ce,
            'loss_implicit_pathol_dice': train_args.weights.implicit_pathol_dice,
        } )
        
    assert len(loss_names) > 0 

    criterion = SetMultiCriterion(
        gen_args = gen_args,
        train_args = train_args,
        weight_dict = weight_dict,
        loss_names = loss_names, 
        device = device,
        )
        
    return criterion




def get_postprocessor(gen_args, train_args, outputs, samples, target, feats, tasks):
    """
    output: list of output dict 
    feat: list of output dict from pre-trained feat extractor
    """

    if 'distance' in tasks and target is not None:
        if gen_args.generator.left_hemis_only:
            target.update({'lp': target['distance'][:, 0][:, None],
                            'lw': target['distance'][:, 1][:, None]}) 
        else:
            target.update({'lp': target['distance'][:, 0][:, None],
                            'lw': target['distance'][:, 1][:, None],
                            'rp': target['distance'][:, 2][:, None],
                            'rw': target['distance'][:, 3][:, None]}) 
        del target['distance']

    if 'registration' in tasks and target is not None:
        target.update({'regx': target['registration'][:, 0][:, None],
                        'regy': target['registration'][:, 1][:, None], 
                        'regz': target['registration'][:, 2][:, None]})  
        del target['registration'] 

    if 'CT' in tasks and target is not None:
        target['CT'] = target['CT'] * 1000 

    if 'segmentation' in tasks and target is not None:
        target['label'] = torch.tensor(gen_args.label_list_segmentation, 
                                        device = target['segmentation'].device)[torch.argmax(target['segmentation'], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 

    for i, output in enumerate(outputs): 

        if feats is not None:
            output.update({'feat': feats[i]['feat']})  

        if 'super_resolution' in tasks:
            output.update({'high_res': output['high_res_residual'] + samples[i]['input']})  
            if 'high_res_residual' in samples[i]:
                samples[i].update({'high_res': samples[i]['high_res_residual'] + samples[i]['input']}) 
        
        if 'bias_field' in tasks:
            output.update({'bias_field': torch.exp(output['bias_field_log'])})
            del output['bias_field_log']

            if 'bias_field_log' in samples[i]:
                samples[i].update({'bias_field': torch.exp(samples[i]['bias_field_log'])})
                del samples[i]['bias_field_log']
        
        if 'distance' in tasks:

            a = 2 

            if gen_args.generator.left_hemis_only:
                output.update({'lp': output['distance'][:, 0][:, None],
                                'lw': output['distance'][:, 1][:, None]}) 
                fake = 70 * (1 - (torch.tanh(a * (output['lw'] + 0.3)) + 1) / 2) + 40 * (1 - (torch.tanh(a * output['lp']) + 1) / 2)
            else:
                output.update({'lp': output['distance'][:, 0][:, None],
                            'lw': output['distance'][:, 1][:, None],
                            'rp': output['distance'][:, 2][:, None],
                            'rw': output['distance'][:, 3][:, None]}) 
                
                fakeL = 70 * (1 - (torch.tanh(a * (output['lw'] + 0.3)) + 1) / 2) + 40 * (1 - (torch.tanh(a * output['lp']) + 1) / 2)
                fakeR = 70 * (1 - (torch.tanh(a * (output['rw'] + 0.3)) + 1) / 2) + 40 * (1 - (torch.tanh(a * output['rp']) + 1) / 2)
                fake = fakeL + fakeR

            output.update({'fake_cortical': fake})
            del output['distance']
        
        if 'registration' in tasks:
            output.update({'regx': output['registration'][:, 0][:, None],
                           'regy': output['registration'][:, 1][:, None], 
                           'regz': output['registration'][:, 2][:, None]}) 
            del output['registration']

        if 'segmentation' in tasks: 
            output['label'] = torch.tensor(gen_args.label_list_segmentation, 
                                         device = output['segmentation'].device)[torch.argmax(output['segmentation'], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 
        
        if 'CT' in tasks:
            output['CT'] = output['CT'] * 1000 

    return outputs, samples, target


#############################################
################ OPTIMIZERS #################
#############################################


def build_optimizer(train_args, params_groups):
    if train_args.optimizer == "adam":
        return torch.optim.Adam(params_groups)  
    elif train_args.optimizer == "adamw":
        return torch.optim.AdamW(params_groups)  # to use with ViTs
    elif train_args.optimizer == "sgd":
        return torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif train_args.optimizer == "lars":
        return utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        ValueError('optim type {args.optimizer.type} supported!')


def build_schedulers(train_args, itr_per_epoch, lr, min_lr):
    if train_args.lr_scheduler == "cosine":
        lr_scheduler = utils.cosine_scheduler(
            lr, # * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
            min_lr,
            train_args.n_epochs, itr_per_epoch,
            warmup_epochs=train_args.warmup_epochs
        )
    elif train_args.lr_scheduler == "multistep":
        lr_scheduler = utils.multistep_scheduler(
            lr, 
            train_args.lr_drops, 
            train_args.n_epochs, itr_per_epoch, 
            warmup_epochs=train_args.warmup_epochs, 
            gamma=train_args.lr_drop_multi
            )  
    wd_scheduler = utils.cosine_scheduler(
        train_args.weight_decay, # set as 0 to disable it
        train_args.weight_decay_end,
        train_args.n_epochs, itr_per_epoch
        )
    return lr_scheduler, wd_scheduler


############################################
################## MODELS ##################
############################################


def build_model(gen_args, train_args, device = 'cpu'):
    gen_args, train_args = process_args(gen_args, train_args, task = gen_args.task)

    backbone = build_backbone(train_args, train_args.backbone)
    head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1)
    model = get_joiner(gen_args.tasks, backbone, head, device) 

    processors = get_processors(gen_args, train_args, gen_args.tasks, device)

    criterion = get_criterion(gen_args, train_args, gen_args.tasks, device)
    
    criterion.to(device)

    model.to(device)
    postprocessor = get_postprocessor

    return gen_args, train_args, model, processors, criterion, postprocessor


def build_conditioned_model(gen_args, train_args, device = 'cpu'): # mask-conditioned inpaiting
    gen_args, train_args = process_args(gen_args, train_args, task = gen_args.task)

    backbone = build_backbone(train_args, train_args.backbone, num_cond = len(train_args.condition.split('+')))
    head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1, stage = 1, exclude_keys = ['pathology'])
    model = get_joiner(gen_args.tasks, backbone, head, device)
    processors = get_processors(gen_args, train_args, gen_args.tasks, device, exclude_keys = ['pathology'])

    criterion = get_criterion(gen_args, train_args, gen_args.tasks, device, exclude_keys = ['pathology'])
    criterion.to(device)

    model.to(device)
    postprocessor = get_postprocessor

    return gen_args, train_args, model, processors, criterion, postprocessor



def build_inpaint_model(gen_args, train_args, device = 'cpu'): # two-stage inpainting
    gen_args, train_args = process_args(gen_args, train_args, task = gen_args.task)

    # stage-0: pathology mask prediction
    pathol_backbone = build_backbone(train_args, train_args.backbone.split('+')[0], num_cond = 0)
    pathol_head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1, stage = 0)
    pathol_model = get_joiner(gen_args.tasks, pathol_backbone, pathol_head, device, postfix = '_pathol')
    pathol_processors = get_processors(train_args, ['pathology'], device) 

    # stage-1: pathology-mask-conditioned task prediction (inpainting)  
    task_backbone = build_backbone(train_args, train_args.backbone.split('+')[1], num_cond = 1)
    task_head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1, stage = 1)
    task_model = get_joiner(gen_args.tasks, task_backbone, task_head, device, postfix = '_task')
    task_processors = get_processors(gen_args, train_args, gen_args.tasks, device, exclude_keys = ['pathology'])

    criterion = get_criterion(gen_args, train_args, gen_args.tasks, device)
    criterion.to(device)

    pathol_model.to(device)
    task_model.to(device)
    postprocessor = get_postprocessor

    return gen_args, train_args, pathol_model, task_model, pathol_processors, task_processors, criterion, postprocessor

 