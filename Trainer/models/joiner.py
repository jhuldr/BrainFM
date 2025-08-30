

"""
Wrapper interface.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

from Trainer.models.unet3d.model import UNet3D
from .head import TaskHead, MultiInputTaskHead
from utils.checkpoint import load_checkpoint 


#supersynth_ckp_path = '/autofs/space/yogurt_002/users/pl629/ckp/wmh-synthseg/PAPER_checkpoint_0101.pth'
supersynth_ckp_path = '/autofs/space/yogurt_002/users/pl629/ckp/wmh-synthseg/AllDataIn_checkpoint_0101.pth'

flair2pathol_feat_ckp_path = '/autofs/space/yogurt_002/users/pl629/ckp/Supv/supv_adni3_flair2pathol_feat_epoch_35.pth' 
flair2pathol_task_ckp_path = '/autofs/space/yogurt_002/users/pl629/ckp/Supv/supv_adni3_flair2pathol_epoch_35.pth' 



def build_supersynth_model(device = 'cpu'):
    # 33 + 4 + 1 + 1 = 39 (SuperSynth)
    backbone = UNet3D(1, f_maps=64, layer_order='gcl', num_groups=8, num_levels=5, is3d=True)
    head = TaskHead(None, f_maps_list = [64], out_channels ={'segmentation': 39}, is_3d = True, out_feat_level = -1)
    model = get_joiner('segmentation', backbone, head) 
    processor = SegProcessor().to(device)
    model.to(device)
    return model, processor


def build_pathol_model(device = 'cpu'):
    backbone = UNet3D(1, f_maps=64, layer_order='gcl', num_groups=8, num_levels=5, is3d=True)
    feat_model = get_joiner('segmentation', backbone, None) 
    task_model = MultiInputTaskHead(None, [64], {'pathology': 1}, True, -1)
    processor = PatholProcessor().to(device)
    feat_model.to(device)
    task_model.to(device)
    return feat_model, task_model, processor




class UncertaintyProcessor(nn.Module):
    def __init__(self, output_names):
        super(UncertaintyProcessor, self).__init__()
        self.output_names = output_names

    def forward(self, outputs, *kwargs): 
        for output_name in self.output_names:
            if 'image' in output_name:
                for output in outputs:
                    output[output_name + '_sigma'] = output[output_name][:, 1][:, None]
                    output[output_name] = output[output_name][:, 0][:, None]
        return outputs
    
class AgeProcessor(nn.Module):
    def __init__(self):
        super(AgeProcessor, self).__init__() 

    def forward(self, outputs, *kwargs): 
        for output in outputs:  
            #output['age'] = output['age'] ** 2
            output['age'] = abs(output['age'])
        return outputs
    

class SegProcessor(nn.Module):
    def __init__(self):
        super(SegProcessor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['segmentation'] = self.softmax(output['segmentation'])
        return outputs
    
class PatholProcessor(nn.Module):
    def __init__(self):
        super(PatholProcessor, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['pathology'] = self.sigmoid(output['pathology'])
        return outputs
 

class PatholSeg(nn.Module):
    def __init__(self, args):
        super(PatholSeg, self).__init__()
        self.sigmoid = nn.Sigmoid()

        paths = args.supervised_pathol_seg_ckp_path
        self.feat_model, self.task_model, self.processor = build_pathol_model()
        load_checkpoint(paths.feat, [self.feat_model], model_keys = ['model'], to_print = False)
        load_checkpoint(paths.task, [self.task_model], model_keys = ['model'], to_print = False)
        for param in self.feat_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
            param.requires_grad = False
        for param in self.task_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
            param.requires_grad = False

        aux_paths = args.supervised_aux_pathol_seg_ckp_path
        if args.aux_modality is not None:
            self.aux_feat_model, self.aux_task_model, self.aux_processor = build_pathol_model()
            load_checkpoint(aux_paths.feat, [self.aux_feat_model], model_keys = ['model'], to_print = False)
            load_checkpoint(aux_paths.task, [self.aux_task_model], model_keys = ['model'], to_print = False)
            for param in self.aux_feat_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
                param.requires_grad = False
            for param in self.aux_task_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
                param.requires_grad = False
        else:
            self.aux_feat_model, self.aux_task_model, self.aux_processor = None, None, None

    def forward(self, outputs, target, curr_dataset, *kwargs): 
        for output in outputs:
            if output['image'].shape == target['image'].shape:
                samples = [ { 'input': output['image'] },  { 'input': target['image'] } ]
                feats, inputs = self.feat_model(samples)
                preds = self.task_model([feat['feat'] for feat in feats], inputs)
                preds = self.processor(preds, samples)
                output['implicit_pathol_pred'] = preds[0]['pathology']
                output['implicit_pathol_orig'] = preds[1]['pathology'] 
            if self.aux_feat_model is not None:
                if output['aux_image'].shape == target['aux_image'].shape:
                    samples = [ { 'input': output['aux_image'] }, { 'input': target['aux_image'] } ]
                    feats, inputs = self.aux_feat_model(samples)
                    preds = self.aux_task_model([feat['feat'] for feat in feats], inputs)
                    preds = self.processor(preds, samples)
                    output['implicit_aux_pathol_pred'] = preds[0]['pathology']
                    output['implicit_aux_pathol_orig'] = preds[1]['pathology'] 
        return outputs
    
    
class ContrastiveProcessor(nn.Module):
    def __init__(self):
        '''
        Ref: https://openreview.net/forum?id=2oCb0q5TA4Y
        '''
        super(ContrastiveProcessor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, outputs, *kwargs):
        for output in outputs:
            output['feat'][-1] = F.normalize(output['feat'][-1], dim = 1)
        return outputs 

class DistProcessor(nn.Module):
    def __init__(self, gen_args):
        super(DistProcessor, self).__init__()
        self.gen_args = gen_args

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['distance'] = torch.clamp(output['distance'], min = - self.gen_args.max_surf_distance, max = self.gen_args.max_surf_distance)
        return outputs


##############################################################################


class MultiInputIndepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    """
    def __init__(self, backbone, head, device, postfix = ''):
        super(MultiInputIndepJoiner, self).__init__()

        self.backbone = backbone.to(device)
        self.head = head.to(device)
        self.postfix = postfix

    def forward(self, input_list, input_name = 'input', cond = []):
        outs = []
        for i, x in enumerate(input_list):  
            if len(cond) > 0: 
                feat = self.backbone.get_feature(torch.concat([x[input_name], cond[i]], dim = 1))
            else:
                feat = self.backbone.get_feature(x[input_name])
            out = {'feat' + self.postfix: feat}
            if self.head is not None: 
                out.update( self.head(feat) )
            outs.append(out)
        return outs, [input[input_name] for input in input_list]


class MultiInputSepDecIndepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    NOTE: keys in head_dict must equal feat_dict
    """
    def __init__(self, backbone, head_dict, device):
        super(MultiInputSepDecIndepJoiner, self).__init__()

        self.backbone = backbone.to(device)
        self.head_dict = head_dict
        for k in self.head_dict.keys():
            self.head_dict[k].to(device)

    def forward(self, input_list):
        outs = []
        for x in input_list:  
            feat_dict = self.backbone.get_feature(x['input'])
            out = {'feat_%s' % k: feat_dict[k] for k in feat_dict.keys()}
            for k in feat_dict.keys():
                if self.head_dict is not None: 
                    out.update( self.head_dict[k](feat_dict[k]) )
            outs.append(out)
        return outs, [input['input'] for input in input_list]
    

class MultiInputDepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    """
    def __init__(self, backbone, head, device):
        super(MultiInputDepJoiner, self).__init__()

        self.backbone = backbone.to(device)
        self.head = head.to(device)

    def forward(self, input_list):
        outs = []
        for x in input_list:  
            feat = self.backbone.get_feature(x['input'])
            out = {'feat': feat}  
            if self.head is not None: 
                out.update( self.head( feat, x) )
            outs.append(out) 
        return outs, [input['input'] for input in input_list]



################################


def get_processors(gen_args, train_args, tasks, device, exclude_keys = []): 
    processors = []
    if train_args.losses.uncertainty is not None:
        processors.append(UncertaintyProcessor(train_args.output_names).to(device))
    if train_args.losses.implicit_pathol:
        processors.append(PatholSeg(train_args).to(device))
        
    if 'contrastive' in tasks:
        processors.append(ContrastiveProcessor().to(device))
    if 'age' in tasks:
        processors.append(AgeProcessor().to(device))
    if 'segmentation' in tasks and 'segmentation' not in exclude_keys:
        processors.append(SegProcessor().to(device))
    if 'distance' in tasks:
        processors.append(DistProcessor(gen_args).to(device))
    if 'pathology' in tasks and 'pathology' not in exclude_keys:  
        processors.append(PatholProcessor().to(device)) 

    return processors





def get_joiner(task, backbone, head, device, postfix = ''):
    if isinstance(head, dict): 
        return get_sep_joiner(task, backbone, head, device)

    return MultiInputIndepJoiner(backbone, head, device, postfix = postfix)
    



def get_sep_joiner(task, backbone, head_dict, device):
    return MultiInputSepDecIndepJoiner(backbone, head_dict, device)