"""
Criterion modules.
"""

import numpy as np
import torch
import torch.nn as nn

from Trainer.models.losses import GradientLoss, SmoothnessLoss, HessianLoss, gaussian_loss, laplace_loss, l1_loss
from utils.misc import viewVolume

uncertainty_loss = {'gaussian': gaussian_loss, 'laplace': laplace_loss}


class SetCriterion(nn.Module):
    """ 
    This class computes the loss for BrainID.
    """
    def __init__(self, gen_args, train_args, weight_dict, loss_names, device):
        """ Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetCriterion, self).__init__()
        self.gen_args = gen_args
        self.train_args = train_args
        self.weight_dict = weight_dict
        self.loss_names = loss_names 
 
        self.mse = nn.MSELoss()

        self.loss_regression_type = train_args.losses.uncertainty if train_args.losses.uncertainty is not None else 'l1' 
        self.loss_regression = uncertainty_loss[train_args.losses.uncertainty] if train_args.losses.uncertainty is not None else l1_loss
        
        self.grad = GradientLoss('l1')
        self.smoothness = SmoothnessLoss('l2')
        self.hessian = HessianLoss('l2')

        self.bflog_loss = nn.L1Loss() if train_args.losses.bias_field_log_type == 'l1' else self.mse

        if 'contrastive' in self.loss_names:
            self.temp_alpha = train_args.contrastive_temperatures.alpha
            self.temp_beta = train_args.contrastive_temperatures.beta
            self.temp_gamma = train_args.contrastive_temperatures.gamma
        
        # initialize weights # NOTE all = 1 for now
        weights_brainseg = torch.ones(gen_args.n_labels).to(device)
        weights_brainseg[gen_args.label_list_segmentation_with_csf==77] = train_args.relative_weight_lesions # give (more) importance to lesions
        weights_brainseg = weights_brainseg / torch.sum(weights_brainseg) 

        self.weights_ce = weights_brainseg[None, :, None, None, None]
        self.weights_dice = weights_brainseg[None, :] 

        # archived
        #self.csf_ind = torch.tensor(np.where(np.array(gen_args.label_list_segmentation)==24)[0][0])
        #self.csf_v = torch.tensor(np.concatenate([np.arange(0, self.csf_ind), np.arange(self.csf_ind+1, gen_args.n_labels)]))  

        self.loss_map = {
            'seg_ce': self.loss_seg_ce,
            'seg_dice': self.loss_seg_dice,
            'pathol_ce': self.loss_pathol_ce,
            'pathol_dice': self.loss_pathol_dice,
            'implicit_pathol_ce': self.loss_implicit_pathol_ce,
            'implicit_pathol_dice': self.loss_implicit_pathol_dice,
            'implicit_aux_pathol_ce': self.loss_implicit_aux_pathol_ce,
            'implicit_aux_pathol_dice': self.loss_implicit_aux_pathol_dice, 

            'T1': self.loss_T1,
            'T1_grad': self.loss_T1_grad,
            'T2': self.loss_T2,
            'T2_grad': self.loss_T2_grad,
            'FLAIR': self.loss_FLAIR,
            'FLAIR_grad': self.loss_FLAIR_grad,
            'CT': self.loss_CT,
            'CT_grad': self.loss_CT_grad, 
            'SR': self.loss_SR,
            'SR_grad': self.loss_SR_grad,

            "age": self.loss_age,
            "distance": self.loss_distance,
            "registration": self.loss_registration,
            "registration_grad": self.loss_registration_grad,
            "registration_hessian": self.loss_registration_hessian,
            "registration_smooth": self.loss_registration_smooth,
            "bias_field_log": self.loss_bias_field_log,
            'contrastive': self.loss_feat_contrastive, 

            "surface": self.loss_surface, # TODO
            #'supervised_seg': self.loss_supervised_seg, # archived
        }

    def loss_feat_contrastive(self, outputs, *kwargs):
        """
        outputs: [feat1, feat2]
        feat shape: (b, feat_dim, s, r, c)
        """
        feat1, feat2 = outputs[0]['feat'][-1], outputs[1]['feat'][-1]
        num = torch.sum(torch.exp(feat1 * feat2 / self.temp_alpha), dim = 1) 
        den = torch.zeros_like(feat1[:, 0]) 
        for i in range(feat1.shape[1]): 
            den1 = torch.exp(feat1[:, i] ** 2 / self.temp_beta)
            den2 = torch.exp((torch.sum(feat1[:, i][:, None] * feat1, dim = 1) - feat1[:, i] ** 2) / self.temp_gamma) 
            den += den1 + den2 
        loss_contrastive = torch.mean(- torch.log(num / den)) 
        return {'loss_contrastive': loss_contrastive}

    def loss_seg_ce(self, outputs, targets, *kwargs):
        """
        Cross entropy of segmentation
        """
        loss_seg_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['segmentation'], min=1e-5)) * self.weights_ce * targets['segmentation'], dim=1)) 
        return {'loss_seg_ce': loss_seg_ce}

    def loss_seg_dice(self, outputs, targets, *kwargs):
        """
        Dice of segmentation
        """
        loss_seg_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['segmentation'] * targets['segmentation']).sum(dim=[2, 3, 4])) 
                                                       / torch.clamp((outputs['segmentation'] + targets['segmentation']).sum(dim=[2, 3, 4]), min=1e-5)))
        return {'loss_seg_dice': loss_seg_dice}
    
    def loss_implicit_pathol_ce(self, outputs, targets, samples, *kwargs):
        """
        Cross entropy of pathology segmentation
        """
        if 'implicit_pathol_pred' in outputs:
            #loss_implicit_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_pathol_pred'], min=1e-5)) * self.weights_ce * samples['pathol'], dim=1)) 
            loss_implicit_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_pathol_pred'], min=1e-5)) * outputs['implicit_pathol_orig'], dim=1))
        else: # no GT image exists
            loss_implicit_pathol_ce = 0.
        return {'loss_implicit_pathol_ce': loss_implicit_pathol_ce}
    
    def loss_implicit_pathol_dice(self, outputs, targets, samples, *kwargs):
        """
        Dice of pathology segmentation
        """
        if 'implicit_pathol_pred' in outputs:
            #loss_implicit_pathol_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['implicit_pathol_pred'] * samples['pathol']).sum(dim=[2, 3, 4])) 
            #                                               / torch.clamp((outputs['implicit_pathol_pred'] + samples['pathol']).sum(dim=[2, 3, 4]), min=1e-5)))
            loss_implicit_pathol_dice = torch.sum((1.0 - 2.0 * ((outputs['implicit_pathol_pred'] * outputs['implicit_pathol_orig']).sum(dim=[2, 3, 4])) 
                                                        / torch.clamp((outputs['implicit_pathol_pred'] + outputs['implicit_pathol_orig']).sum(dim=[2, 3, 4]), min=1e-5)))
        else:
            loss_implicit_pathol_dice = 0.
        return {'loss_implicit_pathol_dice': loss_implicit_pathol_dice}


    def loss_implicit_aux_pathol_ce(self, outputs, targets, samples):
        """
        Cross entropy of pathology segmentation
        """
        if 'implicit_aux_pathol_pred' in outputs:
            #loss_implicit_aux_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_aux_pathol_pred'], min=1e-5)) * self.weights_ce * samples['pathol'], dim=1))  
            loss_implicit_aux_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_aux_pathol_pred'], min=1e-5)) * self.weights_ce * outputs['implicit_aux_pathol_orig'], dim=1))  
        else:
            loss_implicit_aux_pathol_ce = 0.
        return {'loss_implicit_aux_pathol_ce': loss_implicit_aux_pathol_ce}
    
    def loss_implicit_aux_pathol_dice(self, outputs, targets, samples):
        """
        Dice of pathology segmentation
        """
        if 'implicit_aux_pathol_pred' in outputs:
            #loss_implicit_aux_pathol_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['implicit_aux_pathol_pred'] * samples['pathol']).sum(dim=[2, 3, 4])) 
            #                                               / torch.clamp((outputs['implicit_aux_pathol_pred'] + samples['pathol']).sum(dim=[2, 3, 4]), min=1e-5))) 
            loss_implicit_aux_pathol_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['implicit_aux_pathol_pred'] * outputs['implicit_aux_pathol_orig']).sum(dim=[2, 3, 4])) 
                                                        / torch.clamp((outputs['implicit_aux_pathol_pred'] + outputs['implicit_aux_pathol_orig']).sum(dim=[2, 3, 4]), min=1e-5)))  
        else:
            loss_implicit_aux_pathol_dice = 0.
        return {'loss_implicit_aux_pathol_dice': loss_implicit_aux_pathol_dice}

    def loss_surface(self, outputs, targets, *kwargs):  
        return {'loss_surface': self.loss_image(outputs['surface'], targets['surface'])}
    
    def loss_distance(self, outputs, targets, *kwargs):  
        return {'loss_distance': self.loss_image(outputs['distance'], targets['distance'])}
    
    def loss_registration(self, outputs, targets, *kwargs):  
        return {'loss_registration': self.loss_image(outputs['registration'], targets['registration'])}
    
    def loss_registration_grad(self, outputs, targets, *kwargs): 
        return {'loss_registration_grad': self.loss_image_grad(outputs['registration'], targets['registration'])}
    
    def loss_registration_smooth(self, outputs, *kwargs): 
        return {'loss_registration_smooth': self.smoothness(outputs['registration'])}
    
    def loss_registration_hessian(self, outputs, *kwargs):
        return {'loss_registration_hessian': self.hessian(outputs['registration'])} 
    
    def loss_pathol_ce(self, outputs, targets, samples):
        """
        Cross entropy of pathology segmentation
        """
        if 'pathology' in outputs and outputs['pathology'].shape == targets['pathology'].shape:
            loss_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['pathology'], min=1e-5)) * targets['pathology'], dim=1))
        else:
            loss_pathol_ce = 0.
        return {'loss_pathol_ce': loss_pathol_ce}
    
    def loss_pathol_dice(self, outputs, targets, samples):
        """
        Dice of pathology segmentation
        """
        if 'pathology' in outputs and outputs['pathology'].shape == targets['pathology'].shape:
            loss_pathol_dice = torch.sum((1.0 - 2.0 * ((outputs['pathology'] * targets['pathology']).sum(dim=[2, 3, 4])) 
                                                        / torch.clamp((outputs['pathology'] + targets['pathology']).sum(dim=[2, 3, 4]), min=1e-5)))
        else:
            loss_pathol_dice = 0.
        return {'loss_pathol_dice': loss_pathol_dice}
    

    def loss_T1(self, outputs, targets, *kwargs): 
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T1'].shape else 1.
        weights = 1. - targets['T1_DM'] if 'T1_DM' in targets else 1. 
        #weights = 1.
        return {'loss_T1': self.loss_image(outputs['T1'], targets['T1'], outputs['T1_sigma'] if 'T1_sigma' in outputs else None, weights = weights)}
    def loss_T1_grad(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T1'].shape else 1.
        weights = 1. - targets['T1_DM'] if 'T1_DM' in targets else 1.
        #weights = 1.
        return {'loss_T1_grad': self.loss_image_grad(outputs['T1'], targets['T1'], weights)}
    
    def loss_T2(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T2'].shape else 1.
        weights = 1. - targets['T2_DM'] if 'T2_DM' in targets else 1.
        #weights = 1.
        return {'loss_T2': self.loss_image(outputs['T2'], targets['T2'], outputs['T2_sigma'] if 'T2_sigma' in outputs else None, weights)}
    def loss_T2_grad(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T2'].shape else 1.
        weights = 1. - targets['T2_DM'] if 'T2_DM' in targets else 1.
        #weights = 1.
        return {'loss_T2_grad': self.loss_image_grad(outputs['T2'], targets['T2'], weights)}
    
    def loss_FLAIR(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['FLAIR'].shape else 1. 
        weights = 1. - targets['FLAIR_DM'] if 'FLAIR_DM' in targets else 1.
        #weights = 1.
        return {'loss_FLAIR': self.loss_image(outputs['FLAIR'], targets['FLAIR'], outputs['FLAIR_sigma'] if 'FLAIR_sigma' in outputs else None, weights)}
    def loss_FLAIR_grad(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['FLAIR'].shape else 1. 
        weights = 1. - targets['FLAIR_DM'] if 'FLAIR_DM' in targets else 1.
        #weights = 1.
        return {'loss_FLAIR_grad': self.loss_image_grad(outputs['FLAIR'], targets['FLAIR'], weights)}
    
    def loss_CT(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['CT'].shape else 1. 
        weights = 1. - targets['CT_DM'] if 'CT_DM' in targets else 1.
        #weights = 1.
        return {'loss_CT': self.loss_image(outputs['CT'], targets['CT'], outputs['CT_sigma'] if 'CT_sigma' in outputs else None, weights)}
    def loss_CT_grad(self, outputs, targets, *kwargs):
        #weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['CT'].shape else 1.
        weights = 1. - targets['CT_DM'] if 'CT_DM' in targets else 1.
        #weights = 1.
        return {'loss_CT_grad': self.loss_image_grad(outputs['CT'], targets['CT'], weights)}
    
    def loss_SR(self, outputs, targets, samples): 
        loss_SR = self.loss_image(outputs['high_res_residual'], samples['high_res_residual'])
        return {'loss_SR': loss_SR}
    
    def loss_SR_grad(self, outputs, targets, samples): 
        loss_SR_grad = self.loss_image_grad(outputs['high_res_residual'], samples['high_res_residual'])
        return {'loss_SR_grad': loss_SR_grad}
    
    def loss_bias_field_log(self, outputs, targets, samples):
        if 'bias_field_log' in samples:
            bf_soft_mask = 1. - targets['segmentation'][:, 0]
            loss_bias_field_log = self.bflog_loss(outputs['bias_field_log'] * bf_soft_mask, samples['bias_field_log'] * bf_soft_mask)
        else:
            loss_bias_field_log = 0.
        return {'loss_bias_field_log': loss_bias_field_log} 
    
    
    def loss_age(self, outputs, targets, *kwargs): 
        loss_age = abs(outputs['age'] - targets['age']) 
        #print(outputs['age'].item(), outputs['age'].shape, targets['age'].item(), targets['age'].shape)
        return {'loss_age': loss_age}
    

    def loss_image(self, output, target, output_sigma = None, weights = 1., *kwargs): 
        if output.shape == target.shape:
            if output_sigma:
                loss_image = self.loss_regression(output, output_sigma, target)
            else:
                loss_image = self.loss_regression(output, target, weights)
        else: 
            loss_image = 0.
        return loss_image
    
    def loss_image_grad(self, output, target, weights = 1., *kwargs):
        return self.grad(output, target, weights) if output.shape == target.shape else 0. 

    
    def loss_supervised_seg(self, outputs, targets, *kwargs):
        """
        Supervised segmentation differences (for dataset_name == synth)
        """
        onehot_withoutcsf = targets['segmentation'].clone()
        onehot_withoutcsf = onehot_withoutcsf[:, self.csf_v, ...]
        onehot_withoutcsf[:, 0, :, :, :] = onehot_withoutcsf[:, 0, :, :, :] + targets['segmentation'][:, self.csf_ind, :, :, :]

        loss_supervised_seg = torch.sum(self.weights_dice_sup * (1.0 - 2.0 * ((outputs['supervised_seg'] * onehot_withoutcsf).sum(dim=[2, 3, 4])) 
                                                                 / torch.clamp((outputs['supervised_seg'] + onehot_withoutcsf).sum(dim=[2, 3, 4]), min=1e-5)))

        return {'loss_supervised_seg': loss_supervised_seg} 

    def get_loss(self, loss_name, outputs, targets, *kwargs):
        assert loss_name in self.loss_map, f'do you really want to compute {loss_name} loss?'
        return self.loss_map[loss_name](outputs, targets, *kwargs)

    def forward(self, outputs, targets, *kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs, targets, *kwargs))
        return losses
    


class SetMultiCriterion(SetCriterion):
    """ 
    This class computes the loss for BrainID with a list of results as inputs.
    """
    def __init__(self, gen_args, train_args, weight_dict, loss_names, device):
        """ Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetMultiCriterion, self).__init__(gen_args, train_args, weight_dict, loss_names, device)
        self.all_samples = gen_args.generator.all_samples

    def get_loss(self, loss_name, outputs_list, targets, samples_list):
        assert loss_name in self.loss_map, f'do you really want to compute {loss_name} loss?'
        total_loss = 0.
        for i_sample, outputs in enumerate(outputs_list): 
            total_loss += self.loss_map[loss_name](outputs, targets, samples_list[i_sample])['loss_' + loss_name]
        return {'loss_' + loss_name: total_loss / self.all_samples}
    
    def forward(self, outputs_list, targets, samples_list):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {} 
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs_list, targets, samples_list))
        return losses

