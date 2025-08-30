"""
Model heads
"""


import torch
import torch.nn as nn 
import torch.nn.functional as F


def removekey(d, keys):
    r = dict(d)
    for k in keys:
        del r[k]
    return r

        
class TaskHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs 
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, exclude_keys=[], *kwargs):
        super(TaskHead, self).__init__()
        self.out_feat_level = out_feat_level

        layers = [] # additional layers (same-size-output 3x3 conv) before final_conv, if len( f_maps_list ) > 1
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvBlock(in_feature_num, f_maps_list[i+1], stride = 1, is_3d = is_3d) 
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        conv = nn.Conv3d if is_3d else nn.Conv2d 
        fc = nn.Linear

        self.out_channels = removekey(out_channels, exclude_keys)
        self.out_names = self.out_channels.keys() 
        for out_name, out_channels_num in self.out_channels.items():
            if out_channels_num > 0:
                self.add_module("final_conv_%s" % out_name, conv(f_maps_list[-1], out_channels_num, 1))
            else: # single value output (age prediction)
                pool_layers = [nn.MaxPool3d(4, 4),  # (160 -> 40)
                               ConvBlock(f_maps_list[-1], 16, stride = 1, is_3d = is_3d), 
                               nn.MaxPool3d(4, 4), # (40 -> 10)
                               ConvBlock(16, 4, stride = 1, is_3d = is_3d)
                               ] 
                self.pool_layers = nn.ModuleList(pool_layers)
                self.add_module("final_linear1_%s" % out_name, fc(4 * args.size[0] // 16 * args.size[1] // 16 * args.size[2] // 16, 160, 1)) 
                self.add_module("final_linear2_%s" % out_name, fc(160, 10, 1)) 
                self.add_module("final_linear3_%s" % out_name, fc(10, - out_channels_num, 1)) 

    def forward(self, x, *kwargs):
        x = x[self.out_feat_level]
        for layer in self.layers:
            x = layer(x)
        out = {}
        for (name, n_channels) in self.out_channels.items(): 
            if n_channels > 0:
                out[name] = getattr(self, f"final_conv_{name}")(x)
            else:
                for layer in self.pool_layers:
                    x = layer(x) 
                x = torch.flatten(x, 1) # flatten all dimensions except batch  
                x = F.relu(getattr(self, f"final_linear1_{name}")(x)) 
                x = F.relu(getattr(self, f"final_linear2_{name}")(x)) 
                out[name] = torch.squeeze(getattr(self, f"final_linear3_{name}")(x), dim = 1) 
        return out
    

class DepHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-dependent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(DepHead, self).__init__()
        self.out_feat_level = out_feat_level

        f_maps_list[0] += 1 # add one input image/contrast channel

        layers = [] # additional layers (same-size-output 3x3 conv) before final_conv, if len( f_maps_list ) > 1
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvBlock(in_feature_num, f_maps_list[i+1], stride = 1, is_3d = is_3d) 
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        conv = nn.Conv3d if is_3d else nn.Conv2d 
        self.out_names = out_channels.keys()
        for out_name, out_channels_num in out_channels.items():
            self.add_module("final_conv_%s" % out_name, conv(f_maps_list[-1], out_channels_num, 1))

    def forward(self, x, image):
        x = x[self.out_feat_level]
        x = torch.cat([x, image],  dim = 1)
        for layer in self.layers:
            x = layer(x)
        out = {}
        for name in self.out_names: 
            out[name] = getattr(self, f"final_conv_{name}")(x)
        return out
    

    
class MultiInputDepHead(DepHead):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-dependent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(MultiInputDepHead, self).__init__(args, f_maps_list, out_channels, is_3d, out_feat_level)

    def forward(self, feat_list, image_list):   
        outs = []
        for i, x in enumerate(feat_list):
            x = x[self.out_feat_level] 
            x = torch.cat([x, image_list[i]],  dim = 1) 
            for layer in self.layers:
                x = layer(x)
            out = {}
            for name in self.out_names: 
                out[name] = getattr(self, f"final_conv_{name}")(x)
            outs.append(out)
        return outs
    


class MultiInputTaskHead(TaskHead):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-independent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(MultiInputTaskHead, self).__init__(args, f_maps_list, out_channels, is_3d, out_feat_level)

    def forward(self, feat_list, *kwargs):   
        outs = []
        for x in feat_list:
            x = x[self.out_feat_level]
            for layer in self.layers:
                x = layer(x)
            out = {}
            for name in self.out_names: 
                out[name] = getattr(self, f"final_conv_{name}")(x)
            outs.append(out)
        return outs



class ConvBlock(nn.Module):
    """
    Specific same-size-output 3x3 convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, stride=1, is_3d=True):
        super().__init__()

        conv = nn.Conv3d if is_3d else nn.Conv2d 
        self.main = conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out



################################



def get_head(train_args, f_maps_list, out_channels, is_3d, out_feat_level, stage=0, exclude_keys=[]): 
    if 'sep' in train_args.backbone: # separate decoder and head for anomaly/pathology segmentation
        return get_sep_head(train_args, f_maps_list, out_channels, is_3d, out_feat_level)
    elif '+' in train_args.backbone: # two-stage network for inpainting
        if stage == 0:
            return TaskHead(train_args, f_maps_list, {'pathology': 1}, is_3d, out_feat_level)
        else:
            return TaskHead(train_args, f_maps_list, out_channels, is_3d, out_feat_level, exclude_keys = ['pathology'])
    return TaskHead(train_args, f_maps_list, out_channels, is_3d, out_feat_level, exclude_keys)


def get_sep_head(train_args, f_maps_list, out_channels, is_3d, out_feat_level):  
    head_normal = TaskHead(train_args, f_maps_list, out_channels, is_3d, out_feat_level, ['pathology'])
    head_pathol = TaskHead(train_args, f_maps_list, {'pathology': 1}, is_3d, out_feat_level)
    return {'normal': head_normal, 'pathology': head_pathol}