
"""Config utilities for yml file."""
import os
from argparse import Namespace
import collections
import functools
import os
import re

import yaml
# from imaginaire.utils.distributed import master_only_print as print


class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # Treat as AttrDict above.
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, verbose=False):
        super(Config, self).__init__()

        # Update with given configurations.
        if os.path.exists(filename):

            loader = yaml.SafeLoader
            loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            try:
                with open(filename, 'r') as f:
                    cfg_dict = yaml.load(f, Loader=loader)
            except EnvironmentError:
                print('Please check the file with name of "%s"', filename)
            recursive_update(self, cfg_dict)
        else:
            raise ValueError('Provided config path not existed: %s' % filename)

        if verbose:
            print(' imaginaire config '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def rsetattr(obj, attr, val):
    """Recursively find object and set value"""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively find object and return value"""

    def _getattr(obj, attr):
        r"""Get attribute."""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    if u is not None:
        for key, value in u.items():
            if isinstance(value, collections.abc.Mapping):
                d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], dict):
                    d.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    d.__dict__[key] = value
            else:
                d.__dict__[key] = value
    return d


def merge_and_update_from_dict(cfg, dct):
    """
    (Compatible for submitit's Dict as attribute trick)
    Merge dict as dict() to config as CfgNode().
    Args:
        cfg: dict
        dct: dict
    """
    if dct is not None:
        for key, value in dct.items():
            if isinstance(value, dict):
                if key in cfg.keys():
                    sub_cfgnode = cfg[key]
                else:
                    sub_cfgnode = dict()
                    cfg.__setattr__(key, sub_cfgnode) 
                sub_cfgnode = merge_and_update_from_dict(sub_cfgnode, value)
            else:
                cfg[key] = value
    return cfg


def load_config(cfg_files = [], cfg_dir = ''):
    cfg = Config(cfg_files[0]) 
    for cfg_file in cfg_files[1:]:
        add_cfg = Config(cfg_file)
        cfg = merge_and_update_from_dict(cfg, add_cfg)
    return cfg
    

def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace


def preprocess_cfg(cfg_files, cfg_dir = ''):
    config = load_config(cfg_files, cfg_dir)
    args = nested_dict_to_namespace(config)
    return args