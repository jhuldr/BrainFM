
"""
Datasets interface.
"""
from .constants import dataset_setups
from .datasets import BaseGen, BrainIDGen



dataset_options = {  
    'default': BaseGen,
    'brain_id': BrainIDGen,
}




def build_datasets(gen_args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    datasets = {'all': dataset_options[gen_args.dataset_option](gen_args, device)}
    return datasets

