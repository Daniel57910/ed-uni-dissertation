import torch

DISTRIBUTIONS = {
    
    'stack_overflow_v1': {
        'distribution_type': 'normal',
        'params': {
            'dist_object': torch.distributions.Normal,
            'variance': torch.tensor(10)
        }

    }
}