posthoc_methods = ['msp', 'odin', 'mds', 'react', 'knn', 'energy']
training_methods = ['logitnorm', 'oe', 'moe']

# get training config
def get_training_config(method):
    if method == 'logitnorm':
        from ood_methods.logitnorm import logitnorm_config
        return logitnorm_config
    elif method == 'oe':
        from ood_methods.outlier_exposure import oe_config
        return oe_config
    elif method == 'moe':
        from ood_methods.mixture_outlier_exposure import moe_config
        return moe_config

