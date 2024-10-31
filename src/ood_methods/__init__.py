from .msp import msp_score
from .odin import odin_score

ood_methods = {
    "msp": msp_score,
    "odin": odin_score,
    "mds": mds_score,
}

def get_ood_methods(ood_method):
    return ood_methods[ood_method]