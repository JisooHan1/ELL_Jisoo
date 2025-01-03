from .msp import MSP
from .odin import ODIN
from .mds import MDS
from .react import ReAct
from .logitnorm import LogitNorm
from .knn import KNN

def get_ood_methods(ood_method, model=None):
    if ood_method == "msp":
        return MSP(model)
    elif ood_method == "odin":
        return ODIN(model)
    elif ood_method == "mds":
        return MDS(model)
    elif ood_method == "react":
        return ReAct(model)
    elif ood_method == "logitnorm":
        return LogitNorm(model)
    elif ood_method == "knn":
        return KNN(model)
    else:
        raise ValueError(f"Unknown OOD method: {ood_method}")



