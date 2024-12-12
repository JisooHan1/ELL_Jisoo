from .msp import msp_score
from .odin import odin_score
from .mds import MDS
from .react import ReAct
from .logitnorm import LogitNorm
from .knn import KNN

def get_ood_methods(ood_method, model):
    if ood_method == "msp":
        return msp_score
    elif ood_method == "odin":
        return odin_score
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



