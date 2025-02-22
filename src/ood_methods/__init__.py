from .msp import MSP
from .odin import ODIN
from .mds import MDS
from .react import ReAct
from .energy import Energy
from .logitnorm import LogitNormLoss
from .knn import KNN
from .outlier_exposure import OutlierExposureLoss
from .mixture_outlier_exposure import MixtureOutlierExposureLoss

def get_ood_methods(ood_method, model=None):
    if ood_method == "msp":
        return MSP(model)
    elif ood_method == "odin":
        return ODIN(model)
    elif ood_method == "mds":
        return MDS(model)
    elif ood_method == "react":
        return ReAct(model)
    elif ood_method == "energy":
        return Energy(model)
    elif ood_method == "knn":
        return KNN(model)
    elif ood_method == "logitnorm":
        return LogitNormLoss(model)
    elif ood_method == "outlier_exposure":
        return OutlierExposureLoss(model)
    elif ood_method == "mixture_outlier_exposure":
        return MixtureOutlierExposureLoss(model)
    else:
        raise ValueError(f"Unknown OOD method: {ood_method}")



