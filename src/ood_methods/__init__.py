from .msp import msp_score
from .odin import odin_score
from .mds import MDS
from .react import REACT

def get_ood_methods(ood_method, model):
    if ood_method == "msp":
        return msp_score
    elif ood_method == "odin":
        return odin_score
    elif ood_method == "mds":
        detector = MDS(model)
        return detector.mds_score
    elif ood_method == "react":
        detector = REACT(model)
        return detector.react_score
    else:
        raise ValueError(f"Unknown OOD method: {ood_method}")



