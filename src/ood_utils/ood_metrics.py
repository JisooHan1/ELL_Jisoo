import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

def evaluations(id_scores, ood_scores):

    id_scores = id_scores.cpu().numpy()
    ood_scores = ood_scores.cpu().numpy()

    # generate list of label: ID = 1, OOD = 0
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # FPR95
    fprs, tprs, _ = roc_curve(labels, scores)
    tpr_95_idx = np.argmin(np.abs(tprs - 0.95))
    fpr95 = fprs[tpr_95_idx]

    # AUROC, AUPRC
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    return {
        'FPR95': round(fpr95, 4),
        'AUROC': round(auroc, 4),
        'AUPR': round(aupr, 4)
    }

