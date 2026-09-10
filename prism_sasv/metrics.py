"""Evaluation metrics: EER and a-DCF (architecture-agnostic detection cost function)."""
from sklearn.metrics import roc_curve
import numpy as np


def compute_eer(scores, labels):
    """Equal error rate. scores: higher = more likely positive class. labels: {0,1}, 1=positive."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return fpr[idx], thresholds[idx]


def compute_a_dcf(scores, hyp_labels, p_tar=0.9405, p_non=0.0095, p_spoof=0.05,
                   c_miss=1.0, c_fa_asv=10.0, c_fa_cm=10.0):
    """Architecture-agnostic DCF (Shim et al., 2024), minimized over a single threshold.

    Priors match the official ASVspoof5 Track 2 spec (Table 1, ASVspoof5 Evaluation Plan):
    p_tar=0.9405, p_non=0.0095, p_spoof=0.05.

    hyp_labels: array-like of {0: target-bonafide (H0), 1: nontarget-bonafide (H1), 2: spoof (H2)}
    scores: higher = more likely accept (H0). Accept iff score > tau.
    Returns (min_a_dcf, best_tau).
    """
    scores = np.asarray(scores, dtype=float)
    hyp_labels = np.asarray(hyp_labels)

    tar_scores = scores[hyp_labels == 0]
    non_scores = scores[hyp_labels == 1]
    spoof_scores = scores[hyp_labels == 2]
    if tar_scores.size == 0 or non_scores.size == 0 or spoof_scores.size == 0:
        raise ValueError("a-DCF requires at least one trial of each hypothesis (H0, H1, H2)")

    thresholds = np.unique(scores)
    thresholds = np.concatenate(([thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]))

    # Vectorized via searchsorted (O(n log n)) instead of a per-threshold O(n) scan, which is
    # prohibitively slow once there are tens of thousands of unique scores.
    tar_sorted = np.sort(tar_scores)
    non_sorted = np.sort(non_scores)
    spoof_sorted = np.sort(spoof_scores)

    p_miss_tar = np.searchsorted(tar_sorted, thresholds, side="right") / tar_sorted.size
    p_fa_non = 1.0 - np.searchsorted(non_sorted, thresholds, side="right") / non_sorted.size
    p_fa_spoof = 1.0 - np.searchsorted(spoof_sorted, thresholds, side="right") / spoof_sorted.size

    dcf = (p_tar * c_miss * p_miss_tar
           + p_non * c_fa_asv * p_fa_non
           + p_spoof * c_fa_cm * p_fa_spoof)

    best_idx = np.argmin(dcf)
    best_dcf, best_tau = dcf[best_idx], thresholds[best_idx]

    normalizer = min(p_tar * c_miss, p_non * c_fa_asv + p_spoof * c_fa_cm)
    return best_dcf / normalizer, best_tau
