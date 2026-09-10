"""
Core PRISM-SASV methodology: phoneme-conditioned GMMs, utility-based top-K phoneme selection,
and the Branch 2 (personalized identity) / Branch 3 (population authenticity) scoring functions.

See the paper's Methodology section for the full derivation. Summary:
  - `EnrollmentProfile` fits one speaker-specific GMM per phoneme from enrollment audio.
  - `build_fixed_cohort_gmm` fits population-level (cohort / bona-fide / spoof) GMMs per phoneme,
    once, from a disjoint training corpus, shared across all speakers.
  - `compute_phoneme_utility_no_spoof` / `compute_cm_utility` score each phoneme's usefulness for
    Branch 2 / Branch 3 respectively; `select_top_k` keeps the top-K by that score.
  - `score_joint_mms_zscore` (Branch 2) and `score_bf_generic` (Branch 3) aggregate per-phoneme
    evidence over the selected phonemes into the two branch scores, with a tiered fallback when a
    test utterance doesn't contain any of its speaker's top-K phonemes.
"""
from collections import defaultdict

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def train_gmm(vectors, n_components=5, reg_covar=1e-3):
    """Diagonal-covariance GMM, degrading to a single Gaussian for very small phoneme samples."""
    vectors = np.asarray(vectors)
    if len(vectors) < n_components * 2:
        n_components = 1
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag",
                           reg_covar=reg_covar, random_state=42)
    gmm.fit(vectors)
    return gmm


def minmax_norm(d, lo=0.1, hi=1.0):
    """Map dict values linearly into [lo, hi]."""
    if not d:
        return {}
    values = np.array(list(d.values()))
    vmin, vmax = values.min(), values.max()
    out = {}
    for k, v in d.items():
        out[k] = hi if vmax <= vmin else lo + (hi - lo) * (v - vmin) / (vmax - vmin)
    return out


def extract_phoneme_pool(paths, extractor, verbose=False):
    """Pool phoneme-level feature vectors across a flat list of audio paths (no speaker
    grouping) -- used to build the population cohort/bona-fide/spoof GMMs."""
    pool = defaultdict(list)
    it = tqdm(paths, desc="Extracting phoneme pool", leave=False) if verbose else paths
    for path in it:
        for item in extractor.get_phoneme_features_from_file(path):
            pool[item["char"].lower()].append(item["feature"])
    return pool


def build_fixed_cohort_gmm(phoneme_pool, n_components=3, min_samples=5):
    """A single, speaker-independent background GMM per phoneme, trained once from a large
    disjoint population and reused unchanged for every enrolled speaker (a classical ASV
    Universal Background Model, phoneme-conditioned). `phoneme_pool`: {phoneme: [vec, ...]},
    e.g. from `extract_phoneme_pool`."""
    cohort_gmm = {}
    for phn, vecs in phoneme_pool.items():
        if len(vecs) < min_samples:
            continue
        cohort_gmm[phn] = train_gmm(np.array(vecs), n_components)
    return cohort_gmm


class EnrollmentProfile:
    """Per-speaker phoneme profile built once from that speaker's enrollment (bona-fide) audio."""

    def __init__(self, speaker_id, extractor, n_components=5, min_samples=5):
        self.speaker_id = speaker_id
        self.extractor = extractor
        self.n_components = n_components
        self.min_samples = min_samples

        self.raw = {}          # phoneme -> np.ndarray [N_p, D]
        self.target_gmm = {}   # phoneme -> GaussianMixture (G_p^spk)
        self.self_ll = {}      # phoneme -> mean log-likelihood of own data under own GMM
        self.n_p = {}          # phoneme -> sample count

    def fit(self, enroll_paths, verbose=False):
        phn_data = extract_phoneme_pool(enroll_paths, self.extractor, verbose=verbose)
        for phn, vecs in phn_data.items():
            if len(vecs) < self.min_samples:
                continue
            vecs = np.array(vecs)
            gmm = train_gmm(vecs, self.n_components)
            self.raw[phn] = vecs
            self.target_gmm[phn] = gmm
            self.self_ll[phn] = gmm.score(vecs)
            self.n_p[phn] = len(vecs)

    @property
    def phonemes(self):
        return set(self.target_gmm.keys())


def compute_self_ll_std(profile, min_std=1e-3):
    """Per-phoneme std of the enrollment's own vectors' log-likelihood under their own GMM --
    the scale term used to z-normalize Branch 2's test-time evidence."""
    std = {}
    for phn, vecs in profile.raw.items():
        if phn not in profile.target_gmm:
            continue
        per_sample_ll = profile.target_gmm[phn].score_samples(vecs)
        std[phn] = max(float(per_sample_ll.std()), min_std)
    return std


def compute_phoneme_utility_no_spoof(profile, cohort_gmm):
    """Branch 2 (identity) utility u_p: how well phoneme p separates this speaker from the
    population cohort, discounted by sample-size reliability."""
    d_spk = {}
    for phn, vecs in profile.raw.items():
        if phn in cohort_gmm:
            d_spk[phn] = profile.self_ll[phn] - cohort_gmm[phn].score(vecs)
    if not d_spk:
        return {phn: 1.0 for phn in profile.raw}

    d_spk_n = minmax_norm(d_spk)
    n_ref = np.median([profile.n_p[p] for p in d_spk])
    u_p = {}
    for phn in d_spk:
        r_p = min(1.0, profile.n_p[phn] / n_ref) if n_ref > 0 else 1.0
        u_p[phn] = d_spk_n[phn] * r_p
    for phn in profile.raw:
        if phn not in u_p:
            u_p[phn] = 0.05
    return u_p


def compute_cm_utility(profile, bona_ubm, spoof_ubm):
    """Branch 3 (authenticity) utility u_p: how well phoneme p separates bona-fide from spoof in
    the population-level dual UBM, probed using this speaker's own genuine enrollment vectors."""
    d_cm = {}
    for phn, vecs in profile.raw.items():
        if phn in bona_ubm and phn in spoof_ubm:
            d_cm[phn] = bona_ubm[phn].score(vecs) - spoof_ubm[phn].score(vecs)
    if not d_cm:
        return {phn: 1.0 for phn in profile.raw}

    d_cm_n = minmax_norm(d_cm)
    n_ref = np.median([profile.n_p[p] for p in d_cm])
    u_p = {}
    for phn in d_cm:
        r_p = min(1.0, profile.n_p[phn] / n_ref) if n_ref > 0 else 1.0
        u_p[phn] = d_cm_n[phn] * r_p
    for phn in profile.raw:
        if phn not in u_p:
            u_p[phn] = 0.05
    return u_p


def uniform_weight(profile):
    """Ablation baseline: every phoneme weighted equally (u_p = 1), instead of utility-based."""
    return {phn: 1.0 for phn in profile.raw}


def select_top_k(weight_dict, top_k):
    """Ranks phonemes by weight and keeps the top-K (Algorithm 1 in the paper)."""
    ordered = sorted(weight_dict.items(), key=lambda kv: kv[1], reverse=True)
    return set(k for k, _ in ordered[:top_k])


def score_joint_mms_zscore(profile, self_ll_std, weight_dict, test_feats, top_k_set):
    """Branch 2 (S_pho-tgt): utility-weighted average of z-normalized log-likelihood under the
    target speaker's own phoneme GMMs. Tiered fallback: top-K phonemes present in the test
    utterance -> all phonemes matching the profile -> 0.0 if nothing matches."""
    top, w_top, allm, w_all = [], [], [], []
    for item in test_feats:
        phn = item["char"].lower()
        if phn not in profile.target_gmm or phn not in profile.self_ll or phn not in self_ll_std:
            continue
        ll = profile.target_gmm[phn].score_samples(item["feature"].reshape(1, -1))[0]
        z = (ll - profile.self_ll[phn]) / self_ll_std[phn]
        w = weight_dict.get(phn, 0.05)
        allm.append(z); w_all.append(w)
        if phn in top_k_set:
            top.append(z); w_top.append(w)
    if top:
        return np.average(top, weights=w_top), "top_k"
    if allm:
        return np.average(allm, weights=w_all), "all_matched"
    return 0.0, "fail"


def score_bf_generic(bona_ubm, spoof_ubm, weight_dict, test_feats, top_k_set):
    """Branch 3 (S_pho-spf): utility-weighted average bona-fide/spoof log-likelihood ratio
    against the population UBMs (NOT the speaker's own GMM -- population-level, shared across
    all speakers). Same tiered fallback as Branch 2."""
    top, w_top, allm, w_all = [], [], [], []
    for item in test_feats:
        phn = item["char"].lower()
        vec = item["feature"]
        if phn not in bona_ubm or phn not in spoof_ubm:
            continue
        s_bf = (bona_ubm[phn].score_samples(vec.reshape(1, -1))[0]
                - spoof_ubm[phn].score_samples(vec.reshape(1, -1))[0])
        w = weight_dict.get(phn, 0.05)
        allm.append(s_bf); w_all.append(w)
        if phn in top_k_set:
            top.append(s_bf); w_top.append(w)
    if top:
        return np.average(top, weights=w_top)
    if allm:
        return np.average(allm, weights=w_all)
    return 0.0
