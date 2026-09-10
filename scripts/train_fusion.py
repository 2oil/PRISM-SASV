"""
One-time setup: builds the population UBMs (bona-fide / spoof / cohort phoneme GMMs, shared
across every speaker and dataset) and fits the frozen 3-way fusion classifier, both from
ASVspoof2019 LA train -- disjoint from every evaluation dataset used elsewhere in this repo.
Also trains the MMS-linear+ECAPA baseline classifier on the same data.

Run this once before any of eval_full.py / eval_duration.py / ablation.py.

    python scripts/train_fusion.py
"""
import os
import sys
import random
import argparse
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs
from prism_sasv.features import PhonemeFeatureExtractor
from prism_sasv.model import (EnrollmentProfile, extract_phoneme_pool, build_fixed_cohort_gmm,
                               compute_phoneme_utility_no_spoof, compute_cm_utility,
                               compute_self_ll_std, select_top_k, score_joint_mms_zscore,
                               score_bf_generic)
from prism_sasv import fusion
from baselines import ecapa as ecapa_lib
from baselines import linear_cm

BONA_PER_SPEAKER_UBM = 100
SPOOF_PER_SPEAKER_UBM = 250
N_ENROLL_PER_SPEAKER = 30
N_H0_PER_SPEAKER = 30
N_H1_PER_SPEAKER = 100
N_H2_PER_SPEAKER = 100
UBM_SEED = 42     # deliberately different from FIT_SEED, to reduce (not fully eliminate) overlap
FIT_SEED = 123    # between a speaker's UBM contribution and their own enrollment/trial utterances


def _path_of(utt):
    return os.path.join(configs.ASVSPOOF2019_LA_TRAIN_AUDIO_ROOT, f"{utt}.flac")


def _load_protocol():
    by_bona, by_spoof = defaultdict(list), defaultdict(list)
    with open(configs.ASVSPOOF2019_LA_TRAIN_PROTOCOL) as f:
        for line in f:
            parts = line.strip().split()
            spk, utt, label = parts[0], parts[1], parts[-1]
            (by_bona if label == "bonafide" else by_spoof)[spk].append(utt)
    return by_bona, by_spoof


def build_ubms(extractor, by_bona, by_spoof):
    os.makedirs(configs.PRETRAINED_MODELS_DIR, exist_ok=True)
    bona_path = f"{configs.PRETRAINED_MODELS_DIR}/bona_ubm.joblib"
    spoof_path = f"{configs.PRETRAINED_MODELS_DIR}/spoof_ubm.joblib"
    if os.path.exists(bona_path) and os.path.exists(spoof_path):
        print("[*] Loading cached UBMs.")
        return joblib.load(bona_path), joblib.load(spoof_path)

    rng = random.Random(UBM_SEED)

    def sample_paths(by_speaker, n_per_speaker):
        paths = []
        for spk, utts in by_speaker.items():
            sampled = rng.sample(utts, min(n_per_speaker, len(utts)))
            paths += [_path_of(u) for u in sampled]
        return paths

    print("[*] Building bona-fide population UBM...")
    bona_pool = extract_phoneme_pool(sample_paths(by_bona, BONA_PER_SPEAKER_UBM), extractor, verbose=True)
    bona_ubm = build_fixed_cohort_gmm(bona_pool, n_components=configs.POPULATION_GMM_COMPONENTS)

    print("[*] Building spoof population UBM...")
    spoof_pool = extract_phoneme_pool(sample_paths(by_spoof, SPOOF_PER_SPEAKER_UBM), extractor, verbose=True)
    spoof_ubm = build_fixed_cohort_gmm(spoof_pool, n_components=configs.POPULATION_GMM_COMPONENTS)

    joblib.dump(bona_ubm, bona_path)
    joblib.dump(spoof_ubm, spoof_path)
    print(f"[*] bona_ubm: {len(bona_ubm)} phonemes, spoof_ubm: {len(spoof_ubm)} phonemes.")
    return bona_ubm, spoof_ubm


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    extractor = PhonemeFeatureExtractor()
    by_bona, by_spoof = _load_protocol()
    speakers = sorted(by_bona.keys())

    bona_ubm, spoof_ubm = build_ubms(extractor, by_bona, by_spoof)

    # --- MMS-linear + ECAPA baseline classifier (trained on the same bona/spoof pool) ---
    print("[*] Training MMS-linear baseline classifier...")
    rng_lin = random.Random(UBM_SEED)
    bona_paths = [_path_of(u) for spk in speakers for u in
                  rng_lin.sample(by_bona[spk], min(BONA_PER_SPEAKER_UBM, len(by_bona[spk])))]
    spoof_paths = [_path_of(u) for spk in speakers for u in
                   rng_lin.sample(by_spoof[spk], min(SPOOF_PER_SPEAKER_UBM, len(by_spoof[spk])))]
    lin_clf, lin_scaler = linear_cm.train(extractor, bona_paths, spoof_paths)
    linear_cm.save(lin_clf, lin_scaler, configs.PRETRAINED_MODELS_DIR)

    # --- Fusion-classifier fitting trials ---
    print("[*] Building enrollment profiles for fusion-classifier fitting...")
    rng = random.Random(FIT_SEED)
    enroll_utts, h0_pool, h2_pool = {}, {}, {}
    for spk in speakers:
        bona = by_bona[spk][:]
        rng.shuffle(bona)
        enroll_utts[spk] = bona[:N_ENROLL_PER_SPEAKER]
        h0_pool[spk] = bona[N_ENROLL_PER_SPEAKER:N_ENROLL_PER_SPEAKER + N_H0_PER_SPEAKER]
        spoof = by_spoof[spk][:]
        rng.shuffle(spoof)
        h2_pool[spk] = spoof[:N_H2_PER_SPEAKER]

    ecapa = ecapa_lib.load_ecapa()
    profiles, self_ll_std, id_u, id_topk, cm_u, cm_topk, utt_centroid = {}, {}, {}, {}, {}, {}, {}
    for spk in tqdm(speakers, desc="Enrolling"):
        paths = [_path_of(u) for u in enroll_utts[spk]]
        prof = EnrollmentProfile(spk, extractor)
        prof.fit(paths)
        profiles[spk] = prof
        self_ll_std[spk] = compute_self_ll_std(prof)

        u_id = compute_phoneme_utility_no_spoof(prof, bona_ubm)
        id_u[spk] = u_id
        id_topk[spk] = select_top_k(u_id, configs.TOP_K)
        u_cm = compute_cm_utility(prof, bona_ubm, spoof_ubm)
        cm_u[spk] = u_cm
        cm_topk[spk] = select_top_k(u_cm, configs.TOP_K)

        embs = [ecapa_lib.embed(ecapa, p) for p in paths]
        utt_centroid[spk] = np.mean(embs, axis=0)

    print("[*] Constructing H0/H1/H2 fitting trials...")
    trials = []
    for spk in speakers:
        for u in h0_pool[spk]:
            trials.append({"claimed_speaker": spk, "utt": u, "hyp": 0})
        others = [s for s in speakers if s != spk]
        h1_candidates = [(o, u) for o in others for u in by_bona[o]]
        rng.shuffle(h1_candidates)
        for _, u in h1_candidates[:N_H1_PER_SPEAKER]:
            trials.append({"claimed_speaker": spk, "utt": u, "hyp": 1})
        for u in h2_pool[spk]:
            trials.append({"claimed_speaker": spk, "utt": u, "hyp": 2})

    print(f"[*] Scoring {len(trials)} fitting trials...")
    emb_cache = {}
    rows = []
    for trial in tqdm(trials, desc="Scoring"):
        spk = trial["claimed_speaker"]
        path = _path_of(trial["utt"])
        try:
            test_emb = ecapa_lib.embed(ecapa, path, emb_cache)
            s_utt_id = ecapa_lib.cosine(test_emb, utt_centroid[spk])
        except Exception:
            continue
        feats = extractor.get_phoneme_features_from_file(path)
        if not feats:
            continue
        s_pho_tgt, _ = score_joint_mms_zscore(profiles[spk], self_ll_std[spk], id_u[spk], feats, id_topk[spk])
        s_pho_spf = score_bf_generic(bona_ubm, spoof_ubm, cm_u[spk], feats, cm_topk[spk])
        rows.append({**trial, "s_utt_id": s_utt_id, "s_pho_tgt": s_pho_tgt, "s_pho_spf": s_pho_spf})

    df = pd.DataFrame(rows)
    os.makedirs(configs.RESULTS_DIR, exist_ok=True)
    df.to_csv(f"{configs.RESULTS_DIR}/fusion_fitting_trials.csv", index=False)

    print("[*] Fitting frozen 3-way fusion classifier...")
    clf, scaler = fusion.fit(df)
    fusion.save(clf, scaler, fusion.DEFAULT_COLS, configs.PRETRAINED_MODELS_DIR)
    print(f"[*] Done. Saved to {configs.PRETRAINED_MODELS_DIR}/ (fusion_*, bona_ubm.joblib, "
          f"spoof_ubm.joblib, linear_cm_baseline_*).")


if __name__ == "__main__":
    main()
