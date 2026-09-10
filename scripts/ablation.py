"""
PRISM-SASV (v8) ablation study: branch removal (w/o Branch 1/2/3), removing the top-K
restriction (all matched phonemes used instead), and uniform phoneme weighting (utility-based
weighting removed entirely) -- each evaluated as full-utterance SASV-EER / a-DCF across all 4
datasets.

Each ablation condition uses its OWN fusion classifier, refit on fitting trials scored under that
same ablation (e.g. "no_branch1" fits fusion with s_utt_id forced to 0; this mirrors what
scripts/train_fusion.py did for the full model, just with the ablation switch applied throughout).

Requires scripts/train_fusion.py to have been run first (for the population UBMs).

    python scripts/ablation.py --ablation no_branch1 --dataset ASVspoof5
    python scripts/ablation.py --ablation all --dataset all   # full ablation matrix
"""
import os
import sys
import random
import argparse
from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs
from prism_sasv.features import PhonemeFeatureExtractor
from prism_sasv.metrics import compute_eer, compute_a_dcf
from prism_sasv import fusion
from baselines import ecapa as ecapa_lib
from datasets import REGISTRY
from scripts._common import build_system, load_population_models, DATASET_NAMES
from scripts.train_fusion import (_load_protocol, _path_of, N_ENROLL_PER_SPEAKER, N_H0_PER_SPEAKER,
                                   N_H1_PER_SPEAKER, N_H2_PER_SPEAKER, FIT_SEED)

ABLATIONS = ["no_branch1", "no_branch2", "no_branch3", "no_topk", "uniform_weight"]


def fit_ablation_fusion(ablation, extractor, ecapa, bona_ubm, spoof_ubm):
    """Builds fitting trials + fits a fusion classifier for one ablation condition, from
    ASVspoof2019 LA train (same protocol/seed as scripts/train_fusion.py)."""
    cache_path = f"{configs.PRETRAINED_MODELS_DIR}/fusion_{ablation}"
    if os.path.exists(f"{cache_path}_clf.joblib"):
        return fusion.load(configs.PRETRAINED_MODELS_DIR, name=f"fusion_{ablation}")

    by_bona, by_spoof = _load_protocol()
    speakers = sorted(by_bona.keys())
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

    system = build_system("v8", extractor=extractor, ecapa=ecapa, ablation=ablation)
    system.bona_ubm, system.spoof_ubm = bona_ubm, spoof_ubm

    enrolled = {}
    for spk in tqdm(speakers, desc=f"Enrolling ({ablation})"):
        enrolled[spk] = system.enroll(spk, [_path_of(u) for u in enroll_utts[spk]])

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

    rows = []
    for trial in tqdm(trials, desc=f"Scoring fitting trials ({ablation})"):
        spk = trial["claimed_speaker"]
        path = _path_of(trial["utt"])
        try:
            test_emb = ecapa_lib.embed(system.ecapa, path, system._emb_cache)
            s_utt_id = 0.0 if ablation == "no_branch1" else ecapa_lib.cosine(test_emb, enrolled[spk]["centroid"])
        except Exception:
            continue
        feats = extractor.get_phoneme_features_from_file(path)
        if not feats:
            continue
        from prism_sasv.model import score_joint_mms_zscore, score_bf_generic
        s_pho_tgt = 0.0
        if ablation != "no_branch2":
            s_pho_tgt, _ = score_joint_mms_zscore(enrolled[spk]["prof"], enrolled[spk]["self_ll_std"],
                                                   enrolled[spk]["id_u"], feats, enrolled[spk]["id_topk"])
        s_pho_spf = 0.0
        if ablation != "no_branch3":
            s_pho_spf = score_bf_generic(bona_ubm, spoof_ubm, enrolled[spk]["cm_u"], feats,
                                          enrolled[spk]["cm_topk"])
        rows.append({**trial, "s_utt_id": s_utt_id, "s_pho_tgt": s_pho_tgt, "s_pho_spf": s_pho_spf})

    df = pd.DataFrame(rows)
    os.makedirs(configs.RESULTS_DIR, exist_ok=True)
    df.to_csv(f"{configs.RESULTS_DIR}/fusion_fitting_trials_{ablation}.csv", index=False)

    clf, scaler = fusion.fit(df)
    fusion.save(clf, scaler, fusion.DEFAULT_COLS, configs.PRETRAINED_MODELS_DIR, name=f"fusion_{ablation}")
    return clf, scaler, fusion.DEFAULT_COLS


def evaluate(ablation, dataset_name, extractor, ecapa, bona_ubm, spoof_ubm, clf, scaler, cols):
    ds = REGISTRY[dataset_name]
    enroll_paths, trials_df, resolve_path = ds.load_enrollment_and_trials()

    system = build_system("v8", extractor=extractor, ecapa=ecapa, ablation=ablation)
    system.bona_ubm, system.spoof_ubm = bona_ubm, spoof_ubm
    system.clf, system.scaler, system.cols = clf, scaler, cols

    enrolled = {}
    for spk, paths in tqdm(enroll_paths.items(), desc=f"Enrolling ({ablation}/{dataset_name})"):
        try:
            enrolled[spk] = system.enroll(spk, paths)
        except Exception as e:
            print(f"[!] Enrollment failed for {spk}: {e}")

    scores, labels = [], []
    for row in tqdm(trials_df.to_dict("records"), desc=f"Scoring ({ablation}/{dataset_name})"):
        spk = row["claimed_speaker"]
        if spk not in enrolled:
            continue
        s = system.score(enrolled[spk], resolve_path(row["utt"]))
        if s is None:
            continue
        scores.append(s)
        labels.append(row["hyp"])

    eer, _ = compute_eer([s for s, h in zip(scores, labels) if h in (0, 1)],
                          [1 - h for s, h in zip(scores, labels) if h in (0, 1)])
    a_dcf, _ = compute_a_dcf(scores, labels)
    return {"ablation": ablation, "dataset": dataset_name, "n_trials": len(scores),
            "eer": eer, "a_dcf": a_dcf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", choices=ABLATIONS + ["all"], default="all")
    ap.add_argument("--dataset", choices=DATASET_NAMES + ["all"], default="all")
    args = ap.parse_args()

    ablations = ABLATIONS if args.ablation == "all" else [args.ablation]
    datasets_ = DATASET_NAMES if args.dataset == "all" else [args.dataset]

    extractor = PhonemeFeatureExtractor()
    ecapa = ecapa_lib.load_ecapa()
    bona_ubm, spoof_ubm = load_population_models()

    results = []
    for ablation in ablations:
        clf, scaler, cols = fit_ablation_fusion(ablation, extractor, ecapa, bona_ubm, spoof_ubm)
        for dataset_name in datasets_:
            results.append(evaluate(ablation, dataset_name, extractor, ecapa, bona_ubm, spoof_ubm,
                                     clf, scaler, cols))

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    os.makedirs(configs.RESULTS_DIR, exist_ok=True)
    out_path = f"{configs.RESULTS_DIR}/ablation.csv"
    if os.path.exists(out_path):
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Saved to {out_path}")


if __name__ == "__main__":
    main()
