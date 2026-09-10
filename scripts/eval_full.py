"""
Full-utterance SASV evaluation: {v8, linear_cm, ska_tdnn, mfa_conformer} x
{ASVspoof2021, ASVspoof5, FamousFigures, SpoofCeleb}.

Requires scripts/train_fusion.py to have been run first (produces the fusion classifier, UBMs,
and linear_cm baseline classifier under configs.PRETRAINED_MODELS_DIR).

    python scripts/eval_full.py --system v8 --dataset ASVspoof5
    python scripts/eval_full.py --system all --dataset all   # full matrix
"""
import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs
from prism_sasv.features import PhonemeFeatureExtractor
from prism_sasv.metrics import compute_eer, compute_a_dcf
from baselines import ecapa as ecapa_lib
from datasets import REGISTRY
from scripts._common import build_system, DATASET_NAMES

SYSTEMS = ["v8", "linear_cm", "ska_tdnn", "mfa_conformer"]


def evaluate(system_name, dataset_name, extractor, ecapa):
    ds = REGISTRY[dataset_name]
    enroll_paths, trials_df, resolve_path = ds.load_enrollment_and_trials()

    needs_phoneme_features = system_name in ("v8", "linear_cm")
    system = build_system(system_name,
                           extractor=extractor if needs_phoneme_features else None,
                           ecapa=ecapa if system_name != "ska_tdnn" and system_name != "mfa_conformer" else None)

    enrolled = {}
    for spk, paths in tqdm(enroll_paths.items(), desc=f"Enrolling ({system_name}/{dataset_name})"):
        try:
            enrolled[spk] = system.enroll(spk, paths)
        except Exception as e:
            print(f"[!] Enrollment failed for {spk}: {e}")

    scores, labels = [], []
    for row in tqdm(trials_df.to_dict("records"), desc=f"Scoring ({system_name}/{dataset_name})"):
        spk = row["claimed_speaker"]
        if spk not in enrolled:
            continue
        path = resolve_path(row["utt"])
        s = system.score(enrolled[spk], path)
        if s is None:
            continue
        scores.append(s)
        labels.append(row["hyp"])

    eer, _ = compute_eer([s for s, h in zip(scores, labels) if h in (0, 1)],
                          [1 - h for s, h in zip(scores, labels) if h in (0, 1)])
    a_dcf, _ = compute_a_dcf(scores, labels)
    return {"system": system_name, "dataset": dataset_name,
             "n_trials": len(scores), "eer": eer, "a_dcf": a_dcf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS + ["all"], default="all")
    ap.add_argument("--dataset", choices=DATASET_NAMES + ["all"], default="all")
    args = ap.parse_args()

    systems = SYSTEMS if args.system == "all" else [args.system]
    datasets_ = DATASET_NAMES if args.dataset == "all" else [args.dataset]

    needs_extractor = any(s in ("v8", "linear_cm") for s in systems)
    needs_ecapa = any(s in ("v8", "linear_cm") for s in systems)
    extractor = PhonemeFeatureExtractor() if needs_extractor else None
    ecapa = ecapa_lib.load_ecapa() if needs_ecapa else None

    results = []
    for system_name in systems:
        for dataset_name in datasets_:
            results.append(evaluate(system_name, dataset_name, extractor, ecapa))

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    os.makedirs(configs.RESULTS_DIR, exist_ok=True)
    out_path = f"{configs.RESULTS_DIR}/eval_full.csv"
    if os.path.exists(out_path):
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Saved to {out_path}")


if __name__ == "__main__":
    main()
