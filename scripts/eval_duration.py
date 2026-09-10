"""
Short-utterance robustness evaluation: same system x dataset matrix as eval_full.py, but the
TEST side of each trial is truncated to 1/2/3/4 seconds (VAD-located speech onset, then a
contiguous window -- see prism_sasv/vad.py) before scoring. Enrollment always uses full-duration
audio.

Requires scripts/train_fusion.py to have been run first.

    python scripts/eval_duration.py --system v8 --dataset ASVspoof5 --durations 1,2,3,4
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
from prism_sasv.vad import truncate_to_duration
from baselines import ecapa as ecapa_lib
from datasets import REGISTRY
from scripts._common import build_system, DATASET_NAMES
from scripts.eval_full import SYSTEMS

TRUNC_CACHE_DIR = os.path.join(configs.SCRATCH_DIR, "duration_truncated")


def truncated_path(orig_path, dataset_name, duration_sec):
    key = f"{dataset_name}_{duration_sec}s_{abs(hash(orig_path))}.wav"
    return os.path.join(TRUNC_CACHE_DIR, key)


def get_or_truncate(orig_path, dataset_name, duration_sec):
    out_path = truncated_path(orig_path, dataset_name, duration_sec)
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = truncate_to_duration(orig_path, duration_sec, out_path)
    return out_path if ok else None


def evaluate(system_name, dataset_name, duration_sec, extractor, ecapa):
    ds = REGISTRY[dataset_name]
    enroll_paths, trials_df, resolve_path = ds.load_enrollment_and_trials()

    needs_phoneme_features = system_name in ("v8", "linear_cm")
    system = build_system(system_name,
                           extractor=extractor if needs_phoneme_features else None,
                           ecapa=ecapa)

    enrolled = {}
    for spk, paths in tqdm(enroll_paths.items(), desc=f"Enrolling ({system_name}/{dataset_name})"):
        try:
            enrolled[spk] = system.enroll(spk, paths)
        except Exception as e:
            print(f"[!] Enrollment failed for {spk}: {e}")

    scores, labels = [], []
    for row in tqdm(trials_df.to_dict("records"),
                     desc=f"Scoring ({system_name}/{dataset_name}/{duration_sec}s)"):
        spk = row["claimed_speaker"]
        if spk not in enrolled:
            continue
        full_path = resolve_path(row["utt"])
        clip_path = get_or_truncate(full_path, dataset_name, duration_sec)
        if clip_path is None:
            continue
        s = system.score(enrolled[spk], clip_path)
        if s is None:
            continue
        scores.append(s)
        labels.append(row["hyp"])

    eer, _ = compute_eer([s for s, h in zip(scores, labels) if h in (0, 1)],
                          [1 - h for s, h in zip(scores, labels) if h in (0, 1)])
    a_dcf, _ = compute_a_dcf(scores, labels)
    return {"system": system_name, "dataset": dataset_name, "duration_sec": duration_sec,
             "n_trials": len(scores), "eer": eer, "a_dcf": a_dcf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS + ["all"], default="all")
    ap.add_argument("--dataset", choices=DATASET_NAMES + ["all"], default="all")
    ap.add_argument("--durations", default=",".join(str(d) for d in configs.DURATIONS_SEC))
    args = ap.parse_args()

    systems = SYSTEMS if args.system == "all" else [args.system]
    datasets_ = DATASET_NAMES if args.dataset == "all" else [args.dataset]
    durations = [int(d) for d in args.durations.split(",")]

    needs_extractor = any(s in ("v8", "linear_cm") for s in systems)
    extractor = PhonemeFeatureExtractor() if needs_extractor else None
    ecapa = ecapa_lib.load_ecapa() if needs_extractor else None

    results = []
    for system_name in systems:
        for dataset_name in datasets_:
            for duration_sec in durations:
                results.append(evaluate(system_name, dataset_name, duration_sec, extractor, ecapa))

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    os.makedirs(configs.RESULTS_DIR, exist_ok=True)
    out_path = f"{configs.RESULTS_DIR}/eval_duration.csv"
    if os.path.exists(out_path):
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Saved to {out_path}")


if __name__ == "__main__":
    main()
