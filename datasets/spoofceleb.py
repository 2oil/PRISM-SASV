"""
SpoofCeleb evaluation partition as a SASV trial set.

SpoofCeleb's official SASV protocol is pairwise (1 enrollment utt, 1 test utt, label), designed
for single-shot ASV verification. Our method needs a multi-utterance enrollment profile per
speaker (enough samples to fit per-phoneme GMMs), so instead: (1) build a genuine enrollment pool
per speaker from the bona-fide (attack == "a00") files in the metadata, and (2) reuse the
official protocol's (enroll_path, test_path, label) pairings to determine which test files are
H0/H1/H2 for each claimed speaker (via the enroll_path's speaker identity), keeping the
officially-curated trial semantics while giving the method the enrollment volume it needs.
"""
import os
import random
from collections import defaultdict

import pandas as pd

from configs import SPOOFCELEB_METADATA_CSV, SPOOFCELEB_PROTOCOL_CSV, SPOOFCELEB_AUDIO_ROOT

NAME = "SpoofCeleb"
N_ENROLL_PER_SPEAKER = 35
N_H0_PER_SPEAKER = 30
N_H1_PER_SPEAKER = 100
N_H2_PER_SPEAKER = 100


def resolve_path(rel):
    return os.path.join(SPOOFCELEB_AUDIO_ROOT, rel)


def load_enrollment_and_trials(seed=42):
    rng = random.Random(seed)
    meta = pd.read_csv(SPOOFCELEB_METADATA_CSV)
    file_to_speaker = dict(zip(meta["file"], meta["speaker"]))

    bona_files_by_speaker = defaultdict(list)
    for _, row in meta[meta["attack"] == "a00"].iterrows():
        bona_files_by_speaker[row["speaker"]].append(row["file"])
    for spk in bona_files_by_speaker:
        bona_files_by_speaker[spk].sort()

    proto = pd.read_csv(SPOOFCELEB_PROTOCOL_CSV, header=None, names=["enroll", "test", "label"])
    proto["claimed_speaker"] = proto["enroll"].map(file_to_speaker)
    proto["hyp"] = proto["label"].map({"target": 0, "nontarget": 1, "spoof": 2})

    speakers = sorted(bona_files_by_speaker.keys())

    enroll_paths, enroll_files = {}, {}
    for spk in speakers:
        files = bona_files_by_speaker[spk][:]
        rng.shuffle(files)
        chosen = files[:N_ENROLL_PER_SPEAKER]
        enroll_files[spk] = set(chosen)
        enroll_paths[spk] = [resolve_path(f) for f in sorted(chosen)]

    trials = []
    for spk in speakers:
        sub = proto[proto["claimed_speaker"] == spk]
        for hyp, n in [(0, N_H0_PER_SPEAKER), (1, N_H1_PER_SPEAKER), (2, N_H2_PER_SPEAKER)]:
            pool = sub[sub["hyp"] == hyp]
            pool = pool[~pool["test"].isin(enroll_files[spk])]  # avoid literal file overlap
            pool = pool["test"].unique().tolist()
            rng.shuffle(pool)
            for f in pool[:n]:
                trials.append({"claimed_speaker": spk, "utt": f, "hyp": hyp})

    return enroll_paths, pd.DataFrame(trials), resolve_path
