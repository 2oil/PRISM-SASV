"""
ASVspoof2021 LA eval as a SASV trial set.

Enrollment: official ASVspoof2019 LA eval ASV protocol (11-19 bona-fide utterances/speaker).
Trials: ASVspoof2021 LA ASV protocol, restricted to codec=none/condition=none (the only rows
whose audio exists locally as plain, untransformed flac). ASVspoof2021 LA eval reuses ASVspoof2019
LA eval's underlying speakers/recordings, so this is a legitimate personalized-enrollment
scenario; only the UBM/fusion-classifier training domain (ASVspoof2019 LA TRAIN) is disjoint from
both enrollment and trial speakers.
"""
import os
import random

import pandas as pd

from configs import (ASVSPOOF2021_ENROLL_PROTOCOLS, ASVSPOOF2021_ENROLL_AUDIO_ROOT,
                      ASVSPOOF2021_TRIAL_PROTOCOL, ASVSPOOF2021_TEST_AUDIO_ROOT)

NAME = "ASVspoof2021"
N_SPEAKERS = 48
N_H0_PER_SPEAKER = 30
N_H1_PER_SPEAKER = 100
N_H2_PER_SPEAKER = 100


def resolve_path(utt):
    return os.path.join(ASVSPOOF2021_TEST_AUDIO_ROOT, f"{utt}.flac")


def _load_enroll_protocol():
    enroll = {}
    for p in ASVSPOOF2021_ENROLL_PROTOCOLS:
        with open(p) as f:
            for line in f:
                spk, utts = line.strip().split(" ", 1)
                enroll[spk] = utts.split(",")
    return enroll


def _load_trial_protocol():
    rows = []
    with open(ASVSPOOF2021_TRIAL_PROTOCOL) as f:
        for line in f:
            parts = line.strip().split()
            spk_raw, utt_raw, codec, cond, key_or_attack, target_flag, trim, partition = parts
            if partition != "eval" or codec != "none" or cond != "none":
                continue
            spk = spk_raw[:-len("-none")] if spk_raw.endswith("-none") else spk_raw
            utt = utt_raw[:-len("-none")] if utt_raw.endswith("-none") else utt_raw
            if utt.startswith("LA2021-"):
                utt = utt[len("LA2021-"):]
            hyp = {"target": 0, "nontarget": 1}.get(target_flag, 2)
            rows.append({"speaker": spk, "utt": utt, "hyp": hyp})
    df = pd.DataFrame(rows)
    return df[df["utt"].map(lambda u: os.path.exists(resolve_path(u)))].reset_index(drop=True)


def load_enrollment_and_trials(seed=42):
    enroll_dict = _load_enroll_protocol()
    trial_df = _load_trial_protocol()

    candidates = sorted(set(enroll_dict.keys()) & set(trial_df["speaker"].unique()))
    rng = random.Random(seed)
    speakers = sorted(rng.sample(candidates, min(N_SPEAKERS, len(candidates))))

    enroll_paths = {spk: [os.path.join(ASVSPOOF2021_ENROLL_AUDIO_ROOT, f"{u}.flac")
                           for u in enroll_dict[spk]] for spk in speakers}

    trials = []
    for spk in speakers:
        sub = trial_df[trial_df["speaker"] == spk]
        for hyp, n in [(0, N_H0_PER_SPEAKER), (1, N_H1_PER_SPEAKER), (2, N_H2_PER_SPEAKER)]:
            pool = sub[sub["hyp"] == hyp].to_dict("records")
            rng.shuffle(pool)
            for rec in pool[:n]:
                trials.append({"claimed_speaker": rec["speaker"], "utt": rec["utt"], "hyp": rec["hyp"]})

    return enroll_paths, pd.DataFrame(trials), resolve_path
