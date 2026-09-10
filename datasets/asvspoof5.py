"""
ASVspoof5 (Track 2, eval partition) as a SASV trial set.

The official enrollment set is only 3 utterances/speaker -- too sparse for per-phoneme GMMs, so
each speaker's enrollment pool is extended with a held-out-disjoint subset of their own "target"
(genuine) trial utterances; the remaining held-out genuine utterances become H0 eval trials.
Nontarget/spoof trials are used natively (the official protocol already labels them).
"""
import os
import random

import pandas as pd

from configs import (ASVSPOOF5_AUDIO_ROOT, ASVSPOOF5_ENROLL_PROTOCOL, ASVSPOOF5_TRIAL_PROTOCOL)

NAME = "ASVspoof5"
N_SPEAKERS = 367                 # all available
N_EXT_ENROLL_PER_SPEAKER = 35    # + the official 3 => 38 profiling utterances/speaker
N_H0_PER_SPEAKER = 15
N_H1_PER_SPEAKER = 100
N_H2_PER_SPEAKER = 100


def resolve_path(utt_id):
    return os.path.join(ASVSPOOF5_AUDIO_ROOT, f"{utt_id}.flac")


def _load_enroll_protocol(path):
    d = {}
    with open(path) as f:
        for line in f:
            spk, utts = line.strip().split()
            d[spk] = utts.split(",")
    return d


def _load_trial_protocol(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            spk, utt, label = parts[0], parts[1], parts[-1]
            rows.append((spk, utt, label))
    return pd.DataFrame(rows, columns=["speaker", "utt", "label"])


def load_enrollment_and_trials(seed=42):
    rng = random.Random(seed)
    official_enroll = _load_enroll_protocol(ASVSPOOF5_ENROLL_PROTOCOL)
    trial_df = _load_trial_protocol(ASVSPOOF5_TRIAL_PROTOCOL)

    speakers = sorted(official_enroll.keys())
    speakers = rng.sample(speakers, min(N_SPEAKERS, len(speakers)))
    trial_df = trial_df[trial_df["speaker"].isin(speakers)]

    enroll_paths, trials = {}, []
    for spk in speakers:
        spk_trials = trial_df[trial_df["speaker"] == spk]

        targets = spk_trials[spk_trials["label"] == "target"]["utt"].tolist()
        rng.shuffle(targets)
        ext_enroll = targets[:N_EXT_ENROLL_PER_SPEAKER]
        h0 = targets[N_EXT_ENROLL_PER_SPEAKER:N_EXT_ENROLL_PER_SPEAKER + N_H0_PER_SPEAKER]

        nontargets = spk_trials[spk_trials["label"] == "nontarget"]["utt"].tolist()
        h1 = rng.sample(nontargets, min(N_H1_PER_SPEAKER, len(nontargets)))

        spoofs = spk_trials[spk_trials["label"] == "spoof"]["utt"].tolist()
        h2 = rng.sample(spoofs, min(N_H2_PER_SPEAKER, len(spoofs)))

        enroll_paths[spk] = [resolve_path(u) for u in official_enroll[spk] + ext_enroll]
        for u in h0:
            trials.append({"claimed_speaker": spk, "utt": u, "hyp": 0})
        for u in h1:
            trials.append({"claimed_speaker": spk, "utt": u, "hyp": 1})
        for u in h2:
            trials.append({"claimed_speaker": spk, "utt": u, "hyp": 2})

    return enroll_paths, pd.DataFrame(trials), resolve_path
