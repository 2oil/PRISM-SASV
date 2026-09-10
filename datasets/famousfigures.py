"""
FamousFigures (issf/famousfigures on HuggingFace -- gated dataset, see README "Setup") as a SASV
trial set. No official enroll/trial split is provided, so one is built here directly from the
per-speaker parquet shards: for each speaker, `ENROLL_FRACTION` of their bona-fide pool is
enrollment, the rest is H0; H1 is other speakers' bona-fide; H2 is that speaker's spoof pool.

Extracted audio is cached to `configs.SCRATCH_DIR/famousfigures_wav/` on first run.
"""
import os
import io
import glob
import random

import pandas as pd
import soundfile as sf

from configs import FAMOUSFIGURES_PARQUET_DIR, FAMOUSFIGURES_SPEAKERS, SCRATCH_DIR

NAME = "FamousFigures"
ENROLL_FRACTION = 0.10
MAX_H0_PER_SPEAKER = 550
MAX_H1_PER_SPEAKER = 3670
MAX_H2_PER_SPEAKER = 3670

WAV_DIR = os.path.join(SCRATCH_DIR, "famousfigures_wav")


def _write_wav(out_path, audio_dict):
    wav, sr = sf.read(io.BytesIO(audio_dict["bytes"]), dtype="float32")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, wav, sr, subtype="PCM_16")


def load_enrollment_and_trials(seed=42):
    rng = random.Random(seed)

    print("[*] Scanning FamousFigures parquet shards for per-speaker real/fake row pools...")
    real_by_speaker, fake_by_speaker = {}, {}
    for spk in FAMOUSFIGURES_SPEAKERS:
        shards = sorted(glob.glob(f"{FAMOUSFIGURES_PARQUET_DIR}/{spk}-*-of-*.parquet"))
        real_pool, fake_pool = [], []
        for shard in shards:
            df = pd.read_parquet(shard, columns=["Label"])
            real_pool += [(shard, i) for i in df.index[df["Label"] == "bonafide"]]
            fake_pool += [(shard, i) for i in df.index[df["Label"] == "spoof"]]
        real_by_speaker[spk] = real_pool
        fake_by_speaker[spk] = fake_pool

    enroll_rows, h0_rows, h2_rows = {}, {}, {}
    for spk in FAMOUSFIGURES_SPEAKERS:
        pool = real_by_speaker[spk][:]
        rng.shuffle(pool)
        n_enroll = round(len(pool) * ENROLL_FRACTION)
        enroll_rows[spk] = pool[:n_enroll]
        h0_rows[spk] = pool[n_enroll:n_enroll + MAX_H0_PER_SPEAKER]

        fpool = fake_by_speaker[spk][:]
        rng.shuffle(fpool)
        h2_rows[spk] = fpool[:MAX_H2_PER_SPEAKER]

    h1_rows = {}
    for spk in FAMOUSFIGURES_SPEAKERS:
        cand = []
        for other in FAMOUSFIGURES_SPEAKERS:
            if other == spk:
                continue
            cand += [(other, shard, idx) for shard, idx in h0_rows[other]]
        rng.shuffle(cand)
        h1_rows[spk] = cand[:MAX_H1_PER_SPEAKER]

    print("[*] Extracting audio to wav files + building protocol...")
    shard_cache = {}

    def get_row(shard, idx):
        if shard not in shard_cache:
            shard_cache[shard] = pd.read_parquet(shard, columns=["Audio"])
            if len(shard_cache) > 40:
                shard_cache.pop(next(iter(shard_cache)))
        return shard_cache[shard].loc[idx]

    enroll_paths, trials = {}, []
    for spk in FAMOUSFIGURES_SPEAKERS:
        paths = []
        for shard, idx in enroll_rows[spk]:
            utt_id = f"{spk}_enroll_{idx}_{os.path.basename(shard).split('-')[1]}"
            out_path = os.path.join(WAV_DIR, spk, f"{utt_id}.wav")
            if not os.path.exists(out_path):
                _write_wav(out_path, get_row(shard, idx)["Audio"])
            paths.append(out_path)
        enroll_paths[spk] = paths

        for shard, idx in h0_rows[spk]:
            utt_id = f"{spk}_h0_{idx}_{os.path.basename(shard).split('-')[1]}"
            out_path = os.path.join(WAV_DIR, spk, f"{utt_id}.wav")
            if not os.path.exists(out_path):
                _write_wav(out_path, get_row(shard, idx)["Audio"])
            trials.append({"claimed_speaker": spk, "utt": out_path, "hyp": 0})

        for other, shard, idx in h1_rows[spk]:
            utt_id = f"{other}_h1for{spk}_{idx}_{os.path.basename(shard).split('-')[1]}"
            out_path = os.path.join(WAV_DIR, other, f"{utt_id}.wav")
            if not os.path.exists(out_path):
                _write_wav(out_path, get_row(shard, idx)["Audio"])
            trials.append({"claimed_speaker": spk, "utt": out_path, "hyp": 1})

        for shard, idx in h2_rows[spk]:
            utt_id = f"{spk}_h2_{idx}_{os.path.basename(shard).split('-')[1]}"
            out_path = os.path.join(WAV_DIR, spk, f"{utt_id}.wav")
            if not os.path.exists(out_path):
                _write_wav(out_path, get_row(shard, idx)["Audio"])
            trials.append({"claimed_speaker": spk, "utt": out_path, "hyp": 2})

    return enroll_paths, pd.DataFrame(trials), (lambda utt: utt)  # utt is already a full wav path
