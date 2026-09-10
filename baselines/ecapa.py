"""
Branch 1 (utterance-level speaker identity) encoder: ECAPA-TDNN, pretrained on VoxCeleb.

Requires an external ECAPA-TDNN implementation + checkpoint (this repo does not vendor either --
see README "Setup"). Any standard ECAPA-TDNN checkpoint producing a fixed-size speaker embedding
from a raw 16kHz waveform works; swap `load_ecapa` below for your own loader if you use a
different implementation (e.g. SpeechBrain's `spkrec-ecapa-voxceleb`).
"""
import numpy as np
import soundfile as sf
import librosa
import torch

from configs import ECAPA_MODEL_PATH, ECAPA_CHECKPOINT_PATH, DEVICE


def load_ecapa():
    import sys
    sys.path.append(ECAPA_MODEL_PATH)
    from ecapa_tdnn import ECAPA_TDNN  # external implementation, see README

    model = ECAPA_TDNN(C=1024).to(DEVICE)
    ckpt = torch.load(ECAPA_CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k.replace("speaker_encoder.", ""): v for k, v in state.items()
             if k.startswith("speaker_encoder.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def read_mono_16k(wav_path):
    """soundfile first (fast path); falls back to librosa on decode failure (some flac files
    fail under libsndfile but decode fine via audioread)."""
    try:
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"expected 16kHz, got {sr} for {wav_path}")
        return wav
    except Exception:
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        return wav.astype(np.float32)


@torch.no_grad()
def embed(model, wav_path, cache=None):
    if cache is not None and wav_path in cache:
        return cache[wav_path]
    wav = read_mono_16k(wav_path)
    t = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    emb = model(t, aug=False).squeeze().cpu().numpy()
    if cache is not None:
        cache[wav_path] = emb
    return emb


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
