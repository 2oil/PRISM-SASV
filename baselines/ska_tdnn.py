"""
SKA-TDNN baseline: a single-embedding, integrated SASV model (SASV2_Baseline, Stage 3 --
jointly trained with aamsoftmax + angleproto_sasv losses on ASVspoof2019 LA bona-fide AND spoof
data). Cosine similarity between the enrollment centroid and the test embedding IS the final
SASV score directly -- no separate CM/fusion stage. Representative of the joint/integrated-
embedding SASV paradigm, as opposed to ASV+CM score fusion.

Requires the SASV2_Baseline repo (https://github.com/sasv-challenge/SASV2_Baseline) and its
Stage-3 (ASVspoof2019-finetuned) checkpoint -- see README "Setup". The Stage-3 checkpoint is not
publicly released by the repo authors; obtain it directly if you need it.
"""
import sys

import torch
import torch.nn.functional as F

from configs import SASV2_BASELINE_DIR, SKA_TDNN_CHECKPOINT_PATH


def _import_model_class():
    """SASV2_Baseline's models/SKA_TDNN.py does `from utils import PreEmphasis`, which would
    collide with this project's own `utils` module if left on sys.modules -- swap it out for the
    external repo's `utils.py` just for this import, then restore ours."""
    our_utils = sys.modules.pop("utils", None)
    model_dir = f"{SASV2_BASELINE_DIR}/stage3/ASVspoof2019"
    sys.path.insert(0, model_dir)
    sys.path.insert(0, f"{model_dir}/models")
    try:
        from models.SKA_TDNN import MainModel
    finally:
        sys.modules.pop("utils", None)
        sys.path.remove(model_dir)
        sys.path.remove(f"{model_dir}/models")
        if our_utils is not None:
            sys.modules["utils"] = our_utils
    return MainModel


def load_model(device="cuda"):
    MainModel = _import_model_class()
    model = MainModel()
    ckpt = torch.load(SKA_TDNN_CHECKPOINT_PATH, map_location="cpu")
    backbone_sd = {k[len("__S__."):]: v for k, v in ckpt.items() if k.startswith("__S__.")}
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    if missing or unexpected:
        print(f"[!] SKA-TDNN load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed(model, device, audio_1d_tensor):
    """audio_1d_tensor: (length,) float32 waveform at 16kHz. Returns an L2-normalized embedding."""
    x = audio_1d_tensor.unsqueeze(0).to(device)
    emb = model(x, aug=False)
    return F.normalize(emb, p=2, dim=1)
