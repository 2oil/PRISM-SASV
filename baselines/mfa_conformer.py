"""
MFA-Conformer baseline: the other single-embedding, integrated SASV model from SASV2_Baseline
Stage 3 (same joint aamsoftmax + angleproto_sasv training as SKA-TDNN, different backbone).

Requires the SASV2_Baseline repo, its Stage-3 checkpoint, and `typeguard==2.13.3` (the wenet
Conformer implementation it depends on uses the legacy `check_argument_types()` API, which
newer typeguard versions changed incompatibly) -- see README "Setup".
"""
import sys

import torch
import torch.nn.functional as F

from configs import SASV2_BASELINE_DIR, MFA_CONFORMER_CHECKPOINT_PATH


def _import_model_class():
    our_utils = sys.modules.pop("utils", None)
    model_dir = f"{SASV2_BASELINE_DIR}/stage3/ASVspoof2019"
    sys.path.insert(0, model_dir)
    sys.path.insert(0, f"{model_dir}/models")
    try:
        from models.MFA_Conformer import MainModel
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
    ckpt = torch.load(MFA_CONFORMER_CHECKPOINT_PATH, map_location="cpu")
    backbone_sd = {k[len("__S__."):]: v for k, v in ckpt.items() if k.startswith("__S__.")}
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    if missing or unexpected:
        print(f"[!] MFA-Conformer load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed(model, device, audio_1d_tensor):
    x = audio_1d_tensor.unsqueeze(0).to(device)
    emb = model(x, aug=False)
    return F.normalize(emb, p=2, dim=1)
