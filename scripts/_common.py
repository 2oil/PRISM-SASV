"""Shared helpers for eval_full.py / eval_duration.py / ablation.py: loading the pretrained
components produced by train_fusion.py, and per-trial scoring for each system under test."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs
from prism_sasv.model import (EnrollmentProfile, compute_self_ll_std, compute_phoneme_utility_no_spoof,
                               compute_cm_utility, uniform_weight, select_top_k, score_joint_mms_zscore,
                               score_bf_generic)
from prism_sasv import fusion
from baselines import ecapa as ecapa_lib
from baselines import ska_tdnn, mfa_conformer, linear_cm

import joblib

DATASET_NAMES = ["ASVspoof2021", "ASVspoof5", "FamousFigures", "SpoofCeleb"]


def load_population_models():
    bona_ubm = joblib.load(f"{configs.PRETRAINED_MODELS_DIR}/bona_ubm.joblib")
    spoof_ubm = joblib.load(f"{configs.PRETRAINED_MODELS_DIR}/spoof_ubm.joblib")
    return bona_ubm, spoof_ubm


def load_fusion():
    return fusion.load(configs.PRETRAINED_MODELS_DIR)


class V8System:
    """Wraps branch-score computation + fusion for PRISM-SASV (v8), with ablation switches.

    ablation: None (full model), "no_branch1", "no_branch2", "no_branch3", "no_topk",
              "uniform_weight".
    """

    def __init__(self, extractor, ecapa, bona_ubm, spoof_ubm, clf, scaler, cols, ablation=None):
        self.extractor = extractor
        self.ecapa = ecapa
        self.bona_ubm = bona_ubm
        self.spoof_ubm = spoof_ubm
        self.clf, self.scaler, self.cols = clf, scaler, cols
        self.ablation = ablation
        self._emb_cache = {}

    def enroll(self, speaker_id, enroll_paths):
        prof = EnrollmentProfile(speaker_id, self.extractor)
        prof.fit(enroll_paths)
        self_ll_std = compute_self_ll_std(prof)

        if self.ablation == "uniform_weight":
            id_u = uniform_weight(prof)
            cm_u = uniform_weight(prof)
        else:
            id_u = compute_phoneme_utility_no_spoof(prof, self.bona_ubm)
            cm_u = compute_cm_utility(prof, self.bona_ubm, self.spoof_ubm)

        top_k = len(prof.raw) if self.ablation == "no_topk" else configs.TOP_K
        id_topk = select_top_k(id_u, top_k)
        cm_topk = select_top_k(cm_u, top_k)

        centroid = np.mean([ecapa_lib.embed(self.ecapa, p, self._emb_cache) for p in enroll_paths], axis=0)
        return {"prof": prof, "self_ll_std": self_ll_std, "id_u": id_u, "id_topk": id_topk,
                "cm_u": cm_u, "cm_topk": cm_topk, "centroid": centroid}

    def score(self, enrolled, test_path):
        """Returns the final PRISM-SASV score for one trial (None if scoring failed)."""
        try:
            test_emb = ecapa_lib.embed(self.ecapa, test_path, self._emb_cache)
            s_utt_id = ecapa_lib.cosine(test_emb, enrolled["centroid"])
        except Exception:
            return None
        feats = self.extractor.get_phoneme_features_from_file(test_path)
        if not feats:
            return None

        s_pho_tgt = 0.0
        if self.ablation != "no_branch2":
            s_pho_tgt, _ = score_joint_mms_zscore(enrolled["prof"], enrolled["self_ll_std"],
                                                   enrolled["id_u"], feats, enrolled["id_topk"])
        s_pho_spf = 0.0
        if self.ablation != "no_branch3":
            s_pho_spf = score_bf_generic(self.bona_ubm, self.spoof_ubm, enrolled["cm_u"], feats,
                                          enrolled["cm_topk"])
        if self.ablation == "no_branch1":
            s_utt_id = 0.0

        row = {"s_utt_id": s_utt_id, "s_pho_tgt": s_pho_tgt, "s_pho_spf": s_pho_spf}
        X = self.scaler.transform([[row[c] for c in self.cols]])
        return self.clf.predict_proba(X)[0, list(self.clf.classes_).index(0)]


class EmbeddingBaselineSystem:
    """SKA-TDNN / MFA-Conformer: single embedding, cosine score against the enrollment centroid IS
    the SASV score (no separate CM stage)."""

    def __init__(self, kind, device=None):
        assert kind in ("ska_tdnn", "mfa_conformer")
        self.lib = ska_tdnn if kind == "ska_tdnn" else mfa_conformer
        self.device = device or configs.DEVICE
        self.model = self.lib.load_model(self.device)
        self._emb_cache = {}

    def _embed(self, path):
        if path in self._emb_cache:
            return self._emb_cache[path]
        wav = ecapa_lib.read_mono_16k(path)
        t = torch.from_numpy(wav).float()
        emb = self.lib.embed(self.model, self.device, t).squeeze(0).cpu().numpy()
        self._emb_cache[path] = emb
        return emb

    def enroll(self, speaker_id, enroll_paths):
        embs = [self._embed(p) for p in enroll_paths]
        return {"centroid": np.mean(embs, axis=0)}

    def score(self, enrolled, test_path):
        try:
            emb = self._embed(test_path)
        except Exception:
            return None
        return ecapa_lib.cosine(emb, enrolled["centroid"])


class LinearCMBaselineSystem:
    """MMS-linear + ECAPA score fusion (architecture-matched control for Branches 2/3)."""

    def __init__(self, extractor, ecapa):
        self.extractor = extractor
        self.ecapa = ecapa
        self.clf, self.scaler = linear_cm.load(configs.PRETRAINED_MODELS_DIR)
        self._emb_cache = {}

    def enroll(self, speaker_id, enroll_paths):
        centroid = np.mean([ecapa_lib.embed(self.ecapa, p, self._emb_cache) for p in enroll_paths], axis=0)
        return {"centroid": centroid}

    def score(self, enrolled, test_path):
        try:
            test_emb = ecapa_lib.embed(self.ecapa, test_path, self._emb_cache)
            s_utt_id = ecapa_lib.cosine(test_emb, enrolled["centroid"])
            cm_score = linear_cm.score(self.clf, self.scaler, self.extractor, test_path)
        except Exception:
            return None
        return linear_cm.fuse(cm_score, s_utt_id)


def build_system(name, extractor=None, ecapa=None, ablation=None):
    """name: 'v8', 'linear_cm', 'ska_tdnn', 'mfa_conformer'."""
    if name == "v8":
        bona_ubm, spoof_ubm = load_population_models()
        clf, scaler, cols = load_fusion()
        return V8System(extractor, ecapa, bona_ubm, spoof_ubm, clf, scaler, cols, ablation=ablation)
    if name == "linear_cm":
        return LinearCMBaselineSystem(extractor, ecapa)
    if name in ("ska_tdnn", "mfa_conformer"):
        return EmbeddingBaselineSystem(name)
    raise ValueError(f"Unknown system '{name}'")
