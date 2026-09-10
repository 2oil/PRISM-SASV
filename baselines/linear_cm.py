"""
MMS-linear + ECAPA score-fusion baseline: an architecture-matched control for PRISM-SASV's
Branches 2/3. Instead of phoneme-conditioned GMMs, this mean-pools the SAME SSL frontend's frame
features over the WHOLE utterance and feeds them to a single linear (logistic regression)
classifier trained fresh on ASVspoof2019 LA train bona-fide/spoof data -- i.e. it isolates
"phoneme-conditioned GMM vs. whole-utterance-pooled linear classifier" as the only variable
against Branches 2/3, holding the feature extractor and training-data scope constant.

Final score = sigmoid(cm_logit) + s_utt_id (Branch 1's ECAPA cosine score), matching the
classic SASV Challenge Baseline1 fusion formula.
"""
import numpy as np
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def mean_pooled_mms(feature_extractor, audio_path):
    """feature_extractor: a prism_sasv.features.PhonemeFeatureExtractor (only its SSL frontend +
    aligner's preprocessing are used here; alignment itself is not needed for this baseline)."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor.asr_processor(audio, return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.to(feature_extractor.device)
    feats = feature_extractor.feat_model(input_values).squeeze(0)
    return feats.mean(dim=0).detach().cpu().numpy()


def train(feature_extractor, bona_paths, spoof_paths):
    """Fits mean-pool + logistic regression on labeled (bona-fide, spoof) audio paths."""
    X, y = [], []
    for p in bona_paths:
        try:
            X.append(mean_pooled_mms(feature_extractor, p)); y.append(1)
        except Exception:
            continue
    for p in spoof_paths:
        try:
            X.append(mean_pooled_mms(feature_extractor, p)); y.append(0)
        except Exception:
            continue
    X = np.array(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, y)
    return clf, scaler


def score(clf, scaler, feature_extractor, audio_path):
    """Returns the raw CM probability (index 1 = bona-fide) for one utterance."""
    pooled = mean_pooled_mms(feature_extractor, audio_path)
    return clf.predict_proba(scaler.transform(pooled.reshape(1, -1)))[0, 1]


def fuse(cm_score, s_utt_id):
    """Final SASV score: sigmoid(cm_logit-ish score) + Branch-1 cosine score."""
    return 1.0 / (1.0 + np.exp(-cm_score)) + s_utt_id


def save(clf, scaler, out_dir, name="linear_cm_baseline"):
    joblib.dump(clf, f"{out_dir}/{name}_clf.joblib")
    joblib.dump(scaler, f"{out_dir}/{name}_scaler.joblib")


def load(out_dir, name="linear_cm_baseline"):
    clf = joblib.load(f"{out_dir}/{name}_clf.joblib")
    scaler = joblib.load(f"{out_dir}/{name}_scaler.joblib")
    return clf, scaler
