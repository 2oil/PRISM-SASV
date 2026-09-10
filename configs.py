"""
All machine-specific paths in one place. Fill these in before running anything -- see README
"Setup" for what each one needs to point to and where to get it.
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Branch 1 speaker encoder (ECAPA-TDNN)
# ---------------------------------------------------------------------------
ECAPA_MODEL_PATH = "/path/to/ecapa_tdnn_impl"      # dir containing ecapa_tdnn.py
ECAPA_CHECKPOINT_PATH = "/path/to/ecapa_tdnn.pth"

# ---------------------------------------------------------------------------
# Baselines: SKA-TDNN / MFA-Conformer (SASV2_Baseline, Stage 3)
# ---------------------------------------------------------------------------
SASV2_BASELINE_DIR = "/path/to/SASV2_Baseline"
SKA_TDNN_CHECKPOINT_PATH = "/path/to/SASV2_Baseline/ska_tdnn_stage3.model"
MFA_CONFORMER_CHECKPOINT_PATH = "/path/to/SASV2_Baseline/mfa_conformer_stage3.model"

# ---------------------------------------------------------------------------
# UBM training corpus (population models + fusion-classifier fitting set)
# ---------------------------------------------------------------------------
ASVSPOOF2019_LA_TRAIN_PROTOCOL = "/path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
ASVSPOOF2019_LA_TRAIN_AUDIO_ROOT = "/path/to/LA/ASVspoof2019_LA_train/flac"

# ---------------------------------------------------------------------------
# Evaluation datasets
# ---------------------------------------------------------------------------
ASVSPOOF2021_ENROLL_PROTOCOLS = [
    "/path/to/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt",
    "/path/to/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
]
ASVSPOOF2021_ENROLL_AUDIO_ROOT = "/path/to/LA/ASVspoof2019_LA_eval/flac"
ASVSPOOF2021_TRIAL_PROTOCOL = "/path/to/LA/ASVspoof2021_LA_keys/LA/ASV/trial_metadata.txt"
ASVSPOOF2021_TEST_AUDIO_ROOT = "/path/to/LA/ASVspoof2021_LA_eval/flac"

ASVSPOOF5_ROOT = "/path/to/ASVspoof5"
ASVSPOOF5_AUDIO_ROOT = f"{ASVSPOOF5_ROOT}/release_eval/flac_E_eval"
ASVSPOOF5_ENROLL_PROTOCOL = f"{ASVSPOOF5_ROOT}/protocol/ASVspoof5.eval.track_2.enroll.tsv"
ASVSPOOF5_TRIAL_PROTOCOL = f"{ASVSPOOF5_ROOT}/protocol/ASVspoof5.eval.track_2.trial.tsv"

SPOOFCELEB_ROOT = "/path/to/spoofceleb"
SPOOFCELEB_METADATA_CSV = f"{SPOOFCELEB_ROOT}/metadata/evaluation.csv"
SPOOFCELEB_PROTOCOL_CSV = f"{SPOOFCELEB_ROOT}/protocol/sasv_evaluation_evaluation_protocol.csv"
SPOOFCELEB_AUDIO_ROOT = f"{SPOOFCELEB_ROOT}/flac/evaluation"

# FamousFigures (issf/famousfigures on HuggingFace -- gated; see README) has no official
# enroll/trial split, so we build one ourselves (see datasets/famousfigures.py). Point this at
# wherever you downloaded the per-speaker parquet shards.
FAMOUSFIGURES_PARQUET_DIR = "/path/to/FamousFigures/data"
FAMOUSFIGURES_SPEAKERS = [
    "Anthony_Blinken", "Barack_Obama", "Donald_Trump", "Elon_Musk", "JD_Vance", "Joe_Biden",
    "Kamala_Harris", "Mathew_Miller", "Tim_Walz", "Vivek_Ramaswamy",
]

# ---------------------------------------------------------------------------
# Cache / output directories
# ---------------------------------------------------------------------------
PRETRAINED_MODELS_DIR = "pretrained_models"   # UBM caches, fusion classifier, baseline classifier
RESULTS_DIR = "results"
SCRATCH_DIR = "scratch"                       # VAD-truncated audio cache, dataset protocol csvs

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TOP_K = 12                     # top-K phonemes kept per branch (Section: Utility-Based Top-K)
ENROLLMENT_GMM_COMPONENTS = 5  # per-speaker phoneme GMM (Branch 2)
POPULATION_GMM_COMPONENTS = 3  # population cohort/bona-fide/spoof GMMs (Branch 3 + Branch 2 cohort)
DURATIONS_SEC = [1, 2, 3, 4]   # short-utterance evaluation conditions (plus "full")
