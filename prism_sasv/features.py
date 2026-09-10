"""
Shared frontend for Branches 2 and 3: a frozen CTC phoneme aligner and a frozen SSL feature
extractor, run independently (in parallel) on the same waveform. The aligner's phoneme
boundaries are then used to mean-pool the SSL frame features into one vector per phoneme
instance.
"""
from typing import List, Dict

import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from .backbones.mms import MMS300M


class PhonemeFeatureExtractor:
    """Aligns an utterance into phonemes (CTC) and mean-pools SSL frame features per phoneme.

    Phoneme aligner: facebook/wav2vec2-xlsr-53-espeak-cv-ft (CTC).
    SSL frontend: MMS-300M (nii-yamagishilab/mms-300m-anti-deepfake).
    """

    def __init__(self, aligner_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[*] Loading phoneme aligner (CTC): {aligner_name}")
        self.asr_processor = Wav2Vec2Processor.from_pretrained(aligner_name)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(aligner_name).to(self.device).eval()

        print("[*] Loading SSL feature extractor: MMS-300M")
        self.feat_model = MMS300M().to(self.device).eval()

    def get_phoneme_features_from_file(self, audio_path: str) -> List[Dict]:
        """Returns a list of {"char": phoneme_label, "feature": np.ndarray[D]} per phoneme
        instance in the utterance, in temporal order."""
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            inputs = self.asr_processor(audio, return_tensors="pt", sampling_rate=16000)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                asr_logits = self.asr_model(input_values).logits
                predicted_ids = torch.argmax(asr_logits, dim=-1)
                transcription = self.asr_processor.decode(predicted_ids[0], output_char_offsets=True)
                features = self.feat_model(input_values).squeeze(0)

            raw_offsets = transcription.char_offsets
            if not raw_offsets:
                return []

            num_frames = features.shape[0]
            out = []
            for u in raw_offsets:
                s_idx, e_idx = u["start_offset"], u["end_offset"]
                if s_idx >= num_frames:
                    continue
                e_idx = min(e_idx + 1, num_frames)
                segment = features[s_idx:e_idx]
                if segment.size(0) > 0:
                    out.append({"char": u["char"], "feature": segment.mean(dim=0).cpu().numpy()})
            return out
        except Exception as e:
            print(f"[!] Error processing {audio_path}: {e}")
            return []
