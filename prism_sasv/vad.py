"""VAD-based utterance truncation for the short-utterance robustness evaluation.

Locates where speech STARTS in an utterance (via webrtcvad), then cuts a single contiguous
target_sec window from the ORIGINAL waveform starting at that onset -- natural pauses/silence
inside that window are kept as-is (not stripped or concatenated). This matches a realistic
deployment scenario where a short probe clip is a plain contiguous recording, not an
artificially speech-dense edit.
"""
import numpy as np
import librosa
import soundfile as sf
import webrtcvad

VAD_SR = 16000
FRAME_MS = 30  # webrtcvad requires 10/20/30ms frames
FRAME_LEN = int(VAD_SR * FRAME_MS / 1000)


def _to_pcm16_bytes(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


def find_speech_onset(wav, sr=VAD_SR, aggressiveness=2):
    """Returns the sample index of the first voiced frame, or 0 if none is found."""
    if sr != VAD_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=VAD_SR)
    vad = webrtcvad.Vad(aggressiveness)
    n_frames = len(wav) // FRAME_LEN
    for i in range(n_frames):
        frame = wav[i * FRAME_LEN:(i + 1) * FRAME_LEN]
        pcm = _to_pcm16_bytes(frame)
        try:
            is_speech = vad.is_speech(pcm, VAD_SR)
        except Exception:
            is_speech = True  # malformed frame -- treat as speech rather than silently skip
        if is_speech:
            return i * FRAME_LEN
    return 0


def truncate_to_duration(audio_path, target_sec, out_path, aggressiveness=2, min_available_sec=0.3):
    """Loads audio_path, finds the speech onset via VAD, and writes a contiguous target_sec clip
    starting there to out_path. Returns True on success, False if too little audio remained after
    the onset (caller should skip this trial for this duration)."""
    wav, sr = librosa.load(audio_path, sr=VAD_SR)
    onset = find_speech_onset(wav, VAD_SR, aggressiveness)
    remaining = wav[onset:]
    if len(remaining) < int(min_available_sec * VAD_SR):
        return False
    n_samples = int(target_sec * VAD_SR)
    clip = remaining[:n_samples]
    sf.write(out_path, clip, VAD_SR)
    return True
