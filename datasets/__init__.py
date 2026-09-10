"""
Each dataset module exposes:

    load_enrollment_and_trials(seed=42) -> (enroll_paths, trials_df, resolve_path)

    enroll_paths: {speaker: [audio_path, ...]}
    trials_df:    pandas.DataFrame with columns claimed_speaker, utt, hyp (0=target, 1=nontarget,
                  2=spoof)
    resolve_path: callable(utt) -> full-duration audio path (for the "full utterance" evaluation
                  and as the source audio for VAD-based short-utterance truncation)
"""
from . import asvspoof2021, asvspoof5, spoofceleb, famousfigures

REGISTRY = {
    asvspoof2021.NAME: asvspoof2021,
    asvspoof5.NAME: asvspoof5,
    spoofceleb.NAME: spoofceleb,
    famousfigures.NAME: famousfigures,
}
