"""
SSL frontend: MMS-300M (`nii-yamagishilab/mms-300m-anti-deepfake`), an anti-spoofing post-trained
variant of Meta's Massively Multilingual Speech (MMS) 300M model.

Checkpoint: https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake
Base model: https://huggingface.co/facebook/mms-300m
"""
import torch
from collections import OrderedDict
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config

from ._mms_config import MMS_CONFIGS, GLOBAL_OUTPUT_DIM

CHECKPOINT_PATH = "pretrained_models/mms_300m_anti_deepfake.ckpt"


class MMS300M(torch.nn.Module):
    def __init__(self, checkpoint_path=CHECKPOINT_PATH, freeze=True):
        super().__init__()
        cfg = Wav2Vec2Config(**MMS_CONFIGS)
        self.model = Wav2Vec2Model(cfg)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state = OrderedDict()
        for name, param in state_dict.items():
            if name.startswith("m_ssl.model."):
                state[name[len("m_ssl.model."):]] = param
        self.model.load_state_dict(state)
        self.out_dim = GLOBAL_OUTPUT_DIM

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def extract_feat(self, input_data):
        if (next(self.model.parameters()).device != input_data.device
                or next(self.model.parameters()).dtype != input_data.dtype):
            self.model.to(input_data.device, dtype=input_data.dtype)
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        return self.model(input_tmp, mask=False, features_only=True)["x"]

    def forward(self, input_data):
        return self.extract_feat(input_data)
