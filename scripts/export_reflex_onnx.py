"""
PRAJNA FAST-REFLEX ONNX & C++ EXPORT ENGINE
Converts trained 25k PrajnaFastReflex student model into optimized ONNX graph and C++ header.
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prajna_core.models.fast_reflex import PrajnaFastReflex

class ReflexONNXWrapper(nn.Module):
    def __init__(self, base_model: PrajnaFastReflex):
        super().__init__()
        self.model = base_model

    def forward(self, telemetry: torch.Tensor):
        out = self.model(telemetry)
        return (
            out["eop_logits"],
            out["time_to_threshold"],
            out["scram_probability"],
            out["reflex_latent"]
        )
