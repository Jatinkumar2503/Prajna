"""
PRAJNA FAST-REFLEX ONNX & C++ EXPORT ENGINE
Converts trained 25k PrajnaFastReflex student model into:
1. Optimized ONNX computational graph (checkpoints/prajna_reflex_25k.onnx)
2. Standalone C++ Header / SIMD implementation (src/inference/prajna_reflex_simd.h)
3. Direct numerical parity validation (PyTorch vs ONNX Runtime)
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is in sys.path
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


def export_reflex_onnx(
    checkpoint_path: str = "checkpoints/prajna_reflex_25k_best.pt",
    output_onnx_path: str = "checkpoints/prajna_reflex_25k.onnx",
    output_cpp_header: str = "src/inference/prajna_reflex_simd.h",
    opset_version: int = 17
):
    print("=" * 80)
    print("  PRAJNA FAST-REFLEX EXPORT & SERIALIZATION ENGINE")
    print(f"  Source Checkpoint: {checkpoint_path}")
    print(f"  Target ONNX:       {output_onnx_path}")
    print(f"  Target C++ Header: {output_cpp_header}")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_channels = ckpt.get("num_channels", 16)
    hidden_dim = ckpt.get("hidden_dim", 96)
    num_eop = ckpt.get("num_eop_classes", 64)

    model = PrajnaFastReflex(num_channels=num_channels, hidden_dim=hidden_dim, num_eop_classes=num_eop).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = ReflexONNXWrapper(model).to(device)
    wrapper.eval()

    param_count = model.count_parameters()
    print(f"[+] Loaded Model Parameters: {param_count:,} (Size: {param_count * 4 / 1024:.2f} KB)")

    dummy_input = torch.randn(1, 45, 16, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    input_names = ["telemetry"]
    output_names = ["eop_logits", "time_to_threshold", "scram_probability", "reflex_latent"]

    print("\n[*] Exporting ONNX Computational Graph...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"telemetry": {0: "batch_size"}, "eop_logits": {0: "batch_size"}, "time_to_threshold": {0: "batch_size"}, "scram_probability": {0: "batch_size"}, "reflex_latent": {0: "batch_size"}},
        dynamo=False
    )
    onnx_size_kb = os.path.getsize(output_onnx_path) / 1024.0
    print(f"[+] Exported ONNX Graph successfully: {output_onnx_path} ({onnx_size_kb:.2f} KB)")

    # 2. Validate with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        test_x = torch.randn(2, 45, 16, dtype=torch.float32)

        with torch.no_grad():
            pt_eop, pt_ttl, pt_scram, pt_lat = wrapper(test_x)

        ort_outs = session.run(None, {"telemetry": test_x.numpy()})
        diff_eop = np.max(np.abs(pt_eop.numpy() - ort_outs[0]))
        diff_ttl = np.max(np.abs(pt_ttl.numpy() - ort_outs[1]))
        diff_scram = np.max(np.abs(pt_scram.numpy() - ort_outs[2]))

        print(f"[+] Parity Check: EOP Max Err: {diff_eop:.2e} | TTL Max Err: {diff_ttl:.2e} | SCRAM Max Err: {diff_s
# [Parity validation harness configured]
