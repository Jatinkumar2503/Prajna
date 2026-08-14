"""
PRAJNA INFERENCE BENCHMARKING HARNESS
Measures forward-pass latency, throughput (FPS), and percentile distributions (P50, P90, P99)
across PyTorch Eager and ONNX Runtime execution engines.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def benchmark_pytorch(checkpoint_path: str,
                      batch_size: int = 1,
                      seq_len: int = 45,
                      warmup_runs: int = 20,
                      benchmark_runs: int = 100):
    from prajna_core.models.pinn_foundation import PrajnaFoundationPINN
    from prajna_core.models.fast_reflex import PrajnaFastReflex
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    is_reflex = "reflex" in checkpoint_path or checkpoint.get("scale") == "fast_reflex_25k" or "num_eop_classes" in checkpoint
    
    if is_reflex:
        print(f"\n[*] Benchmarking Prajna Fast-Reflex on CPU L1/L2 Cache (Batch: {batch_size}, SeqLen: {seq_len})...")
        device = torch.device("cpu")
        model = PrajnaFastReflex(
            num_channels=checkpoint.get("num_channels", 16),
            hidden_dim=checkpoint.get("hidden_dim", 96),
            num_eop_classes=checkpoint.get("num_eop_classes", 64)
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        dummy_input = torch.randn(batch_size, seq_len, 16, device=device)
        warmup_runs = 200
        benchmark_runs = 1000
    else:
        print(f"\n[*] Benchmarking PyTorch Foundation PINN on {device} (Batch: {batch_size}, SeqLen: {seq_len})...")
        scale = checkpoint.get("scale", "efficient_125m")
        model = PrajnaFoundationPINN(num_channels=16, scale=scale).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        dummy_input = torch.randn(batch_size, seq_len, 16, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
            
    latencies = np.array(latencies)
    engine_name = "Prajna Fast-Reflex (CPU Cache)" if is_reflex else f"PyTorch Eager ({device})"
    print_benchmark_results(engine_name, latencies, batch_size)


def benchmark_onnx(onnx_path: str,
                   batch_size: int = 1,
                   s
# [Fast-Reflex benchmarking harness attached]
