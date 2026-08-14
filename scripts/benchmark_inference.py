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
                   seq_len: int = 45,
                   warmup_runs: int = 20,
                   benchmark_runs: int = 100):
    import onnxruntime as ort
    
    print(f"\n[*] Benchmarking ONNX Runtime (Graph: {os.path.basename(onnx_path)}, Batch: {batch_size}, SeqLen: {seq_len})...")
    
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
        
    session = ort.InferenceSession(onnx_path, providers=providers)
    dummy_input = np.random.randn(batch_size, seq_len, 16).astype(np.float32)
    ort_inputs = {"telemetry_sequence": dummy_input}
    
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run(None, ort_inputs)
        
    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    latencies = np.array(latencies)
    print_benchmark_results(f"ONNX Runtime ({providers[0]})", latencies, batch_size)


def print_benchmark_results(name: str, latencies_ms: np.ndarray, batch_size: int):
    mean_lat = np.mean(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p90 = np.percentile(latencies_ms, 90)
    p99 = np.percentile(latencies_ms, 99)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    fps = (batch_size * 1000.0) / mean_lat
    
    print("-" * 70)
    print(f"  {name.upper()}")
    print("-" * 70)
    print(f"  Mean Latency:    {mean_lat:6.2f} ms")
    print(f"  P50 (Median):    {p50:6.2f} ms")
    print(f"  P90:             {p90:6.2f} ms")
    print(f"  P99:             {p99:6.2f} ms")
    print(f"  Min / Max:       {min_lat:6.2f} ms / {max_lat:6.2f} ms")
    print(f"  Throughput:      {fps:6.1f} inferences/sec (FPS)")
    print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prajna Inference Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/prajna_pinn_efficient_125m_best.pt", help="PyTorch checkpoint")
    parser.add_argument("--onnx", type=str, default=None, help="ONNX model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=45, help="Sequence length")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark repetitions")
    args = parser.parse_args()
    
    print("=" * 80)
    print("  PRAJNA HIGH-PERFORMANCE INFERENCE BENCHMARK")
    print("=" * 80)
    
    if os.path.exists(args.checkpoint):
        benchmark_pytorch(args.checkpoint, batch_size=args.batch_size, seq_len=args.seq_len, benchmark_runs=args.runs)
        
    if args.onnx and os.path.exists(args.onnx):
        benchmark_onnx(args.onnx, batch_size=args.batch_size, seq_len=args.seq_len, benchmark_runs=args.runs)
