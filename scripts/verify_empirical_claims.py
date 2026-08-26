"""
PRAJNA EMPIRICAL CLAIMS AUDIT & SANITY CHECK SCRIPT
Inspects physical disk files, checkpoints, ONNX graphs, and benchmark logs
to produce 100% verified, reproducible metrics for judges.
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.models.pinn_foundation import PrajnaFoundationPINN
from prajna_core.models.fast_reflex import PrajnaFastReflex


def verify_all_empirical_claims():
    print("=" * 80)
    print("  PRAJNA EMPIRICAL CLAIMS AUDIT & VERIFICATION HARNESS")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. VERIFY CHECKPOINT FILE SIZE & PARAMETER COUNT
    # ---------------------------------------------------------
    ckpt_1b_path = "checkpoints/prajna_pinn_foundation_1b_best.pt"
    print(f"\n[1/4] AUDITING FOUNDATION MODEL CHECKPOINT: {ckpt_1b_path}")

    if os.path.exists(ckpt_1b_path):
        size_bytes = os.path.getsize(ckpt_1b_path)
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024 * 1024 * 1024)
        print(f"  • Disk File Size: {size_bytes:,} Bytes ({size_mb:.2f} MB | {size_gb:.3f} GB)")

        ckpt = torch.load(ckpt_1b_path, map_location="cpu", weights_only=False)
        scale = ckpt.get("scale", "foundation_1b")
        val_loss = ckpt.get("val_loss", None)
        epoch = ckpt.get("epoch", None)

        print(f"  • Checkpoint Metadata Scale: {scale}")
        print(f"  • Checkpoint Best Epoch:    {epoch}")
        print(f"  • Checkpoint Validation Loss: {val_loss}")

        # Instantiate model to count parameters
        model = PrajnaFoundationPINN(num_channels=16, scale=scale)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        total_params = model.count_parameters()
        fp32_theoretical_bytes = total_params * 4
        fp32_theoretical_mb = fp32_theoretical_bytes / (1024 * 1024)
        fp32_theoretical_gb = fp32_theoretical_bytes / (1024 * 1024 * 1024)

        print(f"  • Total Active Parameters:   {total_params:,}")
        print(f"  • Theoretical FP32 Size:     {fp32_theoretical_mb:.2f} MB ({fp32_theoretical_gb:.3f} GB)")
        print(f"  • Ratio (Disk vs Model):     {size_mb / fp32_theoretical_mb:.2f}x")
    else:
        print(f"  [!] Checkpoint file missing: {ckpt_1b_path}")

    # ---------------------------------------------------------
    # 2. VERIFY FAST-REFLEX STUDENT MODEL & INT8 ONNX FOOTPRINT
    # ---------------------------------------------------------
    ckpt_reflex_path = "checkpoints/prajna_reflex_25k_best.pt"
    onnx_fp32_path = "checkpoints/prajna_reflex_25k.onnx"
    onnx_int8_path = "checkpoints/prajna_reflex_25k_int8.onnx"

    print(f"\n[2/4] AUDITING FAST-REFLEX STUDENT & INT8 ONNX GRAPH")
    if os.path.exists(ckpt_reflex_path):
        ckpt_ref = torch.load(ckpt_reflex_path, map_location="cpu", weights_only=False)
        reflex_model = PrajnaFastReflex(num_channels=16, hidden_dim=96, num_eop_classes=64)
        reflex_model.load_state_dict(ckpt_ref["model_state_dict"])
        ref_params = reflex_model.count_parameters()
        print(f"  • Fast-Reflex Parameters:   {ref_params:,}")

    if os.path.exists(onnx_fp32_path):
        fp32_kb = os.path.getsize(onnx_fp32_path) / 1024.0
        print(f"  • ONNX FP32 Graph Size:     {fp32_kb:.2f} KB")

    if os.path.exists(onnx_int8_path):
        int8_kb = os.path.getsize(onnx_int8_path) / 1024.0
        print(f"  • ONNX INT8 Graph Size:     {int8_kb:.2f} KB")
        if os.path.exists(onnx_fp32_path):
            print(f"  • INT8 Compression Ratio:   {fp32_kb / max(0.1, int8_kb):.2f}x reduction")

    # ---------------------------------------------------------
    # 3. VERIFY ONNX RUNTIME LATENCY BENCHMARK (REAL-TIME CPU PROFILING)
    # ---------------------------------------------------------
    print(f"\n[3/4] PROFILING ONNX RUNTIME LATENCY & THROUGHPUT (N=1000 Iterations)")
    import onnxruntime as ort

    if os.path.exists(onnx_int8_path):
        session = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        
        # Test input matching live runtime shape (1, 45, 16)
        dummy_in = np.random.randn(1, 45, 16).astype(np.float32)

        # Warmup
        for _ in range(100):
            session.run(None, {input_name: dummy_in})

        timings_ms = []
        for _ in range(1000):
          t0 = time.perf_counter()
          session.run(None, {input_name: dummy_in})
          t1 = time.perf_counter()
          timings_ms.append((t1 - t0) * 1000.0)

        p50_ms = np.percentile(timings_ms, 50)
        p90_ms = np.percentile(timings_ms, 90)
        p99_ms = np.percentile(timings_ms, 99)
        p50_us = p50_ms * 1000.0
        fps = 1000.0 / max(1e-6, p50_ms)

        print(f"  • P50 Latency (Median):    {p50_ms:.3f} ms ({p50_us:.1f} µs)")
        print(f"  • P90 Latency (90th pct):  {p90_ms:.3f} ms ({p90_ms * 1000:.1f} µs)")
        print(f"  • P99 Latency (99th pct):  {p99_ms:.3f} ms ({p99_ms * 1000:.1f} µs)")
        print(f"  • P50 Throughput (FPS):     {fps:,.1f} Inferences/sec")

    # ---------------------------------------------------------
    # 4. VERIFY EVALUATION JSON LOGS & ACCURACY METRICS
    # ---------------------------------------------------------
    print(f"\n[4/4] AUDITING SAVED EVALUATION METRICS & LOSS LOGS")
    eval_json_path = "eval_results/prajna_eval_foundation_1b.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
        print(f"  • Evaluation Log File:     {eval_json_path}")
        print(f"  • Metrics Keys:            {list(eval_data.keys())}")
        if "metrics" in eval_data:
            m = eval_data["metrics"]
            print(f"    - Validation Loss:        {m.get('val_loss', 'N/A')}")
            print(f"    - Energy Balance Residual:{m.get('energy_loss', 'N/A')}")
            print(f"    - EOP Logit Loss:         {m.get('eop_loss', 'N/A')}")
            print(f"    - EOP Top-1 Accuracy:     {m.get('eop_accuracy', 'N/A')}")
    else:
        print(f"  [!] Evaluation log file missing: {eval_json_path}")

    print("\n" + "=" * 80)
    print("  EMPIRICAL SANITY CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    verify_all_empirical_claims()
