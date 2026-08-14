"""
PRAJNA FAST KNOWLEDGE DISTILLATION ENGINE (Cached Teacher Outputs)
Strategy:
  Phase 1: Run teacher ONCE over all 100K samples → cache soft targets to GPU tensors
  Phase 2: Train 25K student purely from cached targets (no teacher in the loop)
  
Result: ~5-10 minutes total instead of 1.5-2 hours.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure root package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.models.pinn_foundation import PrajnaFoundationPINN
from prajna_core.models.fast_reflex import PrajnaFastReflex
from scripts.train_pinn_production import generate_multi_physics_dataset


def distill_fast(
    teacher_checkpoint: str = "checkpoints/prajna_pinn_foundation_1b_best.pt",
    output_checkpoint: str = "checkpoints/prajna_reflex_25k_best.pt",
    samples: int = 100000,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 3.0,
    alpha: float = 0.7
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("  PRAJNA FAST KNOWLEDGE DISTILLATION (Cached Teacher Strategy)")
    print(f"  Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print(f"  Samples: {samples:,} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"  Temperature: {temperature} | Alpha: {alpha}")
    print("=" * 80)
    
    # =====================================================================
    # PHASE 1: Generate Dataset
    # =====================================================================
    print("\n[Phase 1/3] Generating Monte Carlo reactor transient dataset...")
    t0 = time.time()
    X_all, y_eop_all = generate_multi_physics_dataset(num_samples=samples, seq_len=45, num_channels=16)
    print(f"  Dataset generated in {time.time() - t0:.1f}s | Shape: {X_all.shape}")
    
    # =====================================================================
    # PHASE 2: Cache ALL Teacher Outputs in ONE Pass (No teacher in training loop)
    # =====================================================================
    print("\n[Phase 2/3] Caching teacher soft targets (single forward pass)...")
    
    if not os.path.exists(teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint}")
    
    ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    scale = ckpt.get("scale", "foundation_1b")
    
    teacher = PrajnaFoundationPINN(num_channels=16, scale=scale).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"  Loaded frozen teacher ({scale}): {teacher.count_parameters():,} parameters")
    
    # Pre-allocate cached target tensors
    cached_eop_logits = torch.zeros(samples, 64, dtype=torch.float32)
    cached_ttl = torch.zeros(samples, 16, dtype=torch.float32)
    
    cache_loader = DataLoader(TensorDataset(X_all), batch_size=batch_size, shuffle=False)
    
    t1 = time.time()
    idx = 0
    total_batches = len(cache_loader)
    
    with torch.no_grad():
        for batch_i, (x_batch,) in enumerate(cache_loader):
            x_batch = x_batch.to(device)
            out = teacher(x_batch)
            bs = x_batch.shape[0]
            cached_eop_logits[idx:idx+bs] = out["eop_logits"].cpu()
            cached_ttl[idx:idx+bs] = out["time_to_threshold"].cpu()
            idx += bs
            
            if (batch_i + 1) % 100 == 0 or (batch_i + 1) == total_batches:
                elapsed = time.time() - t1
                throughput = idx / elapsed
                eta = (samples - idx) / throughput if throughput > 0 else 0
                print(f"    Cached {idx:,}/{samples:,} samples | {throughput:.0f} samples/s | ETA: {eta:.0f}s")
    
    cache_time = time.time() - t1
    print(f"  [OK] Teacher caching complete in {cache_time:.1f}s ({samples/cache_time:.0f} samples/s)")
    
    # FREE teacher from GPU memory -- no longer needed!
    del teacher, ckpt
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"  [OK] Teacher removed from GPU -- all VRAM now free for student training")
    
    # =====================================================================
    # PHASE 3: Train Student from Cached Targets (BLAZING FAST)
    # =====================================================================
    print(f"\n[Phase 3/3] Training 25K student model from cached targets...")
    
    student = PrajnaFastReflex(num_channels=16, hidden_dim=96, num_eop_classes=64).to(device)
    param_count = student.count_parameters()
    mem_kb = param_count * 4 / 1024
    print(f"  Student: {param_count:,} parameters ({mem_kb:.1f} KB) | Target latency: <=4us")
    
    # Build train/val split from cached data
    val_size = int(0.15 * samples)
    train_size = samples - val_size
    
    # Shuffle indices
    perm = torch.randperm(samples)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    
    train_dataset = TensorDataset(
        X_all[train_idx], y_eop_all[train_idx],
        cached_eop_logits[train_idx], cached_ttl[train_idx]
    )
    val_dataset = TensorDataset(
        X_all[val_idx], y_eop_all[val_idx],
        cached_eop_logits[val_idx], cached_ttl[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce
# [Streaming pipeline module loaded]
