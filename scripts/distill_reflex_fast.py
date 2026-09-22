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
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
    
    t2 = time.time()
    
    for epoch in range(1, epochs + 1):
        student.train()
        train_loss = 0.0
        train_kl = 0.0
        train_acc = 0.0
        batches = 0
        
        for x_batch, y_eop_batch, t_eop_logits, t_ttl in train_loader:
            x_batch = x_batch.to(device)
            y_eop_batch = y_eop_batch.to(device)
            t_eop_logits = t_eop_logits.to(device)
            t_ttl = t_ttl.to(device)
            
            # Student forward (25K params — microseconds)
            student_out = student(x_batch)
            s_eop_logits = student_out["eop_logits"]
            s_ttl = student_out["time_to_threshold"]
            s_scram = student_out["scram_probability"]
            
            # 1. Soft Target KL Divergence (Temperature-scaled)
            p_s = F.log_softmax(s_eop_logits / temperature, dim=-1)
            p_t = F.softmax(t_eop_logits / temperature, dim=-1)
            loss_kd = kl_loss_fn(p_s, p_t) * (temperature ** 2)
            
            # 2. Hard Target Cross-Entropy
            loss_ce = ce_loss_fn(s_eop_logits, y_eop_batch)
            
            # 3. TTL Regression (mimic teacher countdown)
            loss_ttl = mse_loss_fn(s_ttl, t_ttl)
            
            # 4. SCRAM Interlock (abnormal = 1, normal = 0)
            is_abnormal = (y_eop_batch > 0).float().unsqueeze(-1)
            loss_scram = F.binary_cross_entropy(s_scram, is_abnormal)
            
            # Composite
            total_loss = (alpha * loss_kd) + ((1 - alpha) * loss_ce) + torch.mul(loss_ttl, 0.5) + torch.mul(loss_scram, 0.5)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_kl += loss_kd.item()
            preds = torch.argmax(s_eop_logits, dim=-1)
            train_acc += (preds == y_eop_batch).float().mean().item()
            batches += 1
        
        scheduler.step()
        
        # Validation
        student.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_val, y_val_eop, t_val_eop, t_val_ttl in val_loader:
                x_val = x_val.to(device)
                y_val_eop = y_val_eop.to(device)
                t_val_eop = t_val_eop.to(device)
                t_val_ttl = t_val_ttl.to(device)
                
                s_out = student(x_val)
                
                p_s = F.log_softmax(s_out["eop_logits"] / temperature, dim=-1)
                p_t = F.softmax(t_val_eop / temperature, dim=-1)
                loss_kd = kl_loss_fn(p_s, p_t) * (temperature ** 2)
                loss_ce = ce_loss_fn(s_out["eop_logits"], y_val_eop)
                loss_ttl = mse_loss_fn(s_out["time_to_threshold"], t_val_ttl)
                
                v_loss = (alpha * loss_kd) + ((1 - alpha) * loss_ce) + torch.mul(loss_ttl, 0.5)
                val_loss += v_loss.item()
                
                v_preds = torch.argmax(s_out["eop_logits"], dim=-1)
                val_acc += (v_preds == y_val_eop).float().mean().item()
                val_batches += 1
        
        avg_train = train_loss / batches
        avg_val = val_loss / val_batches
        avg_acc = (val_acc / val_batches) * 100.0
        epoch_time = time.time() - t2
        
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train: {avg_train:.4f} | Val: {avg_val:.4f} | EOP Acc: {avg_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.1f}s")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state_dict": student.state_dict(),
                "val_loss": avg_val,
                "eop_accuracy": avg_acc,
                "parameters": param_count,
                "teacher_checkpoint": teacher_checkpoint,
                "temperature": temperature,
                "alpha": alpha,
                "scale": "fast_reflex_25k",
                "hidden_dim": 96,
                "num_eop_classes": 64,
                "target_latency_us": 4.0
            }, output_checkpoint)
            print(f"    [OK] Best checkpoint saved -> {output_checkpoint}")
    
    total_time = time.time() - t0
    train_time = time.time() - t2
    
    print("\n" + "=" * 80)
    print(f"  DISTILLATION COMPLETE")
    print(f"  Teacher caching:  {cache_time:.1f}s (one-time)")
    print(f"  Student training: {train_time:.1f}s ({epochs} epochs)")
    print(f"  Total wall time:  {total_time:.1f}s")
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print(f"  Checkpoint:       {output_checkpoint}")
    print(f"  Model size:       {param_count:,} params ({mem_kb:.1f} KB)")
    print(f"  Target latency:   <=4 us (L1/L2 cache resident)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prajna Fast Knowledge Distillation")
    parser.add_argument("--teacher", type=str, default="checkpoints/prajna_pinn_foundation_1b_best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/prajna_reflex_25k_best.pt")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()
    
    distill_fast(
        teacher_checkpoint=args.teacher,
        output_checkpoint=args.output,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha
    )
