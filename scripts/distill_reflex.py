"""
PRAJNA KNOWLEDGE DISTILLATION ENGINE
Distills the 264.7M Parameter Foundation PINN Teacher into the 30k PrajnaFastReflex Student.
Transfers:
1. IAEA Emergency Operating Procedure (EOP) soft logit probability distributions (Temperature T=3.0)
2. Time-to-Threshold (TTL) excursion countdown horizons
3. Instant SCRAM Emergency Interlock classification
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


def distill_prajna_reflex(
    teacher_checkpoint: str = "checkpoints/prajna_pinn_foundation_1b_best.pt",
    output_checkpoint: str = "checkpoints/prajna_reflex_25k_best.pt",
    samples: int = 25000,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 3.0,
    alpha: float = 0.7
):
    print("=" * 80)
    print("  PRAJNA KNOWLEDGE DISTILLATION ENGINE")
    print(f"  Teacher Model:     {teacher_checkpoint}")
    print(f"  Student Model:     PrajnaFastReflex (~30k Parameters)")
    print(f"  Distillation Temp: {temperature:.1f} | Alpha: {alpha:.2f}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Distillation Compute Device: {device}")
    
    # 1. Load Frozen Teacher Model
    if not os.path.exists(teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found at: {teacher_checkpoint}")
        
    ckpt = torch.load(teacher_checkpoint, map_location=device)
    scale = ckpt.get("scale", "foundation_1b")
    
    teacher = PrajnaFoundationPINN(num_channels=16, scale=scale).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    
    # Freeze all teacher parameters strictly
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"[+] Loaded Frozen Teacher PINN ({scale}): {teacher.count_parameters():,} parameters (READ-ONLY)")
    
    # 2. Instantiate Student Reflex Model
    student = PrajnaFastReflex(num_channels=16, hidden_dim=96, num_eop_classes=64).to(device)
    print(f"[+] Initialized Student Reflex Model: {student.count_parameters():,} parameters")
    
    # 3. Generate Continuous Multi-Physics Monte Carlo Training Dataset
    print(f"\n[*] Generating {samples:,} Continuous Monte Carlo Multi-Physics Scenarios for Distillation...")
    X_train, y_train_eop = generate_multi_physics_dataset(num_samples=samples, seq_len=45, num_channels=16)
    
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train_eop)
    val_size = int(0.15 * samples)
    train_size = samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
    
    start_time = time.time()
    print("\n[*] Commencing Knowledge Distillation Loop...")
    
    for epoch in range(1, epochs + 1):
        student.train()
        train_loss = 0.0
        train_kl = 0.0
        train_acc = 0.0
        batches = 0
        
        for x_batch, y_eop_batch in train_loader:
            x_batch = x_batch.to(device)
            y_eop_batch = y_eop_batch.to(device)
            
            # Forward Teacher (No Gradients)
            with torch.no_grad():
                teacher_out = teacher(x_batch)
                t_eop_logits = teacher_out["eop_logits"]
                t_ttl = teacher_out["time_to_threshold"]
            
            # Forward Student
            student_out = student(x_batch)
            s_eop_logits = student_out["eop_logits"]
            s_ttl = student_out["time_to_threshold"]
            s_scram = student_out["scram_probability"]
            
            # 1. Soft Target Distillation Loss (KL Divergence with Temperature)
            p_s = F.log_softmax(s_eop_logits / temperature, dim=-1)
            p_t = F.softmax(t_eop_logits / temperature, dim=-1)
            loss_kd = kl_loss_fn(p_s, p_t) * (temperature ** 2)
            
            # 2. Hard Target Ground Truth Loss
            loss_ce = ce_loss_fn(s_eop_logits, y_eop_batch)
            
            # 3. TTL Regression Loss (Mimicking teacher's countdown horizon)
            loss_ttl = mse_loss_fn(s_ttl, t_ttl)
            
            # 4. SCRAM Interlock Loss (1 for abnormal transient, 0 for normal)
            is_abnormal = (y_eop_batch > 0).float().unsqueeze(-1)
            loss_scram = F.binary_cross_entropy(s_scram, is_abnormal)
            
            # Composite Distillation Objective
            total_loss = (alpha * loss_kd) + ((1 - alpha) * loss_ce) + (0.5 * loss_ttl) + (0.5 * loss_scram)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_loss += total_loss.item()
            train_kl += loss_kd.item()
            preds = torch.argmax(s_eop_logits, dim=-1)
            train_acc += (preds == y_eop_batch).float().mean().item()
            batches += 1
            
        scheduler.step()
        
        # Validation Evaluation
        student.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_val, y_val_eop in val_loader:
                x_val = x_val.to(device)
                y_val_eop = y_val_eop.to(device)
                
                t_out = teacher(x_val)
                s_out = student(x_val)
                
                p_s = F.log_softmax(s_out["eop_logits"] / temperature, dim=-1)
                p_t = F.softmax(t_out["eop_logits"] / temperature, dim=-1)
                loss_kd = kl_loss_fn(p_s, p_t) * (temperature ** 2)
                loss_ce = ce_loss_fn(s_out["eop_logits"], y_val_eop)
                loss_ttl = mse_loss_fn(s_out["time_to_threshold"], t_out["time_to_threshold"])
                
                v_loss = (alpha * loss_kd) + ((1 - alpha) * loss_ce) + (0.5 * loss_ttl)
                val_loss += v_loss.item()
                
                v_preds = torch.argmax(s_out["eop_logits"], dim=-1)
                val_acc += (v_preds == y_val_eop).float().mean().item()
                val_batches += 1
                
        avg_train_loss = train_loss / batches
        avg_val_loss = val_loss / val_batches
        avg_val_acc = (val_acc / val_batches) * 100.0
        
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | EOP Acc: {avg_val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save Best Student Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": student.state_dict(),
                "val_loss": avg_val_loss,
                "eop_accuracy": avg_val_acc,
                "parameters": student.count_parameters(),
                "teacher_checkpoint": teacher_checkpoint,
                "temperature": temperature,
                "alpha": alpha
            }, output_checkpoint)
            print(f"    [+] Saved Best Student Checkpoint -> {output_checkpoint} (Val Loss: {avg_val_loss:.4f})")
            
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[OK] Distillation Completed in {elapsed:.2f}s.")
    print(f"[+] Optimal Student Model Saved: {output_checkpoint}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prajna Knowledge Distillation Harness")
    parser.add_argument("--teacher", type=str, default="checkpoints/prajna_pinn_foundation_1b_best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/prajna_reflex_25k_best.pt")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    distill_prajna_reflex(
        teacher_checkpoint=args.teacher,
        output_checkpoint=args.output,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
