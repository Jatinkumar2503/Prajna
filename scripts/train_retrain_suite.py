"""
PRAJNA RETRAINING SUITE (AUDIT REMEDIATION)
Retrains PRAJNA safety models on:
1. 12 Physically Observable Plant Telemetry Channels (eliminating unobservable state leakage)
2. Realistic Sensor Degradation & Instrument Noise (Gaussian noise, RTD/SPND lag, ADC quantization, drift)
3. Corrected Conservation-Only Physics Loss (no DNBR/H2 penalties; signed enthalpy residual; reverse gradient penalty)
4. Saves newly trained checkpoints and logs SHA-256 cryptographic hashes.
"""

import os
import sys
import time
import math
import hashlib
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure root package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.physics import PrajnaPhysicsLoss
from prajna_core.simulator import PhysicalPHWRSimulator
from prajna_core.noise import apply_instrument_noise_suite
from prajna_core.models.fast_reflex import PrajnaFastReflex
from prajna_core.models.pinn_foundation import PrajnaFoundationPINN

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_12ch_transient_dataset(num_samples: int = 4000,
                                    seq_len: int = 45,
                                    seed: int = 42,
                                    apply_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates physically consistent 12-channel dataset via PhysicalPHWRSimulator:
    - X_obs: [Batch, SeqLen, 12] observable plant telemetry channels
    - Y_latent: [Batch, SeqLen, 4] unobservable internal states (quality, delta_t, clad_temp, precursors)
    - Y_eop: [Batch] scenario ground truth class
    """
    torch.manual_seed(seed)
    sim = PhysicalPHWRSimulator()
    
    samples_per_class = num_samples // 5
    remainder = num_samples % 5
    
    all_obs = []
    all_latent = []
    all_eop = []
    
    for s_id in range(5):
        n_class = samples_per_class + (1 if s_id < remainder else 0)
        if n_class <= 0:
            continue
            
        # Simulate physical closed-loop transient with exact matrix exponential
        res = sim.simulate_transient(
            scenario_id=s_id,
            duration_seconds=float(seq_len),
            dt=1.0,
            batch_size=n_class,
            seed=seed + s_id * 100
        )
        obs = res["obs"]  # [n_class, seq_len, 12]
        
        # Add diverse initial condition stochastic variation per sample (±1.5% around nominal)
        var_scale = 1.0 + (torch.rand(n_class, 1, 12) - 0.5) * 0.03
        obs = obs * var_scale
        
        t_out = obs[:, :, 0:1]
        t_in = obs[:, :, 10:11]
        dt = t_out - t_in
        clad = t_out + 45.0
        sq = torch.clamp((t_out - 290.0) * 0.005, min=0.01, max=0.25)
        prec = obs[:, :, 2:3] / 2.25
        
        latent = torch.cat([sq, dt, clad, prec], dim=-1)
        
        if apply_noise:
            obs = apply_instrument_noise_suite(obs, noise_scale=1.0)
            
        all_obs.append(obs)
        all_latent.append(latent)
        all_eop.append(torch.full((n_class,), s_id, dtype=torch.long))
        
    X_obs = torch.cat(all_obs, dim=0)
    Y_latent = torch.cat(all_latent, dim=0)
    Y_eop = torch.cat(all_eop, dim=0)
    
    # Randomly permute samples
    perm = torch.randperm(num_samples)
    return X_obs[perm], Y_latent[perm], Y_eop[perm]


def retrain_reflex_model(device: torch.device, epochs: int = 15) -> str:
    print("\n" + "=" * 80)
    print("  [1/2] RETRAINING PRAJNA REFLEX STUDENT MODEL (12 OBSERVABLE CHANNELS)")
    print("=" * 80)
    
    model = PrajnaFastReflex(num_channels=12, hidden_dim=96, num_eop_classes=64).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Instantiated PrajnaFastReflex with 12 channels. Exact parameter count: {param_count:,}")
    
    # Generate noisy training and validation sets
    X_train, _, Y_train = generate_12ch_transient_dataset(num_samples=3000, seed=42, apply_noise=True)
    X_val, _, Y_val = generate_12ch_transient_dataset(num_samples=1000, seed=100, apply_noise=True)
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    save_path = "checkpoints/prajna_reflex_12ch_noisy.pt"
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out["eop_logits"], by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                preds = torch.argmax(out["eop_logits"], dim=-1)
                val_correct += (preds == by).sum().item()
                
        val_acc = (val_correct / len(val_loader.dataset)) * 100.0
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_channels": 12,
                "val_acc": val_acc,
                "epoch": epoch,
                "param_count": param_count
            }, save_path)
            
    sha256 = compute_file_sha256(save_path)
    print(f"[+] Reflex Model Retrained Successfully: Best Val Acc = {best_acc:.2f}%")
    print(f"[+] Checkpoint: {save_path} (SHA-256: {sha256})")
    return save_path


def retrain_pinn_model(device: torch.device, epochs: int = 10) -> str:
    print("\n" + "=" * 80)
    print("  [2/2] RETRAINING PRAJNA PINN 60M BASELINE (12 CHANNELS + CONSERVATION LOSS)")
    print("=" * 80)
    
    model = PrajnaFoundationPINN(num_channels=12, scale="pinn_60m").to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Instantiated PrajnaFoundationPINN (pinn_60m, 12 channels). Exact parameters: {param_count:,}")
    
    loss_fn = PrajnaPhysicsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_train, _, Y_train = generate_12ch_transient_dataset(num_samples=2000, seed=42, apply_noise=True)
    X_val, _, Y_val = generate_12ch_transient_dataset(num_samples=500, seed=100, apply_noise=True)
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=16, shuffle=False)
    
    save_path = "checkpoints/prajna_pinn_60m_12ch_noisy.pt"
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_energy_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            out = model(bx)
            losses = loss_fn(
                pred_physics=out["physics_trajectories"],
                target_physics=bx,
                eop_logits=out["eop_logits"],
                target_eop=by
            )
            loss = losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item() * bx.size(0)
            total_energy_loss += losses["energy_loss"].item() * bx.size(0)
            
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_energy = total_energy_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                losses = loss_fn(
                    pred_physics=out["physics_trajectories"],
                    target_physics=bx,
                    eop_logits=out["eop_logits"],
                    target_eop=by
                )
                val_loss += losses["total_loss"].item() * bx.size(0)
                preds = torch.argmax(out["eop_logits"], dim=-1)
                val_correct += (preds == by).sum().item()
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = (val_correct / len(val_loader.dataset)) * 100.0
        
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} (Energy: {avg_energy:.6f}) | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "scale": "pinn_60m",
                "num_channels": 12,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "param_count": param_count
            }, save_path)
            
    sha256 = compute_file_sha256(save_path)
    print(f"[+] PINN 60M Retrained Successfully: Best Val Loss = {best_val_loss:.4f}")
    print(f"[+] Checkpoint: {save_path} (SHA-256: {sha256})")
    return save_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Retraining Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")
    
    t0 = time.time()
    reflex_ckpt = retrain_reflex_model(device, epochs=15)
    pinn_ckpt = retrain_pinn_model(device, epochs=8)
    
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"  RETRAINING COMPLETED IN {total_time:.2f}s")
    print(f"  Reflex Checkpoint: {reflex_ckpt}")
    print(f"  PINN Checkpoint:   {pinn_ckpt}")
    print("=" * 80)
