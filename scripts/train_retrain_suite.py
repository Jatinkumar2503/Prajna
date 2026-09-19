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
    Generates dataset with:
    - X_obs: [Batch, SeqLen, 12] observable plant telemetry channels
    - Y_latent: [Batch, SeqLen, 4] unobservable internal states (quality, delta_t, clad_temp, precursors)
    - Y_eop: [Batch] scenario ground truth class
    """
    torch.manual_seed(seed)
    
    t = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    p = t / float(seq_len)
    
    # 5 transient scenario classes
    scenarios = torch.arange(num_samples) % 5
    Y_eop = scenarios.clone()
    
    # 12 Observable Baseline Sensor Values
    # 0: Core Exit Temp (275 - 295 °C)
    # 1: Coolant Flow (70 - 86 kg/s)
    # 2: Neutron Flux (1.8 - 2.7 x10^13)
    # 3: Radiation (0.35 - 0.50 mSv/h)
    # 4: Primary Pressure (148 - 162 bar)
    # 5: Core Power (70 - 105 MWth)
    # 6: Control Rod Height (55 - 80 %)
    # 7: Pressurizer Level (42 - 58 %)
    # 8: Feedwater Temp (210 - 230 °C)
    # 9: Steam Flow (68 - 82 kg/s)
    # 10: Core Inlet Temp (247 - 267 °C)
    # 11: Containment Pressure (98 - 104 kPa)
    
    t_out_base = 275.0 + torch.rand(num_samples, 1, 1) * 20.0
    f_base = 70.0 + torch.rand(num_samples, 1, 1) * 16.0
    flux_base = 1.80 + torch.rand(num_samples, 1, 1) * 0.90
    rad_base = 0.35 + torch.rand(num_samples, 1, 1) * 0.15
    p_base = 148.0 + torch.rand(num_samples, 1, 1) * 14.0
    rod_base = 55.0 + torch.rand(num_samples, 1, 1) * 25.0
    pzr_base = 42.0 + torch.rand(num_samples, 1, 1) * 16.0
    fw_base = 210.0 + torch.rand(num_samples, 1, 1) * 20.0
    sf_base = 68.0 + torch.rand(num_samples, 1, 1) * 14.0
    t_in_base = t_out_base - 28.0  # Physical 28°C core delta-T at nominal flow
    pow_base = flux_base * 39.5
    cont_base = 98.0 + torch.rand(num_samples, 1, 1) * 6.0
    
    # 4 Latent / Virtual Sensor States (Targets, NOT inputs)
    sq_base = 0.02 * torch.ones(num_samples, 1, 1)
    dt_base = 28.0 * torch.ones(num_samples, 1, 1)
    clad_base = t_out_base + 45.0
    prec_base = 0.95 + torch.rand(num_samples, 1, 1) * 0.10
    
    X_obs = torch.zeros(num_samples, seq_len, 12, dtype=torch.float32)
    Y_latent = torch.zeros(num_samples, seq_len, 4, dtype=torch.float32)
    
    for s_id in range(5):
        mask = (scenarios == s_id).view(-1, 1, 1)
        if not mask.any():
            continue
            
        if s_id == 0:  # Steady State Normal
            w = 0.04
            t_out = t_out_base + torch.sin(t * w) * 0.8
            f = f_base + torch.cos(t * w) * 0.6
            flux = flux_base + torch.sin(t * w * 1.5) * 0.03
            rad = rad_base + torch.cos(t * w) * 0.015
            p_prim = p_base + torch.sin(t * w) * 0.4
            pow_th = flux * 39.5
            rod = rod_base + torch.cos(t * w) * 0.3
            pzr = pzr_base + torch.sin(t * w) * 0.4
            fw = fw_base + torch.cos(t * w) * 0.3
            sf = sf_base + torch.sin(t * w) * 0.5
            t_in = t_in_base + torch.sin(t * w) * 0.4
            cont = cont_base + torch.sin(t * 0.01) * 0.05
            
            sq = sq_base + torch.sin(t * w) * 0.002
            dt = t_out - t_in
            clad = t_out + 45.0
            prec = prec_base + torch.sin(t * w) * 0.005
            
        elif s_id == 1:  # LOCA (Pressure drop, thermal excursion, containment rise)
            gamma_loca = 1.0 + torch.rand(num_samples, 1, 1) * 0.4
            t_out = t_out_base + torch.pow(p, gamma_loca) * 65.0
            f = torch.clamp(f_base - p * 40.0, min=15.0)
            flux = torch.clamp(flux_base - p * 1.2, min=0.2)
            rad = rad_base + torch.pow(p, 1.6) * 3.8
            p_prim = torch.clamp(p_base - p * 55.0, min=60.0)
            pow_th = flux * 39.5
            rod = torch.clamp(rod_base - p * rod_base, min=0.0)
            pzr = torch.clamp(pzr_base - p * 35.0, min=10.0)
            fw = fw_base - p * 18.0
            sf = torch.clamp(sf_base - p * 50.0, min=15.0)
            t_in = t_in_base + p * 15.0  # Cold leg warms as cooling degrades
            cont = cont_base + p * 75.0
            
            sq = torch.clamp(sq_base + p * 0.18, max=0.30)
            dt = t_out - t_in
            clad = t_out + 45.0 + p * 40.0
            prec = torch.clamp(prec_base - p * 0.75, min=0.15)
            
        elif s_id == 2:  # Reactivity Insertion (Prompt power jump, Doppler feedback)
            gamma_ria = 1.1 + torch.rand(num_samples, 1, 1) * 0.5
            flux = flux_base + torch.pow(p, gamma_ria) * 3.5
            pow_th = flux * 39.5
            t_out = t_out_base + p * 55.0
            f = f_base - p * 5.0
            rad = rad_base + p * 1.5
            p_prim = p_base + p * 15.0
            rod = torch.clamp(rod_base + p * (100.0 - rod_base), max=100.0)
            pzr = pzr_base + p * 18.0
            fw = fw_base + p * 8.0
            sf = sf_base + p * 15.0
            t_in = t_in_base + p * 10.0
            cont = cont_base + p * 5.0
            
            sq = torch.clamp(sq_base + p * 0.08, max=0.22)
            dt = t_out - t_in
            clad = t_out + 45.0 + p * 30.0
            prec = prec_base + p * 1.2
            
        elif s_id == 3:  # SGTR / Feeder Break (Secondary radiation spike, primary depressurization)
            t_out = t_out_base + p * 25.0
            f = torch.clamp(f_base - p * 28.0, min=30.0)
            flux = flux_base - p * 0.4
            pow_th = flux * 39.5
            rad = rad_base + torch.pow(p, 1.4) * 3.5
            p_prim = torch.clamp(p_base - p * 35.0, min=95.0)
            rod = rod_base - p * 20.0
            pzr = torch.clamp(pzr_base - p * 28.0, min=15.0)
            fw = fw_base - p * 20.0
            sf = torch.clamp(sf_base - p * 30.0, min=25.0)
            t_in = t_in_base + p * 5.0
            cont = cont_base + p * 12.0
            
            sq = torch.clamp(sq_base + p * 0.05, max=0.20)
            dt = t_out - t_in
            clad = t_out + 45.0
            prec = prec_base - p * 0.3
            
        elif s_id == 4:  # Station Blackout (Coastdown flow to natural circ ~14 kg/s, trip to decay heat)
            ones = torch.ones(num_samples, 1, 1)
            p_decay = torch.clamp(3.5 + 4.0 * torch.exp(-p * 4.0), min=3.5) * ones
            pow_th = p_decay
            flux = pow_th / 39.5
            f = torch.clamp(f_base - p * 58.0, min=12.0)  # Coastdown to natural circulation
            t_out = t_out_base + torch.sin(p * math.pi) * 18.0
            rad = rad_base + p * 0.8
            p_prim = torch.clamp(p_base - p * 22.0, min=115.0)
            rod = torch.clamp(rod_base - p * rod_base, min=0.0)  # Full scram
            pzr = pzr_base - p * 16.0
            fw = torch.clamp(fw_base - p * 45.0, min=15.0)
            sf = torch.clamp(sf_base - p * 65.0, min=4.0)
            t_in = t_in_base + p * 4.0
            cont = cont_base + p * 8.0
            
            sq = torch.clamp(sq_base + p * 0.06, max=0.20)
            dt = t_out - t_in
            clad = t_out + 45.0
            prec = torch.clamp(prec_base - p * 0.90, min=0.05)
            
        c_obs = torch.cat([t_out, f, flux, rad, p_prim, pow_th, rod, pzr, fw, sf, t_in, cont], dim=-1)
        c_lat = torch.cat([sq, dt, clad, prec], dim=-1)
        
        X_obs = torch.where(mask, c_obs, X_obs)
        Y_latent = torch.where(mask, c_lat, Y_latent)
        
    if apply_noise:
        X_obs = apply_instrument_noise_suite(X_obs, noise_scale=1.0)
        
    return X_obs, Y_latent, Y_eop


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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
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
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = model(bx)
                    losses = loss_fn(
                        pred_physics=out["physics_trajectories"],
                        target_physics=bx,
                        eop_logits=out["eop_logits"],
                        target_eop=by
                    )
                    loss = losses["total_loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(bx)
                losses = loss_fn(
                    pred_physics=out["physics_trajectories"],
                    target_physics=bx,
                    eop_logits=out["eop_logits"],
                    target_eop=by
                )
                loss = losses["total_loss"]
                loss.backward()
                optimizer.step()
                
            total_train_loss += loss.item() * bx.size(0)
            total_energy_loss += losses["energy_loss"].item() * bx.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_energy = total_energy_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        out = model(bx)
                        losses = loss_fn(
                            pred_physics=out["physics_trajectories"],
                            target_physics=bx,
                            eop_logits=out["eop_logits"],
                            target_eop=by
                        )
                else:
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
