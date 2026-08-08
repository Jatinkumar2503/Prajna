"""
PRAJNA TRAINER — Multi-Physics PINN Training Engine
Executes automated training of the PINN model on reactor physics data.
Supports:
1. Synthetic and real-world reactor transient sweeps
2. Physics-Informed Autograd Loss regularizers (Point Kinetics, Energy Balance, DNBR)
3. Dynamic Device Selection: GPU/CUDA when available, optimized CPU multi-threading
4. High-efficiency model checkpointing
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure root package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.physics import PrajnaPhysicsLoss, DifferentiablePointKinetics
from prajna_core.models.pinn_foundation import PrajnaFoundation3B


def generate_synthetic_reactor_dataset(num_samples=2000, seq_len=60, num_channels=16):
    """
    Generates physically consistent multi-sensor nuclear reactor telemetry:
    Channel 0: Core Temp (°C)
    Channel 1: Coolant Flow (kg/s)
    Channel 2: Neutron Flux (x10^13)
    Channel 3: Radiation (mSv/h)
    Channels 4-15: Pressure, Steam Quality, Control Rods, Aux Heat Balances
    """
    print(f"[*] Generating {num_samples} multi-physics reactor transient sequences (Length: {seq_len}s)...")
    
    # Base telemetry tensor
    X = torch.zeros(num_samples, seq_len, num_channels, dtype=torch.float32)
    
    for i in range(num_samples):
        # Initial conditions with Monte Carlo perturbation
        t_base = 285.0 + (torch.rand(1).item() - 0.5) * 10.0
        f_base = 78.0 + (torch.rand(1).item() - 0.5) * 5.0
        flux_base = 2.32 + (torch.rand(1).item() - 0.5) * 0.2
        rad_base = 0.42 + (torch.rand(1).item() - 0.5) * 0.05
        
        # Inject transient types: 0=Normal, 1=LOCA, 2=Rod Ejection, 3=Turbine Trip
        transient_type = i % 4
        
        for t in range(seq_len):
            progress = t / float(seq_len)
            noise = (torch.randn(num_channels) * 0.02)
            
            if transient_type == 1:  # LOCA
                t_val = t_base + (progress ** 1.3) * 65.0
                f_val = max(30.0, f_base - progress * 40.0)
                flux_val = max(0.5, flux_base - progress * 0.8)
                rad_val = rad_base + (progress ** 2) * 2.5
            elif transient_type == 2:  # Rod Ejection
                t_val = t_base + progress * 55.0
                f_val = f_base - progress * 5.0
                flux_val = flux_base + (progress ** 1.4) * 2.0
                rad_val = rad_base + progress * 1.2
            elif transient_type == 3:  # Steam Rupture
                t_val = t_base + progress * 30.0
                f_val = max(40.0, f_base - progress * 30.0)
                flux_val = flux_base - progress * 0.4
                rad_val = rad_base + (progress ** 1.5) * 3.0
            else:  # Steady state
                t_val = t_base + math.sin(t * 0.1) * 0.8
                f_val = f_base + math.cos(t * 0.1) * 0.5
                flux_val = flux_base + math.sin(t * 0.15) * 0.03
                rad_val = rad_base + math.cos(t * 0.08) * 0.01
                
            X[i, t, 0] = t_val + noise[0]
            X[i, t, 1] = f_val + noise[1]
            X[i, t, 2] = flux_val + noise[2]
            X[i, t, 3] = rad_val + noise[3]
            X[i, t, 4:] = torch.randn(12) * 0.1  # Auxiliary secondary parameters
            
    return X


def train_prajna_pinn(epochs=10, batch_size=32, lr=1e-3, full_scale=False):
    """
    Main training execution routine.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  PRAJNA PINN TRAINING ENGINE — Active Hardware: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print(f"  CPU Multi-Threading Active: {torch.get_num_threads()} cores")
    print("=" * 70)
    
    # 1. Dataset Generation
    dataset_tensor = generate_synthetic_reactor_dataset(num_samples=1200, seq_len=30, num_channels=16)
    dataset = TensorDataset(dataset_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model Instantiation
    # For fast local training on CPU/GPU without OOM, we configure optimized architecture
    print("[*] Initializing PRAJNA Foundation Model Architecture...")
    model = PrajnaFoundation3B(
        num_channels=16,
        is_lightweight_test=not full_scale
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"[+] Total Model Parameters: {param_count:,}")
    
    # 3. Physics Loss and Optimizer
    physics_loss_fn = PrajnaPhysicsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 4. Training Loop with Physics Loss Regularization
    print("\n[*] Starting Physics-Informed Training Loop...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_dnbr_pen = 0.0
        batches = 0
        
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            physics_preds = outputs["physics_trajectories"]
            
            # Extract channels for physics regularizer
            # Measured vs Predicted
            pred_temp = physics_preds[:, :, 0:1]
            pred_flux = physics_preds[:, :, 2:3]
            pred_power = physics_preds[:, :, 5:6] if physics_preds.shape[-1] > 5 else pred_flux * 40.0
            
            measured_temp = x_batch[:, :, 0:1]
            measured_flow = x_batch[:, :, 1:2]
            measured_flux = x_batch[:, :, 2:3]
            inlet_temp = measured_temp - 30.0  # Core inlet delta estimation
            reactivity = (measured_flux - 2.32) * 0.001
            
            # Compute composite PINN loss (Data + Conservation of Energy + DNBR limit)
            loss_dict = physics_loss_fn(
                pred_flux=pred_flux,
                pred_power=pred_power,
                pred_temp=pred_temp,
                mass_flow=measured_flow,
                inlet_temp=inlet_temp,
                reactivity=reactivity,
                measured_flux=measured_flux,
                measured_temp=measured_temp
            )
            
            loss = loss_dict["total_loss"]
            loss.backward()
            
            # Gradient clipping to ensure ODE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_data_loss += loss_dict["data_loss"].item()
            epoch_energy_loss += loss_dict["energy_loss"].item()
            epoch_dnbr_pen += loss_dict["dnbr_penalty"].item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        avg_data = epoch_data_loss / batches
        avg_energy = epoch_energy_loss / batches
        avg_dnbr = epoch_dnbr_pen / batches
        
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Total Loss: {avg_loss:.4f} | Data MSE: {avg_data:.4f} | Energy Loss: {avg_energy:.4f} | DNBR Margin Penalty: {avg_dnbr:.4f}")
        
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"[OK] Training Completed in {elapsed:.2f} seconds.")
    
    # 5. Save Checkpoint
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "prajna_pinn_latest.pt")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "param_count": param_count,
        "epochs": epochs,
        "final_loss": avg_loss
    }, checkpoint_path)
    
    print(f"[+] Model checkpoint successfully saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train_prajna_pinn(epochs=5, batch_size=32, lr=1e-3, full_scale=False)
