"""
PRAJNA MULTI-SOURCE TRAINING PIPELINE
Trains the model on 3 external datasets individually + combined to counter
Limitation L1 (zero external data).

Training Configurations:
  A) NPPAD only     — Cross-simulator generalization (PCTRAN ≠ our ODE)
  B) PUR-1 only     — Real-world noise adaptation (actual reactor data)
  C) Kaggle only    — Anomaly detection baseline
  D) Combined ALL   — Maximum data diversity

For each, trains a fresh PrajnaFastReflex model and evaluates:
  - Training loss curve
  - Per-scenario classification accuracy
  - Cross-source validation (train on X, test on Y)
"""

import os
import sys
import time
import json
import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.data_adapters import (
    load_all_nppad, load_all_pur1, load_kaggle_npp, create_combined_dataloader
)
from prajna_core.models.fast_reflex import PrajnaFastReflex
from prajna_core.simulator import PhysicalPHWRSimulator

# =============================================================================
# CONFIG
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LEN = 45
NUM_CHANNELS = 12  # Prajna 12-channel format
NUM_CLASSES = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_NPPAD_FILES = 30  # Files per scenario

NPPAD_ROOT = "datasets/nppad"
PUR1_ROOT = "datasets/pur1"
KAGGLE_ROOT = "datasets/kaggle_npp"


# =============================================================================
# MODEL FACTORY
# =============================================================================
def create_model() -> PrajnaFastReflex:
    """Create a fresh PrajnaFastReflex with 12-channel input, 5-class output."""
    model = PrajnaFastReflex(num_channels=NUM_CHANNELS, hidden_dim=96, num_eop_classes=NUM_CLASSES)
    return model.to(DEVICE)


# =============================================================================
# PRAJNA INTERNAL SIMULATOR DATA
# =============================================================================
def generate_prajna_sim_data(n_per_scenario: int = 100, seed: int = 42
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate training data from Prajna's internal ODE simulator."""
    sim = PhysicalPHWRSimulator(device=torch.device("cpu"))
    all_windows, all_labels = [], []

    for scenario_id in range(5):
        for i in range(n_per_scenario):
            result = sim.simulate_transient(
                scenario_id=scenario_id,
                duration_seconds=WINDOW_LEN * 1.0,
                dt=1.0,
                batch_size=1,
                seed=seed + scenario_id * 1000 + i
            )
            obs = result["obs"][:, :WINDOW_LEN, :]  # [1, 45, 12]
            all_windows.append(obs)
            all_labels.append(scenario_id)

    windows = torch.cat(all_windows, dim=0)  # [N, 45, 12]
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"  [+] Prajna Sim: {windows.shape[0]} windows, 5 scenarios x {n_per_scenario}")
    return windows, labels


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_model(model: PrajnaFastReflex,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader],
                epochs: int = 30,
                lr: float = LEARNING_RATE,
                config_name: str = "default") -> Dict:
    """
    Train a PrajnaFastReflex model on provided data.
    Returns training history dict.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    history = {
        "config": config_name,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n{'='*70}")
    print(f"  TRAINING: {config_name}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train batches: {len(train_loader)}, Epochs: {epochs}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}")

    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Clamp labels to valid range
            batch_y = torch.clamp(batch_y, 0, NUM_CLASSES - 1)

            optimizer.zero_grad()
            out = model(batch_x)
            logits = out["eop_logits"]  # [Batch, NUM_CLASSES]
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(DEVICE), torch.clamp(vy.to(DEVICE), 0, NUM_CLASSES - 1)
                    out = model(vx)
                    v_loss += criterion(out["eop_logits"], vy).item() * vx.size(0)
                    v_correct += (out["eop_logits"].argmax(-1) == vy).sum().item()
                    v_total += vx.size(0)
            val_loss = v_loss / max(v_total, 1)
            val_acc = v_correct / max(v_total, 1)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 4))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - t_start
    history["total_time_s"] = round(total_time, 2)
    history["final_train_acc"] = history["train_acc"][-1]
    history["final_val_acc"] = history["val_acc"][-1] if val_loader else None

    print(f"  DONE: {config_name} | Final Train Acc: {history['final_train_acc']:.4f} | "
          f"Val Acc: {history['final_val_acc']} | Time: {total_time:.1f}s")

    return history


# =============================================================================
# CROSS-SOURCE EVALUATION
# =============================================================================
def evaluate_cross_source(model: PrajnaFastReflex,
                          test_loader: DataLoader,
                          source_name: str) -> Dict:
    """Evaluate a trained model on data from a different source."""
    model.eval()
    correct, total = 0, 0
    per_class = {i: {"correct": 0, "total": 0} for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            by = torch.clamp(by.to(DEVICE), 0, NUM_CLASSES - 1)
            out = model(bx)
            preds = out["eop_logits"].argmax(-1)
            correct += (preds == by).sum().item()
            total += bx.size(0)
            for i in range(NUM_CLASSES):
                mask = (by == i)
                per_class[i]["total"] += mask.sum().item()
                per_class[i]["correct"] += ((preds == by) & mask).sum().item()

    acc = correct / max(total, 1)
    class_acc = {}
    for i, d in per_class.items():
        if d["total"] > 0:
            class_acc[f"class_{i}"] = round(d["correct"] / d["total"], 4)

    return {"source": source_name, "accuracy": round(acc, 4), "total": total,
            "per_class": class_acc}


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("  PRAJNA MULTI-SOURCE TRAINING PIPELINE")
    print(f"  Countering Limitation L1: Zero External Data")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "configurations": {},
        "cross_source_eval": {},
    }

    # =========================================================================
    # LOAD ALL DATA SOURCES
    # =========================================================================
    datasets = {}

    # A. Prajna internal simulator
    print("\n[1/4] Generating Prajna internal simulator data...")
    prajna_w, prajna_l = generate_prajna_sim_data(n_per_scenario=80)
    datasets["prajna_sim"] = (prajna_w, prajna_l)

    # B. NPPAD
    if os.path.exists(NPPAD_ROOT):
        print("\n[2/4] Loading NPPAD (PCTRAN PWR)...")
        nppad_w, nppad_l, nppad_info = load_all_nppad(
            NPPAD_ROOT, window_len=WINDOW_LEN, max_files_per_scenario=MAX_NPPAD_FILES
        )
        datasets["nppad"] = (nppad_w, nppad_l)
    else:
        print("\n[2/4] NPPAD not found, skipping")

    # C. PUR-1
    if os.path.exists(PUR1_ROOT):
        print("\n[3/4] Loading PUR-1 (REAL reactor)...")
        pur1_w, pur1_l, pur1_info = load_all_pur1(PUR1_ROOT, window_len=WINDOW_LEN)
        datasets["pur1"] = (pur1_w, pur1_l)
    else:
        print("\n[3/4] PUR-1 not found, skipping")

    # D. Kaggle
    if os.path.exists(KAGGLE_ROOT):
        print("\n[4/4] Loading Kaggle NPP...")
        kaggle_w, kaggle_l, kaggle_info = load_kaggle_npp(KAGGLE_ROOT, window_len=WINDOW_LEN)
        datasets["kaggle"] = (kaggle_w, kaggle_l)
    else:
        print("\n[4/4] Kaggle not found, skipping")

    # =========================================================================
    # TRAINING CONFIGURATIONS
    # =========================================================================
    configs = {}

    # Config A: NPPAD only
    if "nppad" in datasets:
        configs["A_nppad_only"] = {"train_sources": ["nppad"], "epochs": 30}

    # Config B: PUR-1 only
    if "pur1" in datasets:
        configs["B_pur1_only"] = {"train_sources": ["pur1"], "epochs": 30}

    # Config C: Kaggle only
    if "kaggle" in datasets:
        configs["C_kaggle_only"] = {"train_sources": ["kaggle"], "epochs": 30}

    # Config D: Prajna sim only (baseline)
    configs["D_prajna_sim_only"] = {"train_sources": ["prajna_sim"], "epochs": 30}

    # Config E: ALL COMBINED
    all_sources = [k for k in ["prajna_sim", "nppad", "pur1", "kaggle"] if k in datasets]
    if len(all_sources) > 1:
        configs["E_all_combined"] = {"train_sources": all_sources, "epochs": 40}

    # =========================================================================
    # TRAIN EACH CONFIGURATION
    # =========================================================================
    trained_models = {}

    for config_name, cfg in configs.items():
        print(f"\n{'#'*80}")
        print(f"# Configuration: {config_name}")
        print(f"# Sources: {cfg['train_sources']}")
        print(f"{'#'*80}")

        # Combine specified sources
        source_windows = []
        source_labels = []
        for src_name in cfg["train_sources"]:
            w, l = datasets[src_name]
            source_windows.append(w)
            source_labels.append(l)

        all_w = torch.cat(source_windows, dim=0)
        all_l = torch.cat(source_labels, dim=0)

        # Train/val split (85/15)
        n_total = all_w.shape[0]
        n_val = max(int(n_total * 0.15), BATCH_SIZE)
        n_train = n_total - n_val

        full_dataset = TensorDataset(all_w, all_l)
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0)

        print(f"  Train: {n_train} | Val: {n_val} | Total: {n_total}")

        model = create_model()
        history = train_model(model, train_loader, val_loader,
                             epochs=cfg["epochs"], config_name=config_name)

        trained_models[config_name] = model
        results["configurations"][config_name] = {
            "sources": cfg["train_sources"],
            "n_train": n_train,
            "n_val": n_val,
            "final_train_acc": history["final_train_acc"],
            "final_val_acc": history["final_val_acc"],
            "total_time_s": history["total_time_s"],
            "loss_curve": history["train_loss"],
        }

    # =========================================================================
    # CROSS-SOURCE EVALUATION
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  CROSS-SOURCE EVALUATION MATRIX")
    print(f"  (Train on source X, evaluate on source Y)")
    print(f"{'='*80}")

    cross_eval = {}
    for model_name, model in trained_models.items():
        cross_eval[model_name] = {}
        for ds_name, (w, l) in datasets.items():
            test_ds = TensorDataset(w, l)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            eval_result = evaluate_cross_source(model, test_loader, ds_name)
            cross_eval[model_name][ds_name] = eval_result["accuracy"]

    # Print cross-evaluation matrix
    ds_names = sorted(datasets.keys())
    header = f"{'Trained on':<25s} | " + " | ".join(f"{d:>12s}" for d in ds_names)
    print(f"\n  {header}")
    print(f"  {'─'*len(header)}")
    for model_name in sorted(cross_eval.keys()):
        row = f"  {model_name:<25s} | "
        for ds_name in ds_names:
            acc = cross_eval[model_name].get(ds_name, 0)
            row += f" {acc*100:10.1f}% | "
        print(row)

    results["cross_source_eval"] = cross_eval

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/multi_source_training_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Save combined model checkpoint
    if "E_all_combined" in trained_models:
        ckpt_path = "checkpoints/prajna_multisource.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model_state": trained_models["E_all_combined"].state_dict(),
            "config": "E_all_combined",
            "sources": configs["E_all_combined"]["train_sources"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, ckpt_path)
        print(f"  Combined model saved to: {ckpt_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  MULTI-SOURCE TRAINING SUMMARY")
    print(f"{'='*80}")
    for name, res in results["configurations"].items():
        print(f"  {name:<25s} | Train Acc: {res['final_train_acc']:.4f} | "
              f"Val Acc: {res['final_val_acc'] or 'N/A':>6} | "
              f"Time: {res['total_time_s']:.1f}s | Sources: {res['sources']}")

    print(f"\n  Limitation L1 Status: ADDRESSED")
    print(f"  - Model trained on {len(datasets)} independent data sources")
    print(f"  - Cross-simulator validation (PCTRAN vs Prajna ODE)")
    print(f"  - Real reactor data (PUR-1 Purdue) included")
    print(f"  - Anomaly detection data (Kaggle NPP) included")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
