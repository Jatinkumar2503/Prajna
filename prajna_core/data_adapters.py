"""
PRAJNA DATA ADAPTERS — External Dataset Integration Pipeline
Normalizes heterogeneous nuclear reactor datasets into Prajna's unified
12-channel [Batch, SeqLen, 12] tensor format.

Supported Sources:
1. NPPAD (PCTRAN PWR simulator) — 97 columns, 18 accident scenarios, ~100 runs each
2. PUR-1 (Purdue Research Reactor) — 8 channels, REAL sensor data with scrams & FDI
3. Kaggle NPP (Synthetic intrusion detection) — 10 channels, anomaly-labeled

Channel Mapping Target (Prajna 12-channel format):
    0: Core Exit Temp (°C)          6: Control Rod Height (%)
    1: Coolant Flow (kg/s)          7: Pressurizer Level (%)
    2: Neutron Flux (×10¹³)         8: Feedwater Temp (°C)
    3: Radiation (mSv/h)            9: Steam Flow (kg/s)
    4: Primary Pressure (bar)      10: Core Inlet Temp (°C)
    5: Core Power (MWth)           11: Containment Pressure (kPa)
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union


# =============================================================================
# PRAJNA NOMINAL PLANT VALUES (for filling unmappable channels)
# =============================================================================
PRAJNA_NOMINAL = {
    0: 293.4, 1: 3500.0, 2: 2.25, 3: 0.40, 4: 85.0, 5: 756.0,
    6: 65.0,  7: 50.0,   8: 245.0, 9: 364.0, 10: 249.0, 11: 101.325,
}

# =============================================================================
# 1. NPPAD ADAPTER (PCTRAN PWR Simulator — 18 Scenarios, 97 Columns)
# =============================================================================
# Exact column names from downloaded dataset at datasets/nppad/Operation_csv_data/
# PCTRAN uses a mix of SI and imperial units depending on configuration.
# The downloaded dataset uses SI-like units (°C for temp, MPa/bar for pressure).

NPPAD_DIRECT_MAP = {
    # PCTRAN_col → (prajna_ch, conversion_func)
    # Temperatures: PCTRAN gives °C directly in this dataset version
    "THA":   0,    # Hot Leg A Temperature → Core Exit Temp (°C)
    "WRCA":  1,    # RCS Coolant Flow Loop A (kg/s)
    "PWR":   2,    # Reactor Power (% of rated) → scale to flux
    "RM1":   3,    # Radiation Monitor 1 (mR/hr)
    "P":     4,    # RCS Pressure (bar or psia — checked at load time)
    "QMWT":  5,    # Core Thermal Power (MWth)
    "RRCO":  6,    # Control Rod Position (fraction 0-1)
    "LVPZ":  7,    # Pressurizer Level (%)
    "TFSB":  8,    # Feedwater / SG Temperature (°C)
    "WSTA":  9,    # Steam Flow Loop A (kg/s)
    "TCA":  10,    # Cold Leg A Temperature → Core Inlet (°C)
    "PRB":  11,    # Containment Building Pressure (kPa)
}

# NPPAD directory names → (prajna_5class_label, full_description)
NPPAD_DIR_TO_LABEL = {
    "Normal": (0, "Normal Operation"),
    "LOCA":   (1, "Large Break LOCA"),
    "LOCAC":  (1, "Cold Leg LOCA"),
    "LLB":    (1, "Large Leak Break"),
    "SGATR":  (3, "SG-A Tube Rupture"),
    "SGBTR":  (3, "SG-B Tube Rupture"),
    "SLBIC":  (3, "Steam Line Break Inside Containment"),
    "SLBOC":  (3, "Steam Line Break Outside Containment"),
    "FLB":    (3, "Feedwater Line Break"),
    "LACP":   (4, "Loss of All AC Power / Station Blackout"),
    "LOF":    (4, "Loss of Flow"),
    "LR":     (4, "Loss of Load / Reactor Trip"),
    "RI":     (2, "Reactivity Insertion (Rod Ejection)"),
    "RW":     (2, "Rod Withdrawal at Power"),
    "ATWS":   (2, "ATWS (Anticipated Transient Without SCRAM)"),
    "SP":     (0, "Spurious Trip"),
    "TT":     (0, "Turbine Trip"),
    "MD":     (4, "Mechanical Damage / Pump Trip"),
}


def load_nppad_csv(csv_path: str) -> np.ndarray:
    """Load a single NPPAD CSV and map 97 PCTRAN columns → 12 Prajna channels."""
    df = pd.read_csv(csv_path)
    n_rows = len(df)
    data = np.full((n_rows, 12), np.nan, dtype=np.float32)

    for pctran_col, prajna_ch in NPPAD_DIRECT_MAP.items():
        if pctran_col not in df.columns:
            continue
        vals = pd.to_numeric(df[pctran_col], errors='coerce').fillna(0).values.astype(np.float32)

        if prajna_ch == 0:   # THA: Hot leg temp
            # Check if data looks like Fahrenheit (values > 400 suggest °F)
            if vals.max() > 400:
                vals = (vals - 32.0) * 5.0 / 9.0
            data[:, 0] = vals

        elif prajna_ch == 1:  # WRCA: Coolant flow
            # If values > 100000, likely lbm/hr; convert to kg/s
            if vals.max() > 100000:
                vals = vals * 0.000126
            data[:, 1] = vals

        elif prajna_ch == 2:  # PWR: Reactor power %
            # Convert % power to flux (×10¹³ n/cm²·s): 100% → 2.25
            data[:, 2] = vals * 0.0225

        elif prajna_ch == 3:  # RM1: Radiation
            # mR/hr → mSv/h (1 mR/hr ≈ 0.01 mSv/h)
            data[:, 3] = vals * 0.01

        elif prajna_ch == 4:  # P: RCS Pressure
            # If values > 500, likely psia; convert to bar
            if vals.max() > 500:
                vals = vals * 0.06895
            data[:, 4] = vals

        elif prajna_ch == 5:  # QMWT: Core thermal power (MWth direct)
            data[:, 5] = vals

        elif prajna_ch == 6:  # RRCO: Control rod position
            # If max ≤ 1.0, it's a fraction; convert to %
            if vals.max() <= 1.5:
                vals = vals * 100.0
            data[:, 6] = vals

        elif prajna_ch == 7:  # LVPZ: Pressurizer level (% direct)
            data[:, 7] = vals

        elif prajna_ch == 8:  # TFSB: Feedwater temp
            if vals.max() > 400:
                vals = (vals - 32.0) * 5.0 / 9.0
            data[:, 8] = vals

        elif prajna_ch == 9:  # WSTA: Steam flow
            if vals.max() > 100000:
                vals = vals * 0.000126
            data[:, 9] = vals

        elif prajna_ch == 10:  # TCA: Cold leg temp
            if vals.max() > 400:
                vals = (vals - 32.0) * 5.0 / 9.0
            data[:, 10] = vals

        elif prajna_ch == 11:  # PRB: Containment pressure
            # If values < 5, likely psig; convert to kPa
            if vals.max() < 50:
                vals = vals * 6.895 + 101.325
            data[:, 11] = vals

    # Fill NaN channels with nominal values
    for ch in range(12):
        mask = np.isnan(data[:, ch])
        if mask.any():
            data[mask, ch] = PRAJNA_NOMINAL[ch]

    return data


def load_all_nppad(nppad_root: str,
                   data_subdir: str = "Operation_csv_data",
                   window_len: int = 45,
                   stride: int = 15,
                   max_files_per_scenario: int = 20) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Loads all NPPAD scenarios, maps to Prajna 12ch, and windows the data.
    
    Returns:
        windows: [N_total, window_len, 12]
        labels:  [N_total]
        info:    {scenario_name: n_windows}
    """
    data_root = os.path.join(nppad_root, data_subdir)
    if not os.path.exists(data_root):
        data_root = nppad_root  # Fallback

    all_windows = []
    all_labels = []
    info = {}

    for dir_name, (label, desc) in NPPAD_DIR_TO_LABEL.items():
        scenario_dir = os.path.join(data_root, dir_name)
        if not os.path.isdir(scenario_dir):
            continue

        csv_files = sorted(glob.glob(os.path.join(scenario_dir, "*.csv")))[:max_files_per_scenario]
        scenario_windows = 0

        for csv_path in csv_files:
            try:
                data_12ch = load_nppad_csv(csv_path)
                n_rows = data_12ch.shape[0]
                for start in range(0, n_rows - window_len + 1, stride):
                    window = data_12ch[start:start + window_len]
                    all_windows.append(window)
                    all_labels.append(label)
                    scenario_windows += 1
            except Exception as e:
                print(f"  [WARN] Skipping {csv_path}: {e}")

        info[f"{dir_name} ({desc})"] = scenario_windows
        if scenario_windows > 0:
            print(f"  ✓ {dir_name:8s} → class {label} | {scenario_windows:5d} windows from {len(csv_files)} files")

    if not all_windows:
        raise ValueError(f"No NPPAD data found in {data_root}")

    return (
        torch.tensor(np.array(all_windows), dtype=torch.float32),
        torch.tensor(all_labels, dtype=torch.long),
        info
    )


# =============================================================================
# 2. PUR-1 ADAPTER (Purdue Research Reactor — REAL Data)
# =============================================================================

def load_pur1_dataset(pur1_root: str,
                      dataset_name: str = "real_dataset_normalized",
                      window_len: int = 45,
                      stride: int = 15) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Loads PUR-1 real reactor data → Prajna 12ch format.
    
    PUR-1 columns: nfd-1-cps, nfd-1-cr, rr-active-state, rr-position,
                   ss1-active-state, ss1-position, ss2-active-state, ss2-position
    """
    csv_path = os.path.join(pur1_root, f"{dataset_name}.csv")
    df = pd.read_csv(csv_path).drop(columns=['index'], errors='ignore')
    df = df.ffill().fillna(0)
    raw = df.values.astype(np.float32)
    n_rows = len(raw)

    data_12ch = np.zeros((n_rows, 12), dtype=np.float32)

    # Fill nominal baseline
    for ch, val in PRAJNA_NOMINAL.items():
        data_12ch[:, ch] = val

    # Map neutron flux: nfd-1-cps (normalized 0-1.15) → Prajna ch2 (×10¹³)
    if "nfd-1-cps" in df.columns:
        flux = df["nfd-1-cps"].values.astype(np.float32)
        flux_max = max(flux.max(), 1e-6)
        data_12ch[:, 2] = flux / flux_max * 2.25  # Scale to ×10¹³

    # Core power proportional to flux
    data_12ch[:, 5] = data_12ch[:, 2] / 2.25 * 756.0

    # Temperature correlated with power
    power_frac = np.clip(data_12ch[:, 5] / 756.0, 0, 2)
    data_12ch[:, 0] = 249.0 + 44.4 * power_frac  # Core exit
    data_12ch[:, 10] = 249.0                       # Core inlet (constant)

    # Control rod: average of available rod positions
    rod_cols = [c for c in ["rr-position", "ss1-position", "ss2-position"] if c in df.columns]
    if rod_cols:
        rod_avg = df[rod_cols].values.astype(np.float32).mean(axis=1)
        data_12ch[:, 6] = rod_avg * 100.0

    # Extract REAL noise profile (the critical value of PUR-1)
    noise_profile = _extract_noise_stats(raw, list(df.columns))

    # Window
    windows = []
    for start in range(0, n_rows - window_len + 1, stride):
        windows.append(data_12ch[start:start + window_len])

    # Label: 0 = normal, 4 = scram-like
    if "scram" in dataset_name.lower():
        label_val = 4
    elif "fdi" in dataset_name.lower():
        label_val = 0  # Cyber attack → labeled normal (adversarial)
    else:
        label_val = 0

    labels = torch.full((len(windows),), label_val, dtype=torch.long)
    
    return (
        torch.tensor(np.array(windows), dtype=torch.float32),
        labels,
        {"source": dataset_name, "n_rows": n_rows, "n_windows": len(windows),
         "is_real_data": True, "noise_profile": noise_profile}
    )


def load_all_pur1(pur1_root: str, window_len: int = 45, stride: int = 15
                  ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Loads all PUR-1 sub-datasets and combines them."""
    all_w, all_l = [], []
    info = {}
    for ds in ["real_dataset_normalized", "scrams_normal", "transient_normal",
               "FDI1_normal", "FDI2_normal"]:
        csv_path = os.path.join(pur1_root, f"{ds}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            w, l, meta = load_pur1_dataset(pur1_root, ds, window_len, stride)
            all_w.append(w)
            all_l.append(l)
            info[ds] = meta["n_windows"]
            print(f"  ✓ PUR-1/{ds}: {meta['n_windows']} windows from {meta['n_rows']} rows (REAL DATA)")
        except Exception as e:
            print(f"  [WARN] PUR-1/{ds}: {e}")

    return torch.cat(all_w), torch.cat(all_l), info


# =============================================================================
# 3. KAGGLE NPP ADAPTER (Synthetic Intrusion Detection)
# =============================================================================

def load_kaggle_npp(kaggle_root: str,
                    window_len: int = 45,
                    stride: int = 15) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Loads Kaggle NPP intrusion detection data → Prajna 12ch."""
    csv_files = glob.glob(os.path.join(kaggle_root, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {kaggle_root}")

    df = pd.read_csv(csv_files[0])
    n_rows = len(df)
    data_12ch = np.zeros((n_rows, 12), dtype=np.float32)

    # Map Kaggle columns → Prajna channels (fuzzy matching)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "reactor" in cl and "temp" in cl:
            col_map[col] = 0
        elif "coolant" in cl and "flow" in cl:
            col_map[col] = 1
        elif "neutron" in cl or "flux" in cl:
            col_map[col] = 2
        elif "radiation" in cl:
            col_map[col] = 3
        elif "pressure" in cl and "coolant" not in cl and "containment" not in cl:
            col_map[col] = 4
        elif "power" in cl or "output" in cl:
            col_map[col] = 5
        elif "steam" in cl and "temp" in cl:
            col_map[col] = 8
        elif "coolant" in cl and "press" in cl:
            # Secondary pressure mapping
            col_map[col] = 11

    # Fill nominal baseline
    for ch, val in PRAJNA_NOMINAL.items():
        data_12ch[:, ch] = val

    # Override with mapped columns
    for col_name, ch_idx in col_map.items():
        vals = pd.to_numeric(df[col_name], errors='coerce').fillna(0).values.astype(np.float32)
        # Scale to Prajna ranges
        if ch_idx == 0:  # Temp → should be ~293°C
            data_12ch[:, 0] = vals
        elif ch_idx == 1:  # Flow → scale from m³/s to kg/s (×900 for D2O density)
            if vals.max() < 100:
                vals = vals * 230.0  # Approximate scaling
            data_12ch[:, 1] = vals
        elif ch_idx == 2:  # Neutron flux
            if vals.max() > 1e10:
                vals = vals / 1e13  # Normalize to ×10¹³
            data_12ch[:, 2] = vals
        elif ch_idx == 3:  # Radiation (µSv/h → mSv/h)
            if vals.max() < 10:
                pass  # Already in mSv/h range
            else:
                vals = vals * 0.001
            data_12ch[:, 3] = vals
        elif ch_idx == 4:  # Pressure (bar)
            data_12ch[:, 4] = vals
        elif ch_idx == 5:  # Power (MWth)
            data_12ch[:, 5] = vals
        elif ch_idx == 8:
            data_12ch[:, 8] = vals
        elif ch_idx == 11:
            if vals.max() > 50:
                vals = vals * 6.895  # MPa → kPa
            data_12ch[:, 11] = vals

    # Derive missing channels from available ones
    if data_12ch[:, 5].max() > 0:
        pf = np.clip(data_12ch[:, 5] / 756.0, 0, 2)
        data_12ch[:, 10] = 249.0  # Inlet
        if col_map.get("Reactor_Temperature_C") is None:
            data_12ch[:, 0] = 249.0 + 44.4 * pf

    # Extract labels
    label_col = None
    for c in df.columns:
        if c.lower() in ("label", "class", "anomaly", "intrusion", "target"):
            label_col = c
            break

    if label_col:
        raw_labels = df[label_col].values
        # Map: 0=normal, 1=anomaly→treat as generic "unknown event" (class 0 for now)
        labels_arr = np.where(raw_labels > 0, 0, 0).astype(np.int64)
    else:
        labels_arr = np.zeros(n_rows, dtype=np.int64)

    # Window
    windows = []
    win_labels = []
    for start in range(0, n_rows - window_len + 1, stride):
        windows.append(data_12ch[start:start + window_len])
        # Use majority label in window
        win_labels.append(int(np.median(labels_arr[start:start + window_len])))

    info = {"source": "kaggle_npp", "n_rows": n_rows, "n_windows": len(windows),
            "mapped_columns": col_map}
    print(f"  ✓ Kaggle NPP: {len(windows)} windows from {n_rows} rows")

    return (
        torch.tensor(np.array(windows), dtype=torch.float32),
        torch.tensor(win_labels, dtype=torch.long),
        info
    )


# =============================================================================
# 4. NOISE PROFILE EXTRACTION (from real PUR-1 data)
# =============================================================================

def _extract_noise_stats(raw_data: np.ndarray, col_names: List[str]) -> Dict:
    """Extract real noise characteristics from PUR-1 sensor data."""
    stats = {}
    for i, col in enumerate(col_names):
        if i >= raw_data.shape[1]:
            break
        signal = raw_data[:, i]
        if len(signal) < 100:
            continue
        # Detrend with moving average
        kernel = np.ones(50) / 50
        smoothed = np.convolve(signal, kernel, mode='same')
        noise = signal - smoothed
        n = len(noise)
        mean_n = np.mean(noise)
        std_n = np.std(noise)
        if std_n < 1e-10:
            continue
        centered = (noise - mean_n) / std_n
        stats[col] = {
            "std": float(std_n),
            "kurtosis": float(np.mean(centered ** 4) - 3.0),
            "skewness": float(np.mean(centered ** 3)),
            "autocorr_lag1": float(np.mean((noise[:-1] - mean_n) * (noise[1:] - mean_n)) / (std_n**2)),
        }
    return stats


# =============================================================================
# 5. UNIFIED DATALOADER FACTORY
# =============================================================================

def create_combined_dataloader(
    nppad_root: Optional[str] = None,
    pur1_root: Optional[str] = None,
    kaggle_root: Optional[str] = None,
    prajna_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    window_len: int = 45,
    batch_size: int = 32,
    shuffle: bool = True,
    max_nppad_files: int = 20,
) -> Tuple[DataLoader, Dict]:
    """
    Creates a unified DataLoader combining all available data sources.
    Returns (DataLoader, metadata).
    """
    all_w, all_l = [], []
    meta = {"sources": [], "counts": {}}

    # 1. Prajna internal simulator
    if prajna_data is not None:
        w, l = prajna_data
        all_w.append(w)
        all_l.append(l)
        meta["sources"].append("prajna_sim")
        meta["counts"]["prajna_sim"] = w.shape[0]

    # 2. NPPAD
    if nppad_root and os.path.exists(nppad_root):
        print("\n[NPPAD] Loading PCTRAN PWR scenarios...")
        w, l, info = load_all_nppad(nppad_root, window_len=window_len,
                                     max_files_per_scenario=max_nppad_files)
        all_w.append(w)
        all_l.append(l)
        meta["sources"].append("nppad")
        meta["counts"]["nppad"] = w.shape[0]
        meta["nppad_info"] = info

    # 3. PUR-1
    if pur1_root and os.path.exists(pur1_root):
        print("\n[PUR-1] Loading REAL reactor data...")
        w, l, info = load_all_pur1(pur1_root, window_len=window_len)
        all_w.append(w)
        all_l.append(l)
        meta["sources"].append("pur1_real")
        meta["counts"]["pur1_real"] = w.shape[0]
        meta["pur1_info"] = info

    # 4. Kaggle NPP
    if kaggle_root and os.path.exists(kaggle_root):
        print("\n[Kaggle] Loading NPP intrusion detection data...")
        w, l, info = load_kaggle_npp(kaggle_root, window_len=window_len)
        all_w.append(w)
        all_l.append(l)
        meta["sources"].append("kaggle_npp")
        meta["counts"]["kaggle_npp"] = w.shape[0]

    if not all_w:
        raise ValueError("No data sources found!")

    combined_w = torch.cat(all_w, dim=0)
    combined_l = torch.cat(all_l, dim=0)
    meta["total"] = combined_w.shape[0]

    dataset = TensorDataset(combined_w, combined_l)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=0, pin_memory=True, drop_last=True)
    return loader, meta


# =============================================================================
# 6. CLI — Integration Test
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("PRAJNA DATA ADAPTER — External Dataset Integration Report")
    print("=" * 80)

    nppad = "datasets/nppad"
    pur1 = "datasets/pur1"
    kaggle = "datasets/kaggle_npp"

    loader, meta = create_combined_dataloader(
        nppad_root=nppad if os.path.exists(nppad) else None,
        pur1_root=pur1 if os.path.exists(pur1) else None,
        kaggle_root=kaggle if os.path.exists(kaggle) else None,
        window_len=45,
        batch_size=32,
    )

    print(f"\n{'='*80}")
    print(f"COMBINED DATALOADER SUMMARY")
    print(f"{'='*80}")
    print(f"  Total samples: {meta['total']:,}")
    print(f"  Sources: {meta['sources']}")
    for src, count in meta['counts'].items():
        pct = count / meta['total'] * 100
        print(f"    {src:20s}: {count:6,} windows ({pct:.1f}%)")

    batch_x, batch_y = next(iter(loader))
    print(f"\n  Sample batch shape: x={batch_x.shape}, y={batch_y.shape}")
    print(f"  Value range: [{batch_x.min():.2f}, {batch_x.max():.2f}]")
    print(f"  Label classes: {torch.unique(batch_y).tolist()}")
    print("=" * 80)
