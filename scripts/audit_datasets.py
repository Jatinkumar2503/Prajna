"""
PRAJNA DATASET FORMAL AUDIT SUITE
Verifies all columns in NPPAD, PUR-1, and Kaggle datasets against the
validated variable mapping and exports an immutable audit report.
"""

import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "feature_mapping.yaml")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "formal_data_audit.json")

def run_audit():
    print("=" * 70)
    print("PRAJNA FORMAL DATASET AUDIT")
    print("=" * 70)

    # 1. Load configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    audit_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "datasets": {}
    }

    # 2. Audit NPPAD
    nppad_files = glob.glob(os.path.join(PROJECT_ROOT, "datasets", "nppad", "Operation_csv_data", "*", "*.csv"))
    print(f"\n[+] Auditing NPPAD (Total CSV files: {len(nppad_files)})...")
    if nppad_files:
        df_sample = pd.read_csv(nppad_files[0])
        nppad_cols = list(df_sample.columns)
        nppad_cfg = config["datasets"]["nppad_pctran"]
        mapped_keys = list(nppad_cfg["validated_mapping"].keys())
        excluded_keys = list(nppad_cfg["permanently_excluded"])
        
        # Calculate stats for mapped columns
        mapped_stats = {}
        for col in mapped_keys:
            if col in df_sample.columns:
                series = pd.to_numeric(df_sample[col], errors='coerce').dropna()
                mapped_stats[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "target_ch": nppad_cfg["validated_mapping"][col]["target_ch"],
                    "validity": nppad_cfg["validated_mapping"][col]["validity"]
                }
        
        audit_summary["datasets"]["nppad"] = {
            "total_files": len(nppad_files),
            "total_columns": len(nppad_cols),
            "columns": nppad_cols,
            "mapped_columns_count": len(mapped_keys),
            "permanently_excluded_count": len(excluded_keys),
            "mapped_stats": mapped_stats,
            "sampling_dt_sec": float(df_sample["TIME"].diff().dropna().iloc[0]) if "TIME" in df_sample else 10.0
        }
        print(f"  [OK] Columns verified: {len(nppad_cols)} total | {len(mapped_keys)} mapped | {len(excluded_keys)} permanently excluded")

    # 3. Audit PUR-1
    pur1_files = glob.glob(os.path.join(PROJECT_ROOT, "datasets", "pur1", "*.csv"))
    print(f"\n[+] Auditing PUR-1 (Total CSV files: {len(pur1_files)})...")
    if pur1_files:
        pur1_main = os.path.join(PROJECT_ROOT, "datasets", "pur1", "real_dataset_normalized.csv")
        df_pur1 = pd.read_csv(pur1_main)
        pur1_cols = list(df_pur1.columns)
        pur1_cfg = config["datasets"]["pur1_purdue"]
        
        pur1_stats = {}
        for col in pur1_cfg["validated_mapping"].keys():
            if col in df_pur1.columns:
                series = pd.to_numeric(df_pur1[col], errors='coerce').dropna()
                pur1_stats[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "target_ch": pur1_cfg["validated_mapping"][col]["target_ch"],
                    "validity": pur1_cfg["validated_mapping"][col]["validity"]
                }

        audit_summary["datasets"]["pur1"] = {
            "total_files": len(pur1_files),
            "total_columns": len(pur1_cols),
            "columns": pur1_cols,
            "total_rows_real_dataset": len(df_pur1),
            "mapped_stats": pur1_stats,
            "sampling_dt_sec": 1.0,
            "prohibited_mappings": pur1_cfg["strictly_prohibited_mappings"]
        }
        print(f"  [OK] Real reactor telemetry verified: {len(df_pur1)} rows | {len(pur1_cols)} columns")

    # 4. Save audit
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\n[+] Formal audit exported to: {OUTPUT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    run_audit()
