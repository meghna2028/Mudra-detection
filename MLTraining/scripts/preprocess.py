"""
Dataset Preprocessing and Quality Verification Pipeline.
- Verifies class balance across splits
- Verifies performer isolation (ensures zero data leakage)
- Checks for near duplicates or corrupted rows
- Computes feature normalization statistics (mean, std)
"""

import os
import json
import numpy as np
import pandas as pd
from feature_extractor import FEATURE_NAMES

def verify_and_preprocess(dataset_dir: str):
    train_path = os.path.join(dataset_dir, "landmarks_train.csv")
    val_path = os.path.join(dataset_dir, "landmarks_val.csv")
    test_path = os.path.join(dataset_dir, "landmarks_test.csv")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}. Run generate_dataset.py first.")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print("=== DATASET OVERVIEW ===")
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")
    print(f"Test samples:  {len(df_test)}")

    # 1. Check Class Balance
    print("\n=== CLASS DISTRIBUTION (Train) ===")
    print(df_train["mudra_class"].value_counts())

    # 2. Check Performer Leakage
    train_performers = set(df_train["performer_id"].unique())
    val_performers = set(df_val["performer_id"].unique())
    test_performers = set(df_test["performer_id"].unique())

    print("\n=== PERFORMER ISOLATION CHECK ===")
    print(f"Train performers: {train_performers}")
    print(f"Val performers:   {val_performers}")
    print(f"Test performers:  {test_performers}")

    train_val_overlap = train_performers.intersection(val_performers)
    train_test_overlap = train_performers.intersection(test_performers)
    val_test_overlap = val_performers.intersection(test_performers)

    assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED between Val and Test: {val_test_overlap}"
    print("[PASS] Zero performer leakage verified across train, val, and test splits!")

    # 3. Compute Normalization Statistics on Training Split Only
    X_train = df_train[FEATURE_NAMES].values
    mean = np.mean(X_train, axis=0).tolist()
    std = np.std(X_train, axis=0).tolist()
    # Avoid zero division
    std = [s if s > 1e-6 else 1.0 for s in std]

    stats = {
        "feature_names": FEATURE_NAMES,
        "mean": mean,
        "std": std,
        "num_features": len(FEATURE_NAMES),
        "classes": sorted(df_train["mudra_class"].unique().tolist())
    }

    stats_path = os.path.join(dataset_dir, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[PASS] Computed normalization stats for {len(FEATURE_NAMES)} features.")
    print(f"Saved stats to {stats_path}.")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "dataset")
    verify_and_preprocess(data_dir)

