"""
Independent Evaluation Pipeline on Completely Unseen Test Split (Performers 9 & 10).
Measures:
- Overall Test Accuracy
- Per-class Precision, Recall, F1-Score, Support
- Per-Class Confusion Matrix
- Inference Latency Benchmark
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
from feature_extractor import FEATURE_NAMES
from train import MudraMLP, MUDRAS, CLASS_TO_IDX

def evaluate_test_set(base_dir: str):
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")

    with open(os.path.join(dataset_dir, "normalization_stats.json")) as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    df_test = pd.read_csv(os.path.join(dataset_dir, "landmarks_test.csv"))
    X_test_raw = df_test[FEATURE_NAMES].values.astype(np.float32)
    X_test_norm = (X_test_raw - mean) / std
    y_true_names = df_test["mudra_class"].values
    y_true = np.array([CLASS_TO_IDX[c] for c in y_true_names], dtype=np.int64)

    test_performers = df_test["performer_id"].unique().tolist()
    print("================ INDEPENDENT TEST SET EVALUATION ================")
    print(f"Test Set: {len(df_test)} samples across unseen performers: {test_performers}")
    print(f"Number of classes: {len(MUDRAS)}")

    # Load MLP Model
    mlp = MudraMLP(input_dim=len(FEATURE_NAMES), num_classes=len(MUDRAS), mean=mean, std=std)
    mlp.load_state_dict(torch.load(os.path.join(models_dir, "mlp_best.pt"), weights_only=True))
    mlp.eval()

    # Latency benchmark
    t_tensor = torch.tensor(X_test_raw[:1], dtype=torch.float32)
    warmups = [mlp(t_tensor) for _ in range(20)]
    t0 = time.perf_counter()
    n_runs = 500
    for _ in range(n_runs):
        _ = mlp(t_tensor)
    t1 = time.perf_counter()
    avg_latency_ms = ((t1 - t0) / n_runs) * 1000.0

    # Predictions
    with torch.no_grad():
        test_probs = mlp(torch.tensor(X_test_raw, dtype=torch.float32))
        mlp_preds = torch.argmax(test_probs, dim=1).cpu().numpy()

    test_acc = accuracy_score(y_true, mlp_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, mlp_preds, average="macro")

    print(f"\n--- Multi-Layer Perceptron Independent Test Results ---")
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Macro Precision:       {macro_prec:.4f}")
    print(f"Macro Recall:          {macro_rec:.4f}")
    print(f"Macro F1-Score:        {macro_f1:.4f}")
    print(f"Average CPU Latency:   {avg_latency_ms:.3f} ms / frame (~{1000.0/avg_latency_ms:.0f} FPS)")

    # Per-class classification report
    print("\n--- Detailed Per-Mudra Classification Report ---")
    report = classification_report(y_true, mlp_preds, target_names=MUDRAS, digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, mlp_preds)
    print("--- Confusion Matrix (Rows: Actual, Columns: Predicted) ---")
    header = f"{'Actual \\ Pred':<15}" + "".join([f"{m[:5]:>7}" for m in MUDRAS])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{MUDRAS[i]:<15}" + "".join([f"{val:>7}" for val in row])
        print(row_str)

    # Save test report JSON
    eval_results = {
        "overall_test_accuracy": float(test_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "average_latency_ms": float(avg_latency_ms),
        "test_performers": test_performers,
        "total_test_samples": len(df_test),
        "confusion_matrix": cm.tolist(),
        "classes": MUDRAS,
    }
    with open(os.path.join(models_dir, "test_evaluation_report.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nSaved test evaluation report to {os.path.join(models_dir, 'test_evaluation_report.json')}")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    evaluate_test_set(base)

