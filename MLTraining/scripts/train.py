"""
ML Model Training and Benchmarking Pipeline for Bharatanatyam Mudra Classification.
Experiments with:
1. Random Forest (scikit-learn)
2. Support Vector Machine (scikit-learn)
3. Gradient Boosting Classifier (scikit-learn)
4. Multi-Layer Perceptron (PyTorch Deep Neural Network)

Evaluates on the independent validation split and selects the best model.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from feature_extractor import FEATURE_NAMES

MUDRAS = [
    "Pataka",
    "Tripataka",
    "Ardhapataka",
    "Kartarimukha",
    "Mayura",
    "Ardhachandra",
    "Mushti",
    "Shikhara",
    "Suchi",
    "Kapittha"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(MUDRAS)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(MUDRAS)}

class MudraMLP(nn.Module):
    def __init__(self, input_dim: int = 77, num_classes: int = 10, mean=None, std=None):
        super().__init__()
        if mean is not None and std is not None:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
            self.has_norm = True
        else:
            self.has_norm = False

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if self.has_norm:
            x = (x - self.mean) / self.std
        return self.network(x)


def load_dataset(csv_path: str, mean: np.ndarray, std: np.ndarray):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_names = df["mudra_class"].values
    y = np.array([CLASS_TO_IDX[c] for c in y_names], dtype=np.int64)

    X_norm = (X - mean) / std
    return X, X_norm, y, y_names


def train_models(base_dir: str):
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "normalization_stats.json")) as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    X_train_raw, X_train, y_train, _ = load_dataset(os.path.join(dataset_dir, "landmarks_train.csv"), mean, std)
    X_val_raw, X_val, y_val, val_names = load_dataset(os.path.join(dataset_dir, "landmarks_val.csv"), mean, std)

    print(f"Loaded Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples. Features: {X_train.shape[1]}")

    results = {}

    # 1. Random Forest
    print("\n--- [1/4] Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_val, rf_preds, average="macro")
    print(f"Random Forest -> Val Acc: {rf_acc * 100:.2f}%, Macro F1: {rf_f1:.4f}")
    results["Random Forest"] = {"accuracy": rf_acc, "f1": rf_f1, "model": rf}
    joblib.dump(rf, os.path.join(models_dir, "random_forest.joblib"))

    # 2. Support Vector Machine
    print("\n--- [2/4] Training Support Vector Machine (RBF kernel) ---")
    svm = SVC(C=5.0, kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_val)
    svm_acc = accuracy_score(y_val, svm_preds)
    svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y_val, svm_preds, average="macro")
    print(f"Support Vector Machine -> Val Acc: {svm_acc * 100:.2f}%, Macro F1: {svm_f1:.4f}")
    results["Support Vector Machine"] = {"accuracy": svm_acc, "f1": svm_f1, "model": svm}
    joblib.dump(svm, os.path.join(models_dir, "svm.joblib"))

    # 3. Gradient Boosting Classifier
    print("\n--- [3/4] Training Gradient Boosting ---")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_val)
    gb_acc = accuracy_score(y_val, gb_preds)
    gb_prec, gb_rec, gb_f1, _ = precision_recall_fscore_support(y_val, gb_preds, average="macro")
    print(f"Gradient Boosting -> Val Acc: {gb_acc * 100:.2f}%, Macro F1: {gb_f1:.4f}")
    results["Gradient Boosting"] = {"accuracy": gb_acc, "f1": gb_f1, "model": gb}
    joblib.dump(gb, os.path.join(models_dir, "gradient_boosting.joblib"))

    # 4. Multi-Layer Perceptron (PyTorch Deep Neural Network with embedded normalization)
    print("\n--- [4/4] Training Multi-Layer Perceptron (PyTorch Deep NN) ---")
    torch.manual_seed(42)
    mlp = MudraMLP(input_dim=len(FEATURE_NAMES), num_classes=len(MUDRAS), mean=mean, std=std)

    train_tensor_x = torch.tensor(X_train_raw, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    val_tensor_x = torch.tensor(X_val_raw, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y), batch_size=32, shuffle=True)

    criterion = nn.NLLLoss() # Since output is Softmax, log(probs) with NLLLoss
    optimizer = optim.AdamW(mlp.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_mlp_acc = 0.0
    best_weights = None

    for epoch in range(1, 46):
        mlp.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            probs = mlp(bx)
            # Clip for numerical stability in log
            log_probs = torch.log(torch.clamp(probs, min=1e-7))
            loss = criterion(log_probs, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation check
        mlp.eval()
        with torch.no_grad():
            val_probs = mlp(val_tensor_x)
            val_preds = torch.argmax(val_probs, dim=1).cpu().numpy()
            cur_acc = accuracy_score(y_val, val_preds)

        scheduler.step(cur_acc)

        if cur_acc > best_mlp_acc:
            best_mlp_acc = cur_acc
            best_weights = mlp.state_dict().copy()

        if epoch % 10 == 0 or epoch == 45:
            print(f"Epoch {epoch:02d}/45 - Loss: {total_loss / len(train_loader):.4f} - Val Acc: {cur_acc * 100:.2f}% (Best: {best_mlp_acc * 100:.2f}%)")

    mlp.load_state_dict(best_weights)
    mlp.eval()
    with torch.no_grad():
        final_probs = mlp(val_tensor_x)
        mlp_preds = torch.argmax(final_probs, dim=1).cpu().numpy()
        mlp_acc = accuracy_score(y_val, mlp_preds)
        mlp_prec, mlp_rec, mlp_f1, _ = precision_recall_fscore_support(y_val, mlp_preds, average="macro")

    print(f"Multi-Layer Perceptron -> Val Acc: {mlp_acc * 100:.2f}%, Macro F1: {mlp_f1:.4f}")
    results["Multi-Layer Perceptron"] = {"accuracy": mlp_acc, "f1": mlp_f1, "model": mlp}
    torch.save(mlp.state_dict(), os.path.join(models_dir, "mlp_best.pt"))

    # Model Selection Summary
    print("\n================ MODEL BENCHMARK SUMMARY ================")
    print(f"{'Model Architecture':<30} | {'Validation Accuracy':<20} | {'Macro F1':<10}")
    print("-" * 68)
    best_name = None
    best_val_score = -1.0
    for name, r in results.items():
        print(f"{name:<30} | {r['accuracy'] * 100:.2f}%{'':<14} | {r['f1']:.4f}")
        if r["accuracy"] > best_val_score:
            best_val_score = r["accuracy"]
            best_name = name

    print(f"\n[SELECTED] Best Model Architecture: {best_name} with {best_val_score * 100:.2f}% Val Accuracy.")

    # Save benchmark report
    summary = {
        name: {"val_accuracy": float(r["accuracy"]), "val_macro_f1": float(r["f1"])}
        for name, r in results.items()
    }
    summary["best_model"] = best_name
    with open(os.path.join(models_dir, "validation_benchmarks.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    train_models(base)

