#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage:
python fhe_lr_classification.py \
  --data-file data_p/data.address.csv \
  --sample-fraction 0.04 \
  --use-class-weights \
  --lr-bits 8 \
  --simulate-max-samples 4096 \
  --execute-samples 256 \
  --save-results
"""
from sklearn.preprocessing import StandardScaler
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

import concrete.compiler
from concrete.ml.sklearn import LogisticRegression as FHE_LR

# -----------------------------------------------------------------------------
# Feature definitions
# -----------------------------------------------------------------------------
BASIC = [
    "f_tx", "f_received", "f_coinbase",
    "f_spent_digits_-3", "f_spent_digits_-2", "f_spent_digits_-1", "f_spent_digits_0",
    "f_spent_digits_1", "f_spent_digits_2", "f_spent_digits_3", "f_spent_digits_4",
    "f_spent_digits_5", "f_spent_digits_6", "f_received_digits_-3", "f_received_digits_-2",
    "f_received_digits_-1", "f_received_digits_0", "f_received_digits_1", "f_received_digits_2",
    "f_received_digits_3", "f_received_digits_4", "f_received_digits_5", "f_received_digits_6",
    "r_payback", "n_inputs_in_spent", "n_outputs_in_spent",
]

EXTRA = [
    "n_tx", "total_days", "n_spent", "n_received", "n_coinbase", "n_payback",
    "total_spent_btc", "total_received_btc",
    "total_spent_usd", "total_received_usd",
    "mean_balance_btc", "std_balance_btc",
    "mean_balance_usd", "std_balance_usd",
]

MOMENTS = [
    "interval_1st_moment", "interval_2nd_moment", "interval_3rd_moment", "interval_4th_moment",
    "dist_total_1st_moment", "dist_total_2nd_moment", "dist_total_3rd_moment", "dist_total_4th_moment",
    "dist_coinbase_1st_moment", "dist_coinbase_2nd_moment", "dist_coinbase_3rd_moment", "dist_coinbase_4th_moment",
    "dist_spend_1st_moment", "dist_spend_2nd_moment", "dist_spend_3rd_moment", "dist_spend_4th_moment",
    "dist_receive_1st_moment", "dist_receive_2nd_moment", "dist_receive_3rd_moment", "dist_receive_4th_moment",
    "dist_payback_1st_moment", "dist_payback_2nd_moment", "dist_payback_3rd_moment", "dist_payback_4th_moment",
]

PATTERNS = [
    "tx_input", "tx_output",
    "n_multi_in", "n_multi_out", "n_multi_in_out",
]

# -----------------------------------------------------------------------------
# Args / config
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Logistic Regression to an FHE-friendly pipeline")

    parser.add_argument("--output-path", type=str, default="./data_p")
    parser.add_argument("--result-path", type=str, default="./result")
    parser.add_argument("--scheme", type=str, choices=["address", "entity"], default="address")
    parser.add_argument("--data-file", type=str, default="", help="Optional explicit CSV path.")
    parser.add_argument("--feature-type", type=str, default="bemp")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--use-class-weights", action="store_true", help="Enable smoothed class weights")
    
    # FHE / Compile settings
    parser.add_argument("--lr-bits", type=int, default=8, help="Quantization bits for Logistic Regression")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of CPU threads for training")
    parser.add_argument("--calibration-max-samples", type=int, default=5000)
    parser.add_argument("--simulate-max-samples", type=int, default=4096)
    parser.add_argument("--execute-samples", type=int, default=256)
    
    parser.add_argument("--save-results", action="store_true")

    return parser.parse_args()


def build_feature_list(feature_type: str) -> List[str]:
    features: List[str] = []
    if "b" in feature_type: features += BASIC
    if "e" in feature_type: features += EXTRA
    if "m" in feature_type: features += MOMENTS
    if "p" in feature_type: features += PATTERNS
    return features


def load_sample(file_path: Path, sample_fraction: float, random_state: int) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=100_000)):
        if sample_fraction >= 1.0:
            sampled = chunk
        else:
            sampled = chunk.sample(frac=sample_fraction, random_state=random_state + chunk_idx)
        chunks.append(sampled)
    return pd.concat(chunks, ignore_index=True)


# -----------------------------------------------------------------------------
# Metrics & Reporting
# -----------------------------------------------------------------------------
def normalize_cm(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm / row_sum


def get_all_labels(class_names: Sequence[str]) -> List[int]:
    """Return a stable full label list [0, 1, ..., n_classes-1] to prevent missing class errors."""
    return list(range(len(class_names)))


def process_and_print_metrics(
    title: str, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Sequence[str], 
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    
    labels = get_all_labels(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = normalize_cm(cm)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names), 
        output_dict=True, zero_division=0
    )
    
    overall_acc = accuracy_score(y_true, y_pred)
    
    auc_score = None
    if y_prob is not None:
        try:
            auc_score = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            pass

    # Print "戰報" (Battle Report)
    print("="*60)
    print(f"  {title}")
    print("="*60)
    print(f"  Overall Accuracy : {overall_acc:.4f}")
    print(f"  Macro Precision  : {report['macro avg']['precision']:.4f}")
    print(f"  Macro Recall     : {report['macro avg']['recall']:.4f}")
    print(f"  Macro F1-Score   : {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1      : {report['weighted avg']['f1-score']:.4f}")
    print(f"  ROC-AUC (ovr)    : {auc_score:.4f}" if auc_score is not None else "  ROC-AUC (ovr)    : Not available")
    
    print("\n  [Class-level Metrics]")
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - tp)
        cls_acc = (tp + tn) / cm.sum() if cm.sum() > 0 else 0
        
        print(f"   - {cls:<9}: Precision: {report[cls]['precision']:.4f} | Recall: {report[cls]['recall']:.4f} | "
              f"F1: {report[cls]['f1-score']:.4f} | Acc: {cls_acc:.4f}")
        
    print("\n  [Confusion Matrix]")
    print(cm)
    print("="*60 + "\n")

    metrics: Dict[str, Any] = {
        "confusion_matrix": cm_norm,
        "classification_report": report,
        "auc_macro_ovr": auc_score
    }
    return metrics


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_serializable(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    
    output_path = Path(args.output_path)
    result_path = Path(args.result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data_file = Path(args.data_file) if args.data_file else output_path / f"nanzero_normalization_data.{args.scheme}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_file}")

    print(f" Loading data from {data_file} (fraction: {args.sample_fraction})...")
    data = load_sample(data_file, args.sample_fraction, args.random_state)
    features = build_feature_list(args.feature_type)
    
    X = data.get(features).values.astype(np.float32)
    y = data['class'].values.astype(np.int64)
    class_names = ["Exchange", "Faucet", "Gambling", "Market", "Mixer", "Pool"]

    print(f"  Data shape: {X.shape}")
    print(f"  Classes: {class_names}")

    # 2. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    print(" Normalizing data (StandardScaler + Clipping)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    
    clip_val = 8.0
    X_train = np.clip(X_train, -clip_val, clip_val)
    X_test = np.clip(X_test, -clip_val, clip_val)
    # 3. Class Weights (Square Root Smoothing)
    sample_weight = np.ones((len(y_train), ), dtype='float64')
    if args.use_class_weights:
        print( "Applying smoothed class weights...")
        unique_classes = np.unique(y_train)
        balanced_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        smoothed_weights = np.sqrt(balanced_weights)
        weight_dict = dict(zip(unique_classes, smoothed_weights))
        sample_weight = np.array([weight_dict[y_cls] for y_cls in y_train])

    # 4. Train FHE Logistic Regression
    fhe_device = "cuda" if concrete.compiler.check_gpu_available() else "cpu"
    print(f"\n Initializing FHE Logistic Regression (n_bits={args.lr_bits}) on {fhe_device.upper()}...")
    clf = FHE_LR(n_bits=args.lr_bits, n_jobs=args.n_jobs)
    
    train_t0 = time.time()
    try:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        print(" Current concrete-ml LR doesn't support sample_weight directly. Falling back to unweighted training.")
        clf.fit(X_train, y_train)
    train_time = time.time() - train_t0
    print(f" Training completed in {train_time:.2f} s")

    # 5. Compile Circuit
    calib_size = min(args.calibration_max_samples, X_train.shape[0])
    calib_idx = np.random.choice(X_train.shape[0], calib_size, replace=False)
    
    print(f"\n Compiling FHE circuit using {calib_size} calibration samples...")
    compile_t0 = time.time()
    circuit = clf.compile(X_train[calib_idx], device=fhe_device)
    compile_time = time.time() - compile_t0
    print(f" Compilation finished in {compile_time:.2f} s | Max integer bit width: {circuit.graph.maximum_integer_bit_width()} bits")

    # =========================================================
    # 6. Evaluation (Plaintext, Simulate, Execute)
    # =========================================================
    
    # --- A. Plaintext ---
    y_pred_plain = clf.predict(X_test)
    y_prob_plain = clf.predict_proba(X_test) # LR supports direct probability output in plaintext
    plain_metrics = process_and_print_metrics("Plaintext Inference (Full Test)", y_test, y_pred_plain, class_names, y_prob_plain)

    # --- B. Simulate ---
    sim_n = min(args.simulate_max_samples, len(X_test))
    X_sim, y_sim = X_test[:sim_n], y_test[:sim_n]
    
    print(f"\n Extracting {sim_n} samples for FHE Simulate...")
    sim_t0 = time.time()
    y_pred_sim = clf.predict(X_sim, fhe="simulate")
    sim_time = time.time() - sim_t0
    print(f" Simulation completed in {sim_time:.2f} s")
    
    sim_metrics = process_and_print_metrics(f"FHE Simulate (N={sim_n})", y_sim, y_pred_sim, class_names)

    # --- C. Execute ---
    exec_n = min(args.execute_samples, len(X_test))
    X_exec, y_exec = X_test[:exec_n], y_test[:exec_n]
    
    print(f"\n Extracting {exec_n} samples for Real FHE Execute...")
    exec_t0 = time.time()
    y_pred_exec = clf.predict(X_exec, fhe="execute")
    exec_time = time.time() - exec_t0
    print(f" Execution completed in {exec_time:.2f} s")
    
    exec_metrics = process_and_print_metrics(f"Real FHE Execute (N={exec_n})", y_exec, y_pred_exec, class_names)

    # =========================================================
    # 7. Save Results (Compatible with NN structure)
    # =========================================================
    extra_meta = {
        "model_type": "LogisticRegression",
        "n_bits": args.lr_bits,
        "sample_fraction": args.sample_fraction,
        "use_class_weights": args.use_class_weights,
        "train_time_sec": train_time,
        "compile_time_sec": compile_time,
        "simulate_time_sec": sim_time,
        "execute_time_sec": exec_time,
        "notes": [
            "Single split mode with MinMaxScaler.",
            "LR predict_proba used for plaintext AUC.",
            "Missing classes in execute handled safely via get_all_labels."
        ]
    }

    if args.save_results:
        # Mock Feature Importance (LR uses coef_, we store zeros to prevent old plot code crashing)
        fi_list = [np.zeros(X_train.shape[1])]
        
        y_train_pred = clf.predict(X_train)
        y_train_prob = clf.predict_proba(X_train)
        train_metrics = process_and_print_metrics("Training Set Eval (Hidden)", y_train, y_train_pred, class_names, y_train_prob)
        
        results = {
            "mode": "fhe_lr",
            "feature_type": args.feature_type,
            "scheme": args.scheme,
            "features": list(features),
            "class_names": list(class_names),
            "train_cm_list": [train_metrics["confusion_matrix"]],
            "valid_cm_list": [plain_metrics["confusion_matrix"]], 
            "test_cm_list": [plain_metrics["confusion_matrix"]], 
            "train_rp_list": [train_metrics["classification_report"]],
            "valid_rp_list": [plain_metrics["classification_report"]],
            "train_auc_list": [train_metrics.get("auc_macro_ovr")],
            "valid_auc_list": [plain_metrics.get("auc_macro_ovr")],
            "compiled_simulate": sim_metrics,
            "compiled_execute": exec_metrics,
            "fi_list": fi_list,
            "meta": extra_meta,
        }

        base_name = f"fhe_lr.{args.feature_type}.{args.scheme}"
        if not args.use_class_weights:
            base_name += ".no_cs"
            
        pkl_path = result_path / f"{base_name}_results.pkl"
        json_path = result_path / f"{base_name}_results.json"
        model_path = result_path / f"{base_name}_model.pkl"

        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        json_path.write_text(json.dumps(to_serializable(results), indent=2, ensure_ascii=False))
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        print(f"\n Saved results to:\n- {pkl_path}\n- {json_path}\n- {model_path}")

    print("\n Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
