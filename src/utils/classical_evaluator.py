# src/utils/classical_evaluator.py

import time
import psutil
import tracemalloc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_train, y_train, X_test, y_test, label="Model", quantum_instance=None, fit_func=None, predict_func=None):
    print(f"\n--- {label} ---")
    process = psutil.Process()

    # Define default execution closures if custom kernel wrappers aren't provided
    if fit_func is None:
        fit_func = lambda: model.fit(X_train, y_train)
    if predict_func is None:
        predict_func = lambda: model.predict(X_test)

    # --- Training Phase Benchmark ---
    tracemalloc.start()
    cpu_start_train = process.cpu_times()
    start_train = time.perf_counter()

    fit_func()

    end_train = time.perf_counter()
    cpu_end_train = process.cpu_times()
    train_runtime = end_train - start_train
    
    # Process-specific CPU time spent during training
    train_cpu_seconds = (cpu_end_train.user - cpu_start_train.user) + (cpu_end_train.system - cpu_start_train.system)
    _, train_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Prediction Phase Benchmark ---
    tracemalloc.start()
    cpu_start_pred = process.cpu_times()
    start_pred = time.perf_counter()

    y_pred = predict_func()

    end_pred = time.perf_counter()
    cpu_end_pred = process.cpu_times()
    predict_runtime = end_pred - start_pred

    _, pred_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Total peak memory footprint across train/predict in MB
    peak_mem_mb = max(train_peak_bytes, pred_peak_bytes) / (1024 ** 2)

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    train_acc = accuracy_score(y_train, model.predict(X_train))
    generalization_gap = train_acc - acc

    model_size = len(getattr(model, "support_", []))

    # Feasibility (always True for classical SVM)
    feasibility = True

    # Quantum backend info (optional)
    backend_name = "N/A"
    if quantum_instance:
        backend = quantum_instance.backend
        backend_name = backend.name()

    return {
        "model": label,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "train_accuracy": round(train_acc, 4),
        "generalization_gap": round(generalization_gap, 4),
        "training_runtime": round(train_runtime, 4),
        "prediction_runtime": round(predict_runtime, 4),
        "feasibility": feasibility,
        "memory_MB": round(peak_mem_mb, 2),
        "cpu_percent": round(train_cpu_seconds, 2),  # Process CPU execution time in seconds
        "support_vectors": model_size,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "backend": backend_name
    }