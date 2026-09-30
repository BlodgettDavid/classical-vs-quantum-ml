# src/utils/quantum_evaluator.py

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


def evaluate_quantum_model(qsvm, qkernel, X_train, y_train, X_test, y_test,
                           label="QSVM_Model", feature_map=None, backend_name="statevector"):
    """
    Evaluates a Quantum SVM model, isolating training and prediction benchmarks.
    """
    print(f"\n--- {label} ---")
    process = psutil.Process()

    # --- Training Phase Benchmark (Kernel Evaluation + SVM Fit) ---
    tracemalloc.start()
    cpu_start_train = process.cpu_times()
    start_train = time.perf_counter()

    matrix_train = qkernel.evaluate(x_vec=X_train)
    qsvm.fit(matrix_train, y_train)

    end_train = time.perf_counter()
    cpu_end_train = process.cpu_times()
    train_runtime = end_train - start_train

    train_cpu_seconds = (cpu_end_train.user - cpu_start_train.user) + (cpu_end_train.system - cpu_start_train.system)
    _, train_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Prediction Phase Benchmark ---
    tracemalloc.start()
    cpu_start_pred = process.cpu_times()
    start_pred = time.perf_counter()

    matrix_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
    y_pred = qsvm.predict(matrix_test)

    end_pred = time.perf_counter()
    cpu_end_pred = process.cpu_times()
    predict_runtime = end_pred - start_pred

    _, pred_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory across phases in MB
    peak_mem_mb = max(train_peak_bytes, pred_peak_bytes) / (1024 ** 2)

    # Classification Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    y_train_pred = qsvm.predict(matrix_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    generalization_gap = train_acc - acc

    model_size = len(getattr(qsvm, "support_", []))

    # Circuit Metadata
    num_qubits = feature_map.num_qubits if feature_map else 0
    circuit_depth = feature_map.decompose().depth() if feature_map else 0

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
        "feasibility": True,
        "memory_MB": round(peak_mem_mb, 2),
        "cpu_percent": round(train_cpu_seconds, 2),
        "support_vectors": model_size,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "backend": backend_name,
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth,
        "matrix_train": matrix_train,
        "y_pred": y_pred,
    }