# src/utils/quantum_evaluator.py

import time
import psutil
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_quantum_model(model, X_train, y_train, X_test, y_test,
                           label="QSVM_Model", feature_map=None, backend="qasm_simulator"):
    print(f"\n--- {label} ---")

    # Training runtime
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()
    train_runtime = end_train - start_train

    # Prediction runtime
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    end_pred = time.perf_counter()
    predict_runtime = end_pred - start_pred


    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    train_acc = accuracy_score(y_train, model.predict(X_train))
    generalization_gap = train_acc - acc

    # Resource usage
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 ** 2)
    cpu_percent = psutil.cpu_percent(interval=None)
    model_size = len(getattr(model, "support_", []))  # support vectors

    # Quantum‑specific metrics
    num_qubits = feature_map.num_qubits if feature_map else None
    circuit_depth = feature_map.decompose().depth() if feature_map else None
    feasibility = True  # assume True unless kernel matrix fails

    # Return metrics dictionary
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
        "memory_MB": round(mem_mb, 2),
        "cpu_percent": round(cpu_percent, 2),
        "support_vectors": model_size,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "backend": backend,
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth
    }