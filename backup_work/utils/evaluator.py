# src/utils/evaluator.py

import time
import psutil
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_train, y_train, X_test, y_test, label="Model", quantum_instance=None):
    print(f"\n--- {label} ---")

    # Training time
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    train_duration = end_time - start_time

    # Accuracy metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    generalization_gap = train_acc - acc

    print(f"Training time: {train_duration:.4f} seconds")
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Generalization Gap: {generalization_gap:.4f}")

    # Resource usage
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Memory used: {mem:.2f} MB")

    # Quantum backend info
    backend_name = "N/A"
    if quantum_instance:
        backend = quantum_instance.backend
        backend_name = backend.name()
        print(f"Quantum backend: {backend_name}")
        if hasattr(backend, "options") and hasattr(backend.options, "noise_model"):
            print("Noise model: injected")

    # Return metrics for logging
    return {
        "model": label,
        "accuracy": round(acc, 4),
        "train_accuracy": round(train_acc, 4),
        "generalization_gap": round(generalization_gap, 4),
        "training_time": round(train_duration, 4),
        "memory_MB": round(mem, 2),
        "backend": backend_name
    }