"""Held-out test metrics and reproducible plot generation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import CLASS_NAMES


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evaluation(y_true, scores, output_dir, history, model_path, split_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores)
    y_true = np.asarray(y_true, dtype=int)
    if scores.shape != (len(y_true), len(CLASS_NAMES)) or not len(y_true):
        raise ValueError("Invalid test prediction dimensions.")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or not np.allclose(scores.sum(1), 1, atol=1e-3):
        raise ValueError("Test scores are not valid softmax outputs.")
    predicted = scores.argmax(axis=1)
    report = classification_report(y_true, predicted, labels=range(6),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, predicted, labels=range(6))
    metrics = {"test_accuracy": float(accuracy_score(y_true, predicted)),
               "macro_precision": report["macro avg"]["precision"],
               "macro_recall": report["macro avg"]["recall"],
               "macro_f1": report["macro avg"]["f1-score"],
               "test_samples": len(y_true), "class_names": list(CLASS_NAMES),
               "classification_report": report, "confusion_matrix": matrix.tolist(),
               "model_sha256": file_sha256(model_path), "split_sha256": file_sha256(split_path)}
    (output_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir/"test_predictions.npz", labels=y_true, scores=scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Greens")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted category", ylabel="Actual category",
           title=f"Held-out test confusion matrix (n={len(y_true)})")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()/2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(output_dir/"confusion_matrix.png", dpi=160); plt.close(fig)
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in ((metric, "Training"), (f"val_{metric}", "Validation")):
            if key in history:
                ax.plot(range(1, len(history[key])+1), history[key], label=label)
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"Training and validation {metric}")
        ax.legend(); fig.tight_layout()
        fig.savefig(output_dir/f"{metric}.png", dpi=160); plt.close(fig)
    return metrics
