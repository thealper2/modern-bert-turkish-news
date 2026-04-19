"""
Metric computation for multi-class text classification.
Includes balanced accuracy, precision, recall, macro F1 and per-class stats.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import EvalPrediction

from utils.logger import get_logger

logger = get_logger(__name__)

# Trainer-compatible compute_metrics
def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute classification metrics expected by the HuggingFace Trainer.
    
    Called automatically by Trainer during evaluation; receives an
    EvalPrediction namedtuple with logits and label_ids.
    
    Ags:
        eval_pred: EvalPrediction(predictions=logits, label_ids=labels).
        
    Returns:
        Dict with balanced_accuracy, precision, recall, f1_macro keys.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    
# Full evaluation report
def full_evaluation_report(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    label_names: list[str] | None = None,
) -> dict:
    """
    Compute a comprehensive evaluation report.
    
    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        label_names: Optional human-readable class names.
        
    Returns:
        Dict containing scalar metrics, per-class stats,
        confusion matrix and the sklearn classification report string.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    num_classes = len(np.unique(y_true))
    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]
        
    # Aggregate metrics
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class = [
        {
            "class": label_names[i],
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int((y_true == i).sum()),
        }
        for i in range(len(label_names))
    ]
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    
    logger.info("Evaluation Results:\n%s", report)
    
    return {
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    
# Error analysis
def error_analysis(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
    max_samples: int = 20,
) -> list[dict]:
    """
    Collect misclassified examples for qualitative error analysis.
    
    Args:
        texts: Original text samples.
        y_true: True labels.
        y_pred: Predicted labels.
        label_names: Optional class name list.
        max_samples: Maximum number of errors to return.
        
    Returns:
        List of dicts with text, true_label, predicted_label.    
    """
    errors = []
    
    for text, true, pred in zip(texts, y_true, y_pred):
        if true != pred:
            errors.append({
                "text": text,
                "true_label": label_names[true] if label_names else str(true),
                "predicted_label": label_names[pred] if label_names else str(pred),
            })
        
        if len(errors) >= max_samples:
            break
            
    logger.info("Error analysis: %d misclassified examples collected", len(errors))
    return errors