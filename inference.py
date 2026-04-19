"""
Inference script for the fine-tuned ModernBERT classification model.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import get_logger
from utils.seed import get_device

logger = get_logger("inference")


# Model loading
def load_inference_model(
    model_dir: str | Path,
    device: Optional[torch.device] = None,
):
    """
    Load the fine-tuned model and tokeniser from a local directory.

    Args:
        model_dir: Path to the saved model directory (output of save_best_model).
        device: Target device; auto-detected if None.

    Returns:
        Tuple of (model, tokenizer, device).

    Raises:
        FileNotFoundError: If model_dir does not exist or is missing key files.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: '{model_dir}'")

    required = ["config.json"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model directory '{model_dir}' is missing: {missing}. "
            "Run the pipeline first to generate a saved model."
        )

    if device is None:
        device = get_device()

    logger.info("Loading model from '%s' → device: %s", model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    logger.info("Model loaded | classes: %d", num_labels)
    return model, tokenizer, device

# Core prediction
def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 32,
    show_confidence: bool = True,
) -> list[dict]:
    """
    Run inference on a list of text strings.

    Args:
        texts: Input strings to classify.
        model: Fine-tuned model (eval mode).
        tokenizer: Matching tokeniser.
        device: Target device.
        max_length: Maximum token sequence length.
        batch_size: Number of samples per forward pass.
        show_confidence: Include per-class probability scores when True.

    Returns:
        List of result dicts with keys:
            - text: original input text
            - predicted_class: integer class index
            - predicted_label: human-readable label (from model.config.id2label)
            - confidence: probability of the predicted class
            - scores: dict of {label: probability} for all classes (optional)
    """
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    results: list[dict] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]

        encoding = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**encoding).logits  # (batch, num_labels)

        probs = F.softmax(logits, dim=-1).cpu().numpy()  # (batch, num_labels)
        pred_ids = np.argmax(probs, axis=-1)             # (batch,)

        for text, pred_id, prob_row in zip(chunk, pred_ids, probs):
            result: dict = {
                "text": text,
                "predicted_class": int(pred_id),
                "predicted_label": id2label.get(int(pred_id), str(pred_id)),
                "confidence": float(prob_row[pred_id]),
            }
            if show_confidence:
                result["scores"] = {
                    id2label.get(i, str(i)): round(float(p), 6)
                    for i, p in enumerate(prob_row)
                }
            results.append(result)

        logger.debug(
            "Processed batch %d–%d / %d",
            start + 1, min(start + batch_size, len(texts)), len(texts),
        )

    return results


# Input loading

def load_texts_from_file(
    file_path: str | Path,
    text_column: str = "text",
) -> tuple[list[str], list[int] | None]:
    """
    Load text samples from a .txt or .csv file.

    For .txt files: each non-empty line is treated as one sample.
    For .csv files: *text_column* is read; an optional 'label' column is
    returned as ground-truth labels when present.

    Args:
        file_path: Path to .txt or .csv file.
        text_column: Column name to read from CSV files.

    Returns:
        Tuple of (texts, labels_or_None).

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the CSV is missing the specified text_column.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: '{file_path}'")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        raw = file_path.read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in raw if line.strip()]
        logger.info("Loaded %d lines from '%s'", len(texts), file_path)
        return texts, None

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(file_path)
        if text_column not in df.columns:
            raise ValueError(
                f"CSV file '{file_path}' has no column '{text_column}'. "
                f"Available columns: {list(df.columns)}"
            )
        texts = df[text_column].fillna("").astype(str).tolist()
        labels = df["label"].astype(int).tolist() if "label" in df.columns else None
        logger.info(
            "Loaded %d rows from CSV '%s' (labels: %s)",
            len(texts), file_path, labels is not None,
        )
        return texts, labels

    raise ValueError(f"Unsupported file extension '{suffix}'. Use .txt or .csv.")

# Output saving
def save_results(
    results: list[dict],
    output_path: str | Path,
    output_format: str = "json",
) -> Path:
    """
    Persist prediction results to disk.

    Args:
        results: List of result dicts from predict_batch().
        output_path: Destination file path (extension overrides output_format).
        output_format: 'json' or 'csv'.

    Returns:
        Path to the saved file.
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv" or output_path.suffix.lower() == ".csv":
        # Flatten scores dict into separate columns for CSV readability
        rows = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "scores"}
            if "scores" in r:
                for label, score in r["scores"].items():
                    row[f"score_{label}"] = score
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to '%s'", output_path)
    return output_path


# Optional accuracy report when ground-truth labels are available
def _print_accuracy_report(results: list[dict], true_labels: list[int]) -> None:
    """Print a quick accuracy / F1 summary when ground-truth labels are present."""
    from sklearn.metrics import classification_report

    y_pred = [r["predicted_class"] for r in results]
    y_true = true_labels[: len(y_pred)]

    id2label = {r["predicted_class"]: r["predicted_label"] for r in results}
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [id2label.get(l, str(l)) for l in labels]

    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    logger.info("Ground-truth evaluation:\n%s", report)
    print("\n=== Ground-truth Evaluation ===")
    print(report)


# High-level entry point
def run_inference(
    model_dir: str | Path,
    text: Optional[str] = None,
    file_path: Optional[str | Path] = None,
    text_column: str = "text",
    max_length: int = 512,
    batch_size: int = 32,
    show_confidence: bool = True,
    output_path: Optional[str | Path] = None,
    output_format: str = "json",
    label_names: Optional[list[str]] = None,
) -> list[dict]:
    """
    Full inference pipeline: load model → load inputs → predict → (save).

    Args:
        model_dir: Directory containing the saved model.
        text: Single text string to classify (mutually exclusive with file_path).
        file_path: Path to a .txt or .csv file of texts.
        text_column: CSV column name (used only for CSV files).
        max_length: Maximum tokenisation length.
        batch_size: Inference batch size.
        show_confidence: Include full probability distribution in results.
        output_path: If provided, save results to this path.
        output_format: 'json' or 'csv' (used when output_path has no extension).
        label_names: Override model's id2label mapping with custom names.

    Returns:
        List of prediction result dicts.

    Raises:
        ValueError: If neither text nor file_path is provided.
    """
    if text is None and file_path is None:
        raise ValueError("Provide either --text or --file.")

    model, tokenizer, device = load_inference_model(model_dir)

    # Override label names if supplied
    if label_names:
        if len(label_names) != model.config.num_labels:
            logger.warning(
                "label_names count (%d) does not match model num_labels (%d). Ignoring.",
                len(label_names), model.config.num_labels,
            )
        else:
            model.config.id2label = {i: name for i, name in enumerate(label_names)}
            model.config.label2id = {name: i for i, name in enumerate(label_names)}

    # Collect input texts
    true_labels: list[int] | None = None
    if text is not None:
        texts = [text]
    else:
        texts, true_labels = load_texts_from_file(file_path, text_column)

    if not texts:
        logger.warning("No input texts found. Exiting.")
        return []

    logger.info("Running inference on %d sample(s) …", len(texts))
    results = predict_batch(
        texts, model, tokenizer, device,
        max_length=max_length,
        batch_size=batch_size,
        show_confidence=show_confidence,
    )

    # Print summary to stdout
    if text is not None:
        r = results[0]
        print(f"\nText     : {r['text']}")
        print(f"Predicted: {r['predicted_label']} (class {r['predicted_class']})")
        print(f"Confidence: {r['confidence']:.4f}")
        if show_confidence and "scores" in r:
            print("All scores:")
            for label, score in sorted(r["scores"].items(), key=lambda x: -x[1]):
                print(f"  {label:<20} {score:.6f}")
    else:
        # Summary for file input
        from collections import Counter
        counts = Counter(r["predicted_label"] for r in results)
        print(f"\n=== Inference Summary ({len(results)} samples) ===")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            print(f"  {label:<20} {count:>5}  ({pct:.1f}%)")

    # Ground-truth comparison (CSV with label column)
    if true_labels is not None:
        _print_accuracy_report(results, true_labels)

    # Persist results
    if output_path is not None:
        save_results(results, output_path, output_format)

    return results


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ModernBERT inference — classify text from a file or string.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_dir", type=str, default="models/best_model",
        help="Path to the saved model directory (default: models/best_model)",
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Single text string to classify")
    input_group.add_argument(
        "--file", type=str,
        help="Path to a .txt (one sample per line) or .csv file",
    )

    parser.add_argument(
        "--text_column", type=str, default="text",
        help="CSV column containing the text (default: text)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum token sequence length (default: 512)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--show_confidence", action="store_true", default=True,
        help="Include per-class probability scores in output (default: True)",
    )
    parser.add_argument(
        "--no_confidence", dest="show_confidence", action="store_false",
        help="Suppress per-class probability scores",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Save predictions to this path (e.g. outputs/predictions.json)",
    )
    parser.add_argument(
        "--output_format", type=str, default="json", choices=["json", "csv"],
        help="Output format when output_path has no extension (default: json)",
    )
    parser.add_argument(
        "--label_names", type=str, nargs="*", default=None,
        help="Human-readable class names in label-index order (space-separated)",
    )

    args = parser.parse_args()

    run_inference(
        model_dir=args.model_dir,
        text=args.text,
        file_path=args.file,
        text_column=args.text_column,
        max_length=args.max_length,
        batch_size=args.batch_size,
        show_confidence=args.show_confidence,
        output_path=args.output_path,
        output_format=args.output_format,
        label_names=args.label_names,
    )
