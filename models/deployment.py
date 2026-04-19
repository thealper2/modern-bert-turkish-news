"""
Model card generation and HuggingFace Hub deployment utilities.
"""

import json
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# Model card
def generate_model_card(
    model_name: str,
    dataset_name: str,
    num_labels: int,
    label_names: list[str],
    metrics: dict,
    per_class: list[dict],
    config_dict: dict,
    output_dir: str | Path,
) -> Path:
    """
    Generate a README.md model card for HuggingFace Hub.

    Args:
        model_name: Base pretrained model identifier.
        dataset_name: Human-readable dataset name.
        num_labels: Number of output classes.
        label_names: List of class name strings.
        metrics: Aggregated evaluation metrics dict.
        per_class: Per-class metrics list from full_evaluation_report().
        config_dict: Training config as a flat dict.
        output_dir: Directory where README.md will be written.

    Returns:
        Path to the generated README.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_lines = "\n".join(
        f"- {i}: {name}" for i, name in enumerate(label_names)
    )

    per_class_rows = "\n".join(
        f"| {r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |"
        for r in per_class
    )

    card = f"""---
language: tr
tags:
  - text-classification
  - modernbert
  - turkish
  - news-classification
base_model: {model_name}
datasets:
  - {dataset_name}
metrics:
  - f1
  - accuracy
  - balanced_accuracy
---

# ModernBERT Turkish News Classification

Fine-tuned **{model_name}** on **{dataset_name}** for {num_labels}-class news classification.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | `{model_name}` |
| Task | Multi-class Text Classification |
| Language | Turkish |
| Number of Classes | {num_labels} |

## Labels

{label_lines}

## Evaluation Results (Test Set)

| Metric | Score |
|--------|-------|
| Balanced Accuracy | {metrics.get('balanced_accuracy', 'N/A'):.4f} |
| Macro Precision | {metrics.get('precision_macro', 'N/A'):.4f} |
| Macro Recall | {metrics.get('recall_macro', 'N/A'):.4f} |
| Macro F1 | {metrics.get('f1_macro', 'N/A'):.4f} |
| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
{per_class_rows}

## Training Configuration

```json
{json.dumps(config_dict, indent=2, default=str)}
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "your-hub-username/your-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

text = "Türk haber metni buraya gelecek"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = logits.argmax(-1).item()
print(f"Predicted class: {{predicted_class}}")
```

## Limitations

- Trained on Turkish news; performance on other domains may vary.
- Class distribution imbalance handled with sqrt-inverse class weighting.
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(card, encoding="utf-8")
    logger.info("Model card written to '%s'", readme_path)
    return readme_path


# Push to Hub
def push_to_hub(
    model,
    tokenizer,
    hub_model_id: str,
    hub_token: Optional[str] = None,
    commit_message: str = "Upload fine-tuned ModernBERT model",
) -> None:
    """
    Push model and tokeniser to the HuggingFace Hub.

    Args:
        model: Fine-tuned PreTrainedModel.
        tokenizer: Corresponding tokenizer.
        hub_model_id: Hub repository ID (e.g. 'username/model-name').
        hub_token: HuggingFace API token; reads HF_TOKEN env var if None.
        commit_message: Commit message for the Hub push.

    Raises:
        ValueError: If hub_model_id is empty.
    """
    if not hub_model_id:
        raise ValueError("hub_model_id must be set to push to the Hub.")

    logger.info("Pushing model to HuggingFace Hub: '%s'", hub_model_id)

    kwargs: dict = {"commit_message": commit_message}
    if hub_token:
        kwargs["token"] = hub_token

    tokenizer.push_to_hub(hub_model_id, **kwargs)
    model.push_to_hub(hub_model_id, **kwargs)
    logger.info("Model successfully pushed to '%s'", hub_model_id)


# Save best model locally
def save_best_model(
    model,
    tokenizer,
    output_dir: str | Path,
) -> Path:
    """
    Save the best fine-tuned model and tokeniser to a local directory.

    Args:
        model: Fine-tuned model.
        tokenizer: Tokenizer.
        output_dir: Target directory.

    Returns:
        Path to the saved directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Best model saved to '%s'", output_dir)
    return output_dir
