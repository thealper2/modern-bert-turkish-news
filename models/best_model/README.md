---
language: tr
tags:
  - text-classification
  - modernbert
  - turkish
  - news-classification
base_model: answerdotai/ModernBERT-base
datasets:
  - subset
metrics:
  - f1
  - accuracy
  - balanced_accuracy
---

# ModernBERT Turkish News Classification

Fine-tuned **answerdotai/ModernBERT-base** on **subset** for 9-class news classification.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | `answerdotai/ModernBERT-base` |
| Task | Multi-class Text Classification |
| Language | Turkish |
| Number of Classes | 9 |

## Labels

- 0: 0
- 1: 1
- 2: 2
- 3: 3
- 4: 4
- 5: 5
- 6: 6
- 7: 7
- 8: 8

## Evaluation Results (Test Set)

| Metric | Score |
|--------|-------|
| Balanced Accuracy | 0.7768 |
| Macro Precision | 0.7760 |
| Macro Recall | 0.7768 |
| Macro F1 | 0.7754 |
| Accuracy | 0.7768 |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7981 | 0.7545 | 0.7757 | 110 |
| 1 | 0.8021 | 0.7000 | 0.7476 | 110 |
| 2 | 0.8560 | 0.9727 | 0.9106 | 110 |
| 3 | 0.8829 | 0.8909 | 0.8869 | 110 |
| 4 | 0.7168 | 0.7364 | 0.7265 | 110 |
| 5 | 0.6991 | 0.7182 | 0.7085 | 110 |
| 6 | 0.7477 | 0.7273 | 0.7373 | 110 |
| 7 | 0.7672 | 0.8091 | 0.7876 | 110 |
| 8 | 0.7143 | 0.6818 | 0.6977 | 110 |

## Training Configuration

```json
{
  "model_name": "answerdotai/ModernBERT-base",
  "num_labels": 9,
  "max_seq_length": 512,
  "csv_path": "./yedek/data/subset.csv",
  "text_column": "Haber G\u00f6vdesi",
  "label_column": "S\u0131n\u0131f",
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "learning_rate": 2e-05,
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 32,
  "weight_decay": 0.01,
  "num_train_epochs": 2,
  "warmup_ratio": 0.1,
  "gradient_accumulation_steps": 1,
  "early_stopping_patience": 2,
  "label_smoothing_factor": 0.0,
  "weight_strategy": "sqrt_inverse",
  "fp16": false,
  "bf16": true,
  "seed": 42,
  "output_dir": "/mnt/d/work2/modern-bert-turkish-news/outputs/checkpoints/best_model",
  "logging_dir": "/mnt/d/work2/modern-bert-turkish-news/outputs/logs",
  "save_total_limit": 2,
  "load_best_model_at_end": true,
  "metric_for_best_model": "eval_f1_macro",
  "greater_is_better": true,
  "push_to_hub": false,
  "hub_model_id": null,
  "hub_token": null,
  "report_to": [],
  "lowercase": false,
  "remove_punctuation": true
}
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
print(f"Predicted class: {predicted_class}")
```

## Limitations

- Trained on Turkish news; performance on other domains may vary.
- Class distribution imbalance handled with sqrt-inverse class weighting.
