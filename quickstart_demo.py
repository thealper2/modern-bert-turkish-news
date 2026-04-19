"""
Quickstart demo for the ModernBERT fine-tuning pipeline.

Usage:
    python quickstart_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# 1. Classes
CATEGORIES = [
    "Magazin",
    "Siyaset",
    "Spor",
    "Sağlık",
    "Kültür-Sanat",
    "Finans-Ekonomi",
    "Bilim-Teknoloji",
    "Turizm",
    "Çevre",
]
NUM_CLASSES = len(CATEGORIES)

# 2. Configure the pipeline for a quick sanity check
from config import TrainingConfig

cfg = TrainingConfig(
    csv_path="./yedek/data/subset.csv",
    model_name="answerdotai/ModernBERT-base",  # swap with any BERT-like model for speed
    num_labels=NUM_CLASSES,
    max_seq_length=512,           # shorter for demo
    num_train_epochs=2,           # 1 epoch for quick test
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    weight_strategy="sqrt_inverse",
    early_stopping_patience=2,
    report_to=[],                 # disable TensorBoard for demo
    seed=42,
)

# 3. Run the pipeline (no hyperparameter search for demo speed)
from main import run_pipeline

run_pipeline(
    config=cfg,
    search_mode="none",
    #label_names=CATEGORIES,
    run_error_analysis=True,
)
