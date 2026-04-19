"""
Model and tokenizer initialization for ModernBERT sequence classification.
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from utils.logger import get_logger

logger = get_logger(__name__)

# Tokenizer
def load_tokenizer(
    model_name: str,
    max_length: int = 512,
    cache_dir: Optional[str | Path] = None,
) -> PreTrainedTokenizerBase:
    """
    Load an AutoTokenizer compatible with ModernBERT:
        
    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum sequence length (stored on tokenizer for convenience).
        cache_dir: Optional local cache directory.
        
    Returns:
        Configured tokenizer instance.
    """
    logger.info("Loading tokenizer from '%s'", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    tokenizer.model_max_length = max_length
    logger.info("Tokenizer vocab size: %d | max_length: %d", tokenizer.vocab_size, max_length)
    return tokenizer
    
# Model
def load_model(
    model_name: str,
    num_labels: int,
    label2id: Optional[dict[str, int]] = None,
    id2label: Optional[dict[int, str]] = None,
    cache_dir: Optional[str | Path] = None,
) -> PreTrainedModel:
    """
    Load AutoModelForSequenceClassification from a pretrained checkpoint.
    
    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        label2id: Optional label -> id mapping for model config.
        id2label: Optional id -> label mapping for model config.
        cache_dir: Optional local cache directory.
        
    Returns:
        Model ready for fine-tuning.
    """
    logger.info("Loading model '%s' with %d labels", model_name, num_labels)
    
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id or {str(i): i for i in range(num_labels)},
        id2label=id2label or {i: str(i) for i in range(num_labels)},
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True, # classifier head is always re-initialized
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{n_params:,}", f"{n_trainable:,}",
    )
    return model
    
# Data collator
def get_data_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """
    Return a dynamic-padding data collator.
    
    Args:
        tokenizer: Tokenizer instance (must have a pad token).
        
    Returns:
        DataCollatorWithPadding instance.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
# Tokenization for DataFrames
def tokenize_dataframe(
    df,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 512,
):
    """
    Tokenize a pandas DataFrame and return a HuggingFace Dataset.
    
    Args:
        df: pandas DataFrame with text and label columns.
        tokenizer: Fitted tokenizer.
        text_column: Name of the text column.
        label_column: Name of the label column.
        max_length: Maximum token sequence length.
        
    Returns:
        datasets.Dataset ready for the Trainer.
    """
    from datasets import Dataset
    
    dataset = Dataset.from_pandas(df[[text_column, label_column]])
    
    def _tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )
        
    dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=[text_column],
        desc="Tokenizing",
    )
    
    # Rename label column to 'labels' expected by the Trainer
    dataset = dataset.rename_column(label_column, "labels")
    dataset.set_format("torch")
    return dataset