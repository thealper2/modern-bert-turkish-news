"""
Data loading, cleaning, preprocessing and stratified splitting for the pipeline.
"""

import re
import string
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger(__name__)

# I/O
def load_csv(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Load a CSV dataset and validate required columns.
    
    Args:
        path: Path to the CSV file.
        text_column: Name of the text column.
        label_column: Name of the label column.
        
    Returns:
        Raw DataFrame.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
        
    logger.info("Loading dataset from '%s'", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
        
    return df
    
# Cleaning
def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    expected_labels: Optional[set[int]] = None,
) -> pd.DataFrame:
    """
    Remove rows with missing values and exact duplicate (text, label) pairs.
    Optionally filter rows whose labels are not in *expected_labels*.
    
    Args:
        df: Raw DataFrame.
        text_column: Column containing text.
        label_column: Column containing integer labels.
        expected_labels: Set of valid label integers; None skips label filtering.
        
    Returns:
        Cleaned DataFrame with reset index.
    """
    n_start = len(df)
    
    # Drop rows with NaN in key columns
    df = df.dropna(subset=[text_column, label_column]).copy()
    logger.info("After dropping NaN rows: %d (removed %d)", len(df), n_start - len(df))
    
    # Coerce labels to int
    df[label_column] = df[label_column].astype(int)
    
    # Filter unexpected labels
    if expected_labels is not None:
        mask = df[label_column].isin(expected_labels)
        removed = (~mask).sum()
        df = df[mask].copy()
        if removed:
            logger.warning("Removed %d rows with unexpected labels", removed)
            
    # Drop exact duplicates on (text, label)
    n_before = len(df)
    df = df.drop_duplicates(subset=[text_column, label_column]).copy()
    logger.info(
        "After deduplication: %d (removed %d duplicates)",
        len(df),
        n_before - len(df),
    )
    
    df = df.reset_index(drop=True)
    return df
    
# Text preprocessing
def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """
    Apply text normalization.
    
    Args:
        text: Raw input string.
        lowercase: Convert to lower case when True.
        remove_punctuation: Strip ASCII punctuation when True.
        collapse_whitespace: Replace multiple whitespace with a single space.
        
    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        text = str(text)
        
    if lowercase:
        text = text.lower()
        
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
        
    return text
    
def apply_preprocessing(
    df: pd.DataFrame,
    text_column: str = "text",
    lowercase: bool = False,
    remove_punctuation: bool = False,
) -> pd.DataFrame:
    """
    Apply text preprocessing to all rows in the text column.
    
    Args:
        df: Input DataFrame.
        text_column: Column to transform.
        lowercase: Pass to preprocess_text.
        remove_punctuation: Pass to preprocess_text.
        
    Returns:
        DataFrame with transformed text column.
    """
    logger.info("Applying text preprocessing (lowercase=%s, rm_punct=%s)", lowercase, remove_punctuation)
    df = df.copy()
    df[text_column] = df[text_column].apply(
        lambda t: preprocess_text(t, lowercase=lowercase, remove_punctuation=remove_punctuation)
    )
    return df
    
# Class imbalance analysis
def analyse_class_distribution(
    df: pd.DataFrame,
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Compute per-class counts, percentages and imbalance ratio.
    
    Args:
        df: DataFrame with label column.
        label_column: Column name.
        
    Returns:
        Summary DataFrame sorted by label.
    """
    counts = df[label_column].value_counts().sort_index()
    summary = pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "pct": (counts.values / counts.sum() * 100).round(2),
    })
    summary["imbalance_ratio"] = (summary["count"].max() / summary["count"]).round(2)
    logger.info("Class distribution:\n%s", summary.to_string(index=False))
    return summary
    
# Stratified split
def stratified_split(
    df: pd.DataFrame,
    label_column: str = "label",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train / validation / test split.
    
    Args:
        df: Cleaned DataFrame.
        label_column: Stratification column.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random state for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
        
    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        
    labels = df[label_column].values
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )
    
    # Second split: val vs test
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_frac),
        stratify=temp_df[label_column].values,
        random_state=seed,
    )
    
    logger.info(
        "Split sizes -> train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
# Class weights
def compute_class_weights(
    labels: list[int] | np.ndarray,
    num_classes: int,
    strategy: str = "sqrt_inverse",
) -> np.ndarray:
    """
    Compute per-class weights to mitigate class imbalance.
    
    Strategies:
        - sqrt_inverse      : w_c = 1 / sqrt(freq_c) (default, moderate)
        - inverse           : w_c = 1 / freq_c       (aggressive)
        - effective_samples : w_c based on effective number of samples (beta=0.9999)
        - none              : uniform weights
        
    Args:
        labels: Array of integer labels.
        num_classes: Total number of classes.
        strategy: Weighting strategy name.
        
    Returns:
        Normalised weight array of shape (num_classes,).
        
    Raises:
        ValueError: If *strategy* is not recognised.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        counts[c] = max((labels == c).sum(), 1) # avoid division by zero
        
    if strategy == "none":
        weights = np.ones(num_classes, dtype=float)
        
    elif strategy == "inverse":
        weights = 1.0 / counts
        
    elif strategy == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    
    elif strategy == "effective_samples":
        beta = 0.9999
        effective = (1.0 - beta ** counts) / (1.0 - beta)
        weights = 1.0 / effective
        
    else:
        raise ValueError(
            f"Unknown weight strategy '{strategy}'. "
            "Choose from: sqrt_inverse, inverse, effective_samples, none."
        )
        
    # Normalize so weigts sum to num_classes
    weights = weights / weights.sum() * num_classes
    logger.info("Class weights (%s): %s", strategy, np.round(weights, 4))
    return weights.astype(np.float32)
    
# Full pipeline helper
def prepare_data(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    num_labels: int = 9,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    weight_strategy: str = "sqrt_inverse",
) -> dict:
    """
    End-to-end data preparation: load -> clean -> preprocess -> split -> weights.
    
    Returns:
        Dict with keys: train_df, val_df, test_df, class_weights, distribution.
    """
    expected_labels = set(range(num_labels))
    
    df = load_csv(csv_path, text_column, label_column)
    df = clean_dataframe(df, text_column, label_column, expected_labels)
    df = apply_preprocessing(df, text_column, lowercase, remove_punctuation)
    
    distribution = analyse_class_distribution(df, label_column)
    
    train_df, val_df, test_df = stratified_split(
        df, label_column, train_ratio, val_ratio, test_ratio, seed
    )
    
    class_weights = compute_class_weights(
        train_df[label_column].values, num_labels, strategy=weight_strategy
    )
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "class_weights": class_weights,
        "distribution": distribution,
        "text_column": text_column,
        "label_column": label_column,
    }