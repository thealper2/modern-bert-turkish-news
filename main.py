"""
End-to-end ModernBERT fine-tuning pipeline for Turkish news classification.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TrainingConfig,
    VIZ_DIR,
)
from data.dataset import prepare_data
from evaluation.metrics import compute_metrics, error_analysis, full_evaluation_report
from evaluation.visualizations import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_training_curves,
)
from models.deployment import generate_model_card, push_to_hub, save_best_model
from models.model_builder import (
    get_data_collator,
    load_model,
    load_tokenizer,
    tokenize_dataframe,
)
from training.hyperparameter_search import grid_search, optuna_search
from training.trainer import (
    WeightedTrainer,
    build_trainer,
    build_training_arguments,
)
from utils.logger import get_logger
from utils.seed import detect_precision, get_device, set_seed

logger = get_logger("pipeline")


# Single training run
def train_single(config: TrainingConfig) -> dict:
    """
    Execute a single fine-tuning run and return evaluation metrics.

    Args:
        config: TrainingConfig instance.

    Returns:
        Dict of evaluation metrics on the validation set.
    """
    set_seed(config.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    # Auto-detect mixed precision if not manually set
    if not config.fp16 and not config.bf16:
        precision = detect_precision(device)
        config.fp16 = precision["fp16"]
        config.bf16 = precision["bf16"]

    # Data
    data = prepare_data(
        csv_path=config.csv_path,
        text_column=config.text_column,
        label_column=config.label_column,
        num_labels=config.num_labels,
        lowercase=config.lowercase,
        remove_punctuation=config.remove_punctuation,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        weight_strategy=config.weight_strategy,
    )
    train_df = data["train_df"]
    val_df = data["val_df"]
    class_weights = data["class_weights"]

    # Model & Tokeniser
    tokenizer = load_tokenizer(config.model_name, config.max_seq_length)
    model = load_model(config.model_name, config.num_labels)
    data_collator = get_data_collator(tokenizer)

    # Tokenise datasets
    train_dataset = tokenize_dataframe(
        train_df, tokenizer, config.text_column, config.label_column, config.max_seq_length
    )
    val_dataset = tokenize_dataframe(
        val_df, tokenizer, config.text_column, config.label_column, config.max_seq_length
    )

    # Trainer
    training_args = build_training_arguments(config)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics_fn=compute_metrics,
        class_weights=class_weights,
        label_smoothing=config.label_smoothing_factor,
        early_stopping_patience=config.early_stopping_patience,
    )

    # Train
    logger.info("Starting training …")
    trainer.train()

    # Return validation metrics (used by hyperparameter search)
    eval_results = trainer.evaluate()
    # Strip the 'eval_' prefix for convenience
    return {k.replace("eval_", ""): v for k, v in eval_results.items()}

# Full pipeline
def run_pipeline(
    config: TrainingConfig,
    search_mode: str = "none",  # "none" | "grid" | "optuna"
    n_optuna_trials: int = 20,
    label_names: list[str] | None = None,
    run_error_analysis: bool = True,
) -> None:
    """
    Orchestrate the complete fine-tuning pipeline:
      1. Data preparation & EDA visualisation
      2. (Optional) hyperparameter search
      3. Final model training
      4. Test set evaluation & visualisations
      5. Error analysis
      6. Model saving & optional Hub push

    Args:
        config: TrainingConfig instance.
        search_mode: Hyperparameter search strategy ('none', 'grid', 'optuna').
        n_optuna_trials: Number of Optuna trials (used when search_mode='optuna').
        label_names: Human-readable class names; defaults to str(0..n-1).
        run_error_analysis: Whether to collect and save misclassified examples.
    """
    set_seed(config.seed)

    if label_names is None:
        label_names = [str(i) for i in range(config.num_labels)]

    # 1. Data preparation
    logger.info("=== Step 1: Data Preparation ===")
    data = prepare_data(
        csv_path=config.csv_path,
        text_column=config.text_column,
        label_column=config.label_column,
        num_labels=config.num_labels,
        lowercase=config.lowercase,
        remove_punctuation=config.remove_punctuation,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        weight_strategy=config.weight_strategy,
    )
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]
    class_weights = data["class_weights"]

    plot_class_distribution(data["distribution"])

    # 2. Hyperparameter search (optional)
    if search_mode == "grid":
        logger.info("=== Step 2: Grid Search ===")
        search_results = grid_search(
            base_config=config,
            train_fn=train_single,
            output_dir=EXPERIMENTS_DIR,
        )
        logger.info("Grid search complete. Top result:\n%s", search_results.iloc[0].to_dict())

        # Update config with best hyperparams
        best = search_results.iloc[0]
        for key in ["learning_rate", "per_device_train_batch_size", "weight_decay",
                    "num_train_epochs", "warmup_ratio"]:
            if key in best:
                setattr(config, key, best[key])

    elif search_mode == "optuna":
        logger.info("=== Step 2: Optuna Search ===")
        search_results = optuna_search(
            base_config=config,
            train_fn=train_single,
            n_trials=n_optuna_trials,
            output_dir=EXPERIMENTS_DIR,
        )
        best = search_results.iloc[0]
        for key in ["learning_rate", "per_device_train_batch_size", "weight_decay",
                    "num_train_epochs", "warmup_ratio"]:
            if key in best:
                setattr(config, key, best[key])

    else:
        logger.info("=== Step 2: Skipping hyperparameter search ===")

    # 3. Final training run
    logger.info("=== Step 3: Final Training ===")
    config.output_dir = str(CHECKPOINTS_DIR / "best_model")
    set_seed(config.seed)
    device = get_device()

    if not config.fp16 and not config.bf16:
        precision = detect_precision(device)
        config.fp16 = precision["fp16"]
        config.bf16 = precision["bf16"]

    tokenizer = load_tokenizer(config.model_name, config.max_seq_length)
    model = load_model(config.model_name, config.num_labels)
    data_collator = get_data_collator(tokenizer)

    train_dataset = tokenize_dataframe(train_df, tokenizer, config.text_column, config.label_column, config.max_seq_length)
    val_dataset = tokenize_dataframe(val_df, tokenizer, config.text_column, config.label_column, config.max_seq_length)
    test_dataset = tokenize_dataframe(test_df, tokenizer, config.text_column, config.label_column, config.max_seq_length)

    training_args = build_training_arguments(config)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics_fn=compute_metrics,
        class_weights=class_weights,
        label_smoothing=config.label_smoothing_factor,
        early_stopping_patience=config.early_stopping_patience,
    )

    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # 4. Training visualisations
    logger.info("=== Step 4: Training Visualisations ===")
    plot_training_curves(
        trainer.state.log_history,
        output_dir=VIZ_DIR,
        experiment_name="final_run",
    )

    # 5. Test set evaluation
    logger.info("=== Step 5: Test Set Evaluation ===")
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=-1)
    y_true = test_predictions.label_ids

    report = full_evaluation_report(y_true, y_pred, label_names)

    # Save report
    report_path = REPORTS_DIR / "test_evaluation.json"
    serialisable = {
        "metrics": report["metrics"],
        "per_class": report["per_class"],
        "classification_report": report["classification_report"],
        "confusion_matrix": report["confusion_matrix"].tolist(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation report saved to '%s'", report_path)

    # Visualisations
    plot_confusion_matrix(report["confusion_matrix"], label_names, normalise=True, experiment_name="final_run")
    plot_confusion_matrix(report["confusion_matrix"], label_names, normalise=False, experiment_name="final_run")
    plot_per_class_metrics(report["per_class"], experiment_name="final_run")

    # 6. Error analysis
    if run_error_analysis:
        logger.info("=== Step 6: Error Analysis ===")
        errors = error_analysis(
            texts=test_df[config.text_column].tolist(),
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            label_names=label_names,
        )
        error_path = REPORTS_DIR / "error_analysis.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        logger.info("Error analysis saved to '%s'", error_path)

    # 7. Save best model
    logger.info("=== Step 7: Saving Best Model ===")
    best_model_dir = MODELS_DIR / "best_model"
    save_best_model(trainer.model, tokenizer, best_model_dir)

    # Generate model card
    generate_model_card(
        model_name=config.model_name,
        dataset_name=Path(config.csv_path).stem,
        num_labels=config.num_labels,
        label_names=label_names,
        metrics=report["metrics"],
        per_class=report["per_class"],
        config_dict=asdict(config),
        output_dir=best_model_dir,
    )

    # 8. (Optional) Push to Hub
    if config.push_to_hub and config.hub_model_id:
        logger.info("=== Step 8: Pushing to HuggingFace Hub ===")
        push_to_hub(
            model=trainer.model,
            tokenizer=tokenizer,
            hub_model_id=config.hub_model_id,
            hub_token=config.hub_token,
        )

    logger.info("=== Pipeline complete! ===")
    logger.info("Test F1 (macro): %.4f", report["metrics"]["f1_macro"])
    logger.info("Test Balanced Accuracy: %.4f", report["metrics"]["balanced_accuracy"])


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModernBERT Turkish News Classification Pipeline")
    parser.add_argument("--csv", type=str, default="data/dataset.csv", help="Path to the CSV dataset")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base", help="HuggingFace model name")
    parser.add_argument("--num_labels", type=int, default=9, help="Number of output classes")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum token sequence length")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="LR warmup ratio")
    parser.add_argument("--weight_strategy", type=str, default="sqrt_inverse",
                        choices=["sqrt_inverse", "inverse", "effective_samples", "none"],
                        help="Class weight strategy")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--search", type=str, default="none",
                        choices=["none", "grid", "optuna"], help="Hyperparameter search mode")
    parser.add_argument("--n_trials", type=int, default=20, help="Optuna trial count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--push_to_hub", action="store_true", help="Push best model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub model repository ID")
    parser.add_argument("--hub_token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text during preprocessing")
    parser.add_argument("--label_names", type=str, nargs="*", default=None,
                        help="Human-readable label names (space-separated)")
    args = parser.parse_args()

    cfg = TrainingConfig(
        csv_path=args.csv,
        model_name=args.model,
        num_labels=args.num_labels,
        max_seq_length=args.max_len,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        weight_strategy=args.weight_strategy,
        label_smoothing_factor=args.label_smoothing,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        lowercase=args.lowercase,
    )

    run_pipeline(
        config=cfg,
        search_mode=args.search,
        n_optuna_trials=args.n_trials,
        label_names=args.label_names,
    )
