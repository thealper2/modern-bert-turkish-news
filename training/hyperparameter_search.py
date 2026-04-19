"""
Hyperparameter search for the fine-tuning pipeline.
Supports grid search (default) and Optuna-based Bayesian search (optional).
All results are persisted to outputs/experiments/.
"""

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config import EXPERIMENTS_DIR, TrainingConfig, HYPERPARAM_GRID
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)

# Experiment result serialization
def save_experiment_results(
    results: list[dict],
    output_dir: str | Path = EXPERIMENTS_DIR,
    filename: str = "hyperparameter_search_results",
) -> tuple[Path, Path]:
    """
    Persist experiment results as both JSON and CSV.
    
    Args:
        results: List of result dicts (one per hyperparameter combination).
        output_dir: Directory to write files.
        filename: Base filename (without extension).
        
    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{filename}.json"
    csv_path = output_dir / f"{filename}.csv"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.info("Experiment results saved -> %s | %s", json_path, csv_path)
    return json_path, csv_path
    
# Grid search
def grid_search(
    base_config: TrainingConfig,
    train_fn: Callable[[TrainingConfig], dict],
    param_grid: Optional[dict] = None,
    output_dir: str | Path = EXPERIMENTS_DIR,
) -> pd.DataFrame:
    """
    Exhaustive grid search over hyperparameter combinations.
    
    The *train_fn* callable must accept a TrainingConfig and return a dict
    containing at least the key eval_f1_macro (or whatever
    base_config.metric_for_best_model specifies).
    
    Args:
        base_config: Base configuration with default values.
        train_fn: Training function that receives a TrainingConfig.
        param_grid: Dict mapping param names -> list of values.
        output_dir: Where to persist results.
        
    Returns:
        DataFrame of all experiment results sorted by the primary metric.
    """
    if param_grid is None:
        param_grid = HYPERPARAM_GRID
        
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    logger.info("Grid search: %d combinations over %s", len(combinations), keys)
    
    results = []
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        logger.info("- Run %d/%d: %s", i, len(combinations), params)
        
        # Build a fresh config with overhidden params
        config_dict = asdict(base_config)
        config_dict.update(params)
        config = TrainingConfig(**config_dict)
        
        # Give each run a unique output directory
        run_name = "_".join(f"{k}={v}" for k, v in params.items())
        config.output_dir = str(Path(output_dir) / "checkpoints" / run_name)
        
        set_seed(config.seed)
        start = time.time()
        
        try:
            metrics = train_fn(config)
        except Exception as exc:
            logger.error("Run %d failed: %s", i, exc, exc_info=True)
            metrics = {"error": str(exc)}
            
        elapsed = time.time() - start
        result = {
            "run_id": i,
            "experiment_name": run_name,
            "elapsed_seconds": round(elapsed, 1),
            **params,
            **metrics,
        }
        results.append(result)
        
    save_experiment_results(results, output_dir)
    df = pd.DataFrame(results)
    metric_col = base_config.metric_for_best_model.replace("eval_", "")
    if metric_col in df.columns:
        df = df.sort.values(metric_col, ascending=not base_config.greater_is_better)
        
    return df
    
# Optuna search (optional dependency)
def optuna_search(
    base_config: TrainingConfig,
    train_fn: Callable[[TrainingConfig], dict],
    n_trials: int = 20,
    output_dir: str | Path = EXPERIMENTS_DIR,
) -> pd.DataFrame:
    """
    Bayesian hyperparameter search using Optuna (optional dependency).
    
    Searches over:
        learning_rate: log-uniform [1e-6, 1e-4]
        batch_size:    categorical [8, 16, 32]
        weight_decay:  uniform [0, 0.1]
        num_epochs:    int [2, 8]
        warmup_ratio:  uniform [0.0, 0.2]
        
    Args:
        base_config: Base training configuration.
        train_fn: Training function returning a metrics dict.
        n_trials: Number of Optuna trials.
        output_dir: Where to persist results.
        
    Returns:
        DataFrame of all trial results sorted by primary metric.
        
    Raises:
        ImportError: If Optuna is not installed.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Install optuna: pip install optuna") from exc
        
    results: list[dict] = []
    metric_col = base_config.metric_for_best_model.replace("eval_", "")
    direction = "maximize" if base_config.greater_is_better else "minimize"
    
    def _objective(trial: "optuna.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorial("per_device_train_batch_size", [8, 16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }
        config_dict = asdict(base_config)
        config_dict.update(params)
        config = TrainingConfig(**config_dict)
        run_name = f"trial_{trial.number}"
        config.output_dir = str(Path(output_dir) / "checkpoints" / run_name)
        set_seed(config.seed)
        
        try:
            metrics = train_fn(config)
        except Exception as exc:
            logger.error("Trial %d failed: %s", trial.number, exc, exc_info=True)
            raise optuna.exceptions.TrialPruned()
            
        results.append({"trial": trial.number, **params, **metrics})
        return metrics.get(metric_col, 0.0)
        
    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)
    
    save_experiment_results(results, output_dir, filename="optuna_results")
    df = pd.DataFrame(results)
    if metric_col in df.columns:
        df = df.sort_values(metric_col, ascending=not base_config.greater_is_better)
        
    logger.info("Best Optuna trial: %s", study.best_params)
    return df