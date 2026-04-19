"""
Custom HuggingFace Trainer that supports:
    - Class-weighted cross-entropy loss
    - Label smoothing
    - Early stopping via EarlyStoppingCallback
    
Also includes an optional custom training loop for advanced users.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from utils.logger import get_logger

logger = get_logger(__name__)

# Weight Loss Trainer
class WeightedTrainer(Trainer):
    """
    Trainer subclass that replaces the default cross-entropy with a
    class-weighted variant and optionally applies label smoothing.
    """
    
    def __init__(
        self,
        *args,
        class_weights: Optional[np.ndarray] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            class_weights: Float array of shape (num_classes,).
            label_smoothing: Smoothing factor in [0, 1).
        """
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        Override loss computation to inject class weights and label smoothing.
        
        Args:
            model: The model being trained.
            inputs: Batch dict from the DataLoader.
            return_outputs: Whether to return model outputs alongside the loss.
            
        Returns:
            Loss tensor or (loss, outputs) tuple if return_outputs is True 
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        device = logits.device
        
        # Build weight tensor
        if self._class_weights is not None:
            weight = torch.tensor(self._class_weights, dtype=torch.float, device=device)
        else:
            weight = None
            
        # Compute loss with optional label smoothing
        loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self._label_smoothing)
        loss = loss_fct(logits, labels.long())
        
        return (loss, outputs) if return_outputs else loss
        
# Callback: log per-epoch metrics
class MetricsLoggerCallback(TrainerCallback):
    """Logs eval metrics to the Python logger after each evaluation step."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            lines = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            logger.info("Step %d - %s", state.global_step, lines)
            
# TrainingArguments factory
def build_training_arguments(config) -> TrainingArguments:
    """
    Build a TrainingArguments instance from a TrainingConfig dataclass.
    
    Args:
        config: TrainingConfig instance.
        
    Returns:
        Configured TrainingArguments.
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,
        logging_dir=config.logging_dir,
        logging_steps=50,
        report_to=config.report_to,
        seed=config.seed,
        label_smoothing_factor=0.0, # handled inside WeightedTrainer
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.push_to_hub else None,
        hub_token=config.hub_token if config.push_to_hub else None,
        dataloader_pin_memory=True,
        dataloader_num_workers=0, # set >0 if not on Windows
        remove_unused_columns=False,
    )
    
# Full Trainer setup
def build_trainer(
    model: PreTrainedModel,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics_fn,
    class_weights: Optional[np.ndarray] = None,
    label_smoothing: float = 0.0,
    early_stopping_patience: int = 3,
) -> WeightedTrainer:
    """
    Assemble a WeightedTrainer with early stopping and metrics logging.
    
    Args:
        model: PreTrained model to fine-tune.
        training_args: TrainingArguments instance.
        eval_datsaet: Tokenized training dataset.
        tokenizer: Tokenized validation dataset.
        tokenizer: Tokenizer (needed for data collation).
        data_collator: DataCollatorWithPadding instance.
        compute_metrics_fn: Callable that accepts EvalPrediction.
        class_weights: Optional per-class weight decay.
        label_smoothing: Label smoothin factor.
        early_stopping_patience: Patience for EarlyStoppingCallback.
    
    Returns:
        Configured WeightedTrainer.
    """
    callbacks = [
        MetricsLoggerCallback(),
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ]
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
    return trainer
    
# Optional custom training loop
def custom_training_loop(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    early_stopping_patience: int = 3,
    compute_metrics_fn=None,
) -> dict:
    """
    Manual training loop as an alternative to the Trainer API.
    Useful when fine-grained control over optimisation is needed.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Target device.
        num_epochs: Maximum training epochs.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Number of linear warmup steps.
        class_weights: Optional class weight tensor.
        label_smoothing: Label smoothing factor.
        early_stopping_patience: Patience for early stopping.
        compute_metrics_fn: Optional metric function (receives logits, labels arrays).

    Returns:
        Dict with 'train_losses', 'val_losses', and 'metrics_history'.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fct = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=label_smoothing,
    )

    history = {"train_losses": [], "val_losses": [], "metrics_history": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state: Optional[dict] = None

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fct(outputs.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        history["train_losses"].append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = loss_fct(outputs.logits, labels)
                epoch_val_loss += loss.item()
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        history["val_losses"].append(avg_val_loss)

        epoch_metrics = {}
        if compute_metrics_fn is not None:
            logits_np = np.concatenate(all_logits)
            labels_np = np.concatenate(all_labels)
            epoch_metrics = compute_metrics_fn(logits_np, labels_np)
            history["metrics_history"].append(epoch_metrics)

        logger.info(
            "Epoch %d/%d | train_loss: %.4f | val_loss: %.4f | %s",
            epoch, num_epochs, avg_train_loss, avg_val_loss,
            " | ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items()),
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history
        