"""PCVRHyFormer MVP Trainer with AMP, LR Scheduling, Label Smoothing, and Gradient Accumulation.

Key improvements over baseline trainer:
1. Mixed Precision Training (AMP) with GradScaler
2. Learning Rate Scheduler: Linear Warmup + Cosine Annealing
3. Label Smoothing for BCE loss
4. Gradient Accumulation for larger effective batch size
5. Better logging with per-step metrics
"""

import os
import glob
import shutil
import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing to min_lr.

    Usage:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        for step in range(total_steps):
            scheduler.step()
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class PCVRHyFormerRankingTrainer:
    """Improved PCVRHyFormer trainer with AMP, LR scheduling, and more."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        # [MVP NEW] Additional parameters
        use_amp: bool = True,
        warmup_ratio: float = 0.05,
        label_smoothing: float = 0.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = schema_path
        self.ns_groups_path = ns_groups_path

        # [MVP NEW] AMP setup
        self.use_amp = use_amp and device.startswith('cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logging.info("Mixed Precision Training (AMP) ENABLED")

        # [MVP NEW] Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        if gradient_accumulation_steps > 1:
            logging.info(f"Gradient Accumulation: {gradient_accumulation_steps} steps "
                         f"(effective batch_size = {gradient_accumulation_steps}x)")

        # [MVP NEW] Label smoothing
        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            logging.info(f"Label Smoothing: {label_smoothing}")

        # Dual optimizer
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, "
                         f"{sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, "
                         f"{dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
            self.dense_optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98))
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98))

        # [MVP NEW] LR Scheduler for dense optimizer
        estimated_steps_per_epoch = len(train_loader)
        total_steps = estimated_steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.dense_scheduler = WarmupCosineScheduler(
            self.dense_optimizer, warmup_steps, total_steps)
        logging.info(f"LR Scheduler: warmup={warmup_steps} steps, "
                     f"total={total_steps} steps, warmup_ratio={warmup_ratio}")

        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reinit_sparse_after_epoch = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}")

    def _build_step_dir_name(self, global_step, is_best=False):
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)
        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True
        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(self, global_step, is_best=False, skip_model_file=False):
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self):
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch):
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(self, total_step, val_auc, val_logloss):
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None or val_auc > old_best + self.early_stopping.delta)
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc, "best_val_logloss": val_logloss})
            return

        best_dir = os.path.join(
            self.save_dir, self._build_step_dir_name(total_step, is_best=True))
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()
        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc, "best_val_logloss": val_logloss})

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path):
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _compute_loss(self, logits, label):
        """Compute loss with optional label smoothing."""
        if self.label_smoothing > 0:
            # Smooth labels: y_smooth = y * (1 - eps) + 0.5 * eps
            label = label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.loss_type == 'focal':
            return sigmoid_focal_loss(logits, label,
                                      alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            return F.binary_cross_entropy_with_logits(logits, label)

    def train(self):
        """Main training loop with AMP, LR scheduling, and gradient accumulation."""
        print("Start training (PCVRHyFormer MVP)")
        self.model.train()
        total_step = 0
        accum_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch, accum_step)
                accum_step += 1

                # Only count as a real step when accumulation is complete
                if accum_step % self.gradient_accumulation_steps == 0:
                    total_step += 1
                    loss_sum += loss

                    # [MVP NEW] Step LR scheduler
                    self.dense_scheduler.step()

                    if self.writer:
                        self.writer.add_scalar('Loss/train', loss, total_step)
                        current_lr = self.dense_scheduler.get_last_lr()[0]
                        self.writer.add_scalar('LR/dense', current_lr, total_step)

                    train_pbar.set_postfix({
                        "loss": f"{loss:.4f}",
                        "lr": f"{self.dense_scheduler.get_last_lr()[0]:.2e}"
                    })

                    # Step-level validation
                    if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                        logging.info(f"Evaluating at step {total_step}")
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()

                        logging.info(f"Step {total_step} Validation | "
                                     f"AUC: {val_auc}, LogLoss: {val_logloss}")

                        if self.writer:
                            self.writer.add_scalar('AUC/valid', val_auc, total_step)
                            self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        self._handle_validation_result(total_step, val_auc, val_logloss)

                        if self.early_stopping.early_stop:
                            logging.info(f"Early stopping at step {total_step}")
                            return

            actual_steps = max(1, total_step)
            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # High-cardinality embedding reinit
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(
                    self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay)
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch):
        seq_domains = device_batch['_seq_domains']
        seq_data = {}
        seq_lens = {}
        seq_time_buckets = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data, seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets)

    def _train_step(self, batch, accum_step=0):
        """Single training step with AMP and gradient accumulation."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        # Only zero grad at the start of accumulation
        if accum_step % self.gradient_accumulation_steps == 0:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)

        # [MVP NEW] AMP forward pass
        with autocast(enabled=self.use_amp):
            logits = self.model(model_input).squeeze(-1)
            loss = self._compute_loss(logits, label)
            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

        # [MVP NEW] AMP backward pass
        self.scaler.scale(loss).backward()

        # Only step optimizers when accumulation is complete
        if (accum_step + 1) % self.gradient_accumulation_steps == 0:
            # Unscale before clip
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm, foreach=False)

            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()

        return loss.item() * (self.gradient_accumulation_steps
                              if self.gradient_accumulation_steps > 1 else 1)

    def evaluate(self, epoch=None):
        """Validation with AMP inference."""
        print("Start Evaluation (PCVRHyFormer MVP) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                # [MVP NEW] AMP inference
                with autocast(enabled=self.use_amp):
                    logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu().float())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(
                valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(self, batch):
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']
        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)
        logits = logits.squeeze(-1)
        return logits, label
