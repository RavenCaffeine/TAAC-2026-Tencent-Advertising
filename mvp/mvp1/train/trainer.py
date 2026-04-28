"""PCVRHyFormer MVP Trainer with AMP, LR Scheduling, Label Smoothing, and Gradient Accumulation."""

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
    """Linear warmup followed by cosine annealing to min_lr."""

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
            scale = self.current_step / max(1, self.warmup_steps)
        else:
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
        self, model, train_loader, valid_loader, lr, num_epochs, device,
        save_dir, early_stopping, loss_type='bce', focal_alpha=0.1,
        focal_gamma=2.0, sparse_lr=0.05, sparse_weight_decay=0.0,
        reinit_sparse_after_epoch=1, reinit_cardinality_threshold=0,
        ckpt_params=None, writer=None, schema_path=None, ns_groups_path=None,
        eval_every_n_steps=0, train_config=None,
        use_amp=True, warmup_ratio=0.05, label_smoothing=0.0,
        gradient_accumulation_steps=1, max_grad_norm=1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = schema_path
        self.ns_groups_path = ns_groups_path

        self.use_amp = use_amp and device.startswith('cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logging.info("Mixed Precision Training (AMP) ENABLED")

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        if gradient_accumulation_steps > 1:
            logging.info(f"Gradient Accumulation: {gradient_accumulation_steps} steps")

        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            logging.info(f"Label Smoothing: {label_smoothing}")

        # Dual optimizer
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            logging.info(f"Sparse: {len(sparse_params)} tensors (Adagrad lr={sparse_lr})")
            logging.info(f"Dense: {len(dense_params)} tensors (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
            self.dense_optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98))
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98))

        # LR Scheduler - account for gradient accumulation
        opt_steps_per_epoch = max(1, len(train_loader) // gradient_accumulation_steps)
        eff_epochs = min(num_epochs, 20)
        total_steps = opt_steps_per_epoch * eff_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.dense_scheduler = WarmupCosineScheduler(
            self.dense_optimizer, warmup_steps, total_steps)
        logging.info(f"LR Scheduler: warmup={warmup_steps}, total={total_steps}, "
                     f"steps_per_epoch={opt_steps_per_epoch}")

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
        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}")

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
            cfg = dict(self.train_config) if ns_groups_copied else self.train_config
            if ns_groups_copied:
                cfg['ns_groups_json'] = os.path.basename(self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg, f, indent=2)

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
        is_likely_new_best = (old_best is None or val_auc > old_best + self.early_stopping.delta)
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
        if self.label_smoothing > 0:
            label = label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        if self.loss_type == 'focal':
            return sigmoid_focal_loss(logits, label,
                                      alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            return F.binary_cross_entropy_with_logits(logits, label)

    def train(self):
        print("Start training (PCVRHyFormer MVP)")
        self.model.train()
        total_step = 0
        accum_step = 0
        global_batch = 0

        for epoch in range(1, self.num_epochs + 1):
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                        dynamic_ncols=True)
            loss_sum = 0.0
            epoch_steps = 0

            for step, batch in pbar:
                loss_val = self._train_step(batch, accum_step)
                accum_step += 1
                global_batch += 1

                # Log every batch for TensorBoard
                if self.writer and not math.isnan(loss_val):
                    self.writer.add_scalar('Loss/train', loss_val, global_batch)

                if math.isnan(loss_val):
                    logging.error(f"NaN loss at batch {global_batch}! Stopping.")
                    return

                if accum_step % self.gradient_accumulation_steps == 0:
                    total_step += 1
                    epoch_steps += 1
                    loss_sum += loss_val

                    self.dense_scheduler.step()
                    cur_lr = self.dense_scheduler.get_last_lr()[0]

                    if self.writer:
                        self.writer.add_scalar('LR/dense', cur_lr, total_step)

                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{cur_lr:.2e}"})

                    if total_step <= 5 or total_step % 100 == 0:
                        logging.info(f"Step {total_step} | loss={loss_val:.4f} | lr={cur_lr:.2e}")

                    if (self.eval_every_n_steps > 0 and
                            total_step % self.eval_every_n_steps == 0):
                        logging.info(f"Evaluating at step {total_step}")
                        val_auc, val_ll = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()
                        logging.info(f"Step {total_step} | AUC={val_auc:.6f} LogLoss={val_ll:.6f}")
                        if self.writer:
                            self.writer.add_scalar('AUC/valid', val_auc, total_step)
                            self.writer.add_scalar('LogLoss/valid', val_ll, total_step)
                        self._handle_validation_result(total_step, val_auc, val_ll)
                        if self.early_stopping.early_stop:
                            logging.info(f"Early stopping at step {total_step}")
                            return

            avg_loss = loss_sum / max(1, epoch_steps)
            logging.info(f"Epoch {epoch} avg_loss={avg_loss:.4f} steps={epoch_steps}")

            val_auc, val_ll = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()
            logging.info(f"Epoch {epoch} | AUC={val_auc:.6f} LogLoss={val_ll:.6f}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_ll, total_step)

            self._handle_validation_result(total_step, val_auc, val_ll)
            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

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
                logging.info(f"Rebuilt Adagrad after epoch {epoch}, restored {restored} params")

    def _make_model_input(self, device_batch):
        seq_domains = device_batch['_seq_domains']
        seq_data, seq_lens, seq_tb = {}, {}, {}
        for d in seq_domains:
            seq_data[d] = device_batch[d]
            seq_lens[d] = device_batch[f'{d}_len']
            B = device_batch[d].shape[0]
            L = device_batch[d].shape[2]
            seq_tb[d] = device_batch.get(
                f'{d}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data, seq_lens=seq_lens, seq_time_buckets=seq_tb)

    def _train_step(self, batch, accum_step=0):
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        if accum_step % self.gradient_accumulation_steps == 0:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)

        with autocast(enabled=self.use_amp):
            logits = self.model(model_input).squeeze(-1)
            loss = self._compute_loss(logits, label)
            raw_loss = loss.item()
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if (accum_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm, foreach=False)
            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()

        return raw_loss

    def evaluate(self, epoch=None):
        print("Start Evaluation (PCVRHyFormer MVP)")
        self.model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for step, batch in tqdm(enumerate(self.valid_loader),
                                    total=len(self.valid_loader)):
                with autocast(enabled=self.use_amp):
                    logits, labels = self._evaluate_step(batch)
                all_logits.append(logits.detach().cpu().float())
                all_labels.append(labels.detach().cpu())

        all_logits_t = torch.cat(all_logits, dim=0)
        all_labels_t = torch.cat(all_labels, dim=0).long()

        probs = torch.sigmoid(all_logits_t).numpy()
        labels_np = all_labels_t.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            logging.warning(f"[Eval] {int(nan_mask.sum())}/{len(probs)} NaN predictions")
            valid = ~nan_mask
            probs = probs[valid]
            labels_np = labels_np[valid]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_mask = ~torch.isnan(all_logits_t)
        vl = all_logits_t[valid_mask]
        vlb = all_labels_t[valid_mask]
        if len(vl) > 0:
            logloss = F.binary_cross_entropy_with_logits(vl, vlb.float()).item()
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
