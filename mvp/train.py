"""PCVRHyFormer MVP training entry point.

Extended from baseline train.py with:
1. MVP model support (Target-Aware, Cross-Domain)
2. MVP trainer support (AMP, LR Schedule, Label Smoothing, Grad Accumulation)
3. Additional CLI arguments for new features

Usage:
    python train.py [--enable_cross_domain] [--enable_target_aware] ...
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

# All files (dataset.py, utils.py, model.py, trainer.py) should be in the
# same directory as this script, making the submission self-contained.
from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(schema, per_position_vocab_sizes):
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args():
    parser = argparse.ArgumentParser(description="PCVRHyFormer MVP Training")

    # Paths
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Data pipeline
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--buffer_batches', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--eval_every_n_steps', type=int, default=3000)
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_queries', type=int, default=2)
    parser.add_argument('--num_hyformer_blocks', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'])
    parser.add_argument('--hidden_mult', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.02)
    parser.add_argument('--seq_top_k', type=int, default=50)
    parser.add_argument('--seq_causal', action='store_true', default=False)
    parser.add_argument('--action_num', type=int, default=1)
    parser.add_argument('--use_time_buckets', action='store_true', default=True)
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false')
    parser.add_argument('--rank_mixer_mode', type=str, default='ffn_only',
                        choices=['full', 'ffn_only', 'none'])
    parser.add_argument('--use_rope', action='store_true', default=True)
    parser.add_argument('--no_rope', dest='use_rope', action='store_false')
    parser.add_argument('--rope_base', type=float, default=10000.0)

    # Loss function
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.1)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Sparse optimizer
    parser.add_argument('--sparse_lr', type=float, default=0.05)
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0)
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1)
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0)

    # Embedding construction
    parser.add_argument('--emb_skip_threshold', type=int, default=1000000)
    parser.add_argument('--seq_id_threshold', type=int, default=10000)

    # NS tokenizer
    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'training', 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups)
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'])
    parser.add_argument('--user_ns_tokens', type=int, default=5)
    parser.add_argument('--item_ns_tokens', type=int, default=2)

    # [MVP NEW] Additional flags
    parser.add_argument('--enable_cross_domain', action='store_true', default=True,
                        help='Enable cross-domain interaction layer')
    parser.add_argument('--no_cross_domain', dest='enable_cross_domain', action='store_false')
    parser.add_argument('--enable_target_aware', action='store_true', default=True,
                        help='Enable target-aware query generation')
    parser.add_argument('--no_target_aware', dest='enable_target_aware', action='store_false')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Enable mixed precision training')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio for LR scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0.01,
                        help='Label smoothing epsilon')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    args = parser.parse_args()

    # Environment variables take precedence
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main():
    args = parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"MVP Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())

    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir, schema_path=schema_path,
        batch_size=args.batch_size, valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio, num_workers=args.num_workers,
        buffer_batches=args.buffer_batches, seed=args.seed,
        seq_max_lens=seq_max_lens)

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
    else:
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        # [MVP NEW]
        "enable_cross_domain": args.enable_cross_domain,
        "enable_target_aware": args.enable_target_aware,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"PCVRHyFormer MVP model created")
    logging.info(f"  enable_cross_domain={args.enable_cross_domain}")
    logging.info(f"  enable_target_aware={args.enable_target_aware}")
    logging.info(f"  Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience, label='model')

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader, valid_loader=valid_loader,
        lr=args.lr, num_epochs=args.num_epochs,
        device=args.device, save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr, sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params, writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        # [MVP NEW]
        use_amp=args.use_amp,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )

    trainer.train()
    writer.close()
    logging.info("MVP Training complete!")


if __name__ == "__main__":
    main()
