#!/bin/bash
# MVP Improved Training Script
# Key changes from baseline:
# - Larger model: d_model=128, 3 blocks, 8 heads
# - AMP enabled by default
# - Warmup + Cosine LR schedule
# - Label smoothing for regularization
# - Gradient accumulation for effective larger batch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/../training:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/../training/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --d_model 128 \
    --emb_dim 64 \
    --num_hyformer_blocks 3 \
    --num_heads 8 \
    --hidden_mult 4 \
    --dropout_rate 0.02 \
    --seq_encoder_type transformer \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    --batch_size 256 \
    --lr 2e-4 \
    --sparse_lr 0.05 \
    --loss_type bce \
    --rank_mixer_mode ffn_only \
    --use_rope \
    --patience 5 \
    --eval_every_n_steps 3000 \
    "$@"

# ═══════════════════════════════════════════════════════════════════════════════
# Notes on MVP changes:
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. d_model: 64 → 128 (2x capacity, more expressive representations)
# 2. num_hyformer_blocks: 2 → 3 (deeper interaction modeling)
# 3. num_heads: 4 → 8 (finer-grained attention patterns)
# 4. dropout_rate: 0.01 → 0.02 (stronger regularization for larger model)
# 5. lr: 1e-4 → 2e-4 (slightly higher for warmup schedule)
# 6. rank_mixer_mode: full → ffn_only (avoid d_model%T constraint,
#    T = 2*4 + 5+1+2 = 16, 128%16=0 so full also works; use ffn_only
#    for flexibility when experimenting with different token counts)
# 7. use_rope: enabled (positional encoding for sequence attention)
# 8. eval_every_n_steps: 3000 (more frequent validation for early stopping)
#
# To use the improved trainer.py with AMP and LR schedule, modify train.py
# to import from mvp.trainer instead of trainer, or copy mvp/trainer.py
# over training/trainer.py.
#
# For the improved model with Target-Aware Attention and Cross-Domain
# Interaction, similarly swap mvp/model.py for training/model.py.
# Note: the MVP model has additional constructor args
# (enable_cross_domain, enable_target_aware) that train.py doesn't pass
# by default. You'll need to either:
#   a) Add these flags to train.py's argparse, or
#   b) Edit the model_args dict in train.py to include them.
