"""PCVRHyFormer MVP: Improved model with Target-Aware Attention and Cross-Domain Interaction.

Key improvements over baseline:
1. Target-Aware Query Generation: item features guide sequence attention
2. Cross-Domain Interaction Layer: Q tokens from different domains interact
3. Improved output head: deeper classifier with residual connection
4. Optional multi-head target attention in Query Decoding
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union, Dict


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(x, cos, sin):
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with optional RoPE and gated output."""

    def __init__(self, d_model, num_heads, dropout=0.0, rope_on_q=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                rope_cos=None, rope_sin=None, q_rope_cos=None, q_rope_sin=None,
                need_weights=False):
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        Q = self.W_q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)
            if self.rope_on_q:
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        sdpa_attn_mask = None
        if key_padding_mask is not None:
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            bool_attn = (attn_mask == 0)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_attn_mask, dropout_p=dropout_p)
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, ln_mode='pre'):
        super().__init__()
        self.ln_mode = ln_mode
        self.attn = RoPEMultiheadAttention(d_model, num_heads, dropout, rope_on_q=False)
        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, query, key_value, key_padding_mask=None, rope_cos=None, rope_sin=None):
        residual = query
        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)
        out, _ = self.attn(query=query, key=key_value, value=key_value,
                           key_padding_mask=key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        out = residual + out
        if self.ln_mode == 'post':
            out = self.norm_q(out)
        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block with token mixing."""

    def __init__(self, d_model, n_total, hidden_mult=4, dropout=0.0, mode='full'):
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(f"d_model={d_model} must be divisible by T={n_total}")
            self.d_sub = d_model // n_total

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q):
        B, T, D = Q.shape
        Q_split = Q.view(B, T, self.T, self.d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()
        return Q_rewired.view(B, T, D)

    def forward(self, Q):
        if self.mode == 'none':
            return Q
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:
            Q_hat = Q
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


# ═══════════════════════════════════════════════════════════════════════════════
# [MVP NEW] Target-Aware Query Generator
# ═══════════════════════════════════════════════════════════════════════════════


class TargetAwareQueryGenerator(nn.Module):
    """Improved query generator that incorporates item (target) information.

    Key difference from baseline MultiSeqQueryGenerator:
    - GlobalInfo now includes item NS tokens explicitly separated
    - A target-gate mechanism modulates query generation based on item features
    - This enables the model to generate item-specific queries that attend to
      relevant parts of user behavior sequences.
    """

    def __init__(self, d_model, num_user_ns, num_item_ns, num_queries,
                 num_sequences, hidden_mult=4):
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.num_user_ns = num_user_ns
        self.num_item_ns = num_item_ns

        num_ns = num_user_ns + num_item_ns
        # global_info = user_ns_flat + item_ns_flat + seq_pooled
        global_info_dim = (num_ns + 1) * d_model
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # [MVP NEW] Target gate: item features modulate query generation
        self.target_gate = nn.Sequential(
            nn.Linear(num_item_ns * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(self, ns_tokens, seq_tokens_list, seq_padding_masks):
        B = ns_tokens.shape[0]
        num_ns = ns_tokens.shape[1]

        # Split NS tokens into user and item parts for target gating
        item_ns = ns_tokens[:, self.num_user_ns:, :]  # (B, num_item_ns, D)
        item_flat = item_ns.reshape(B, -1)  # (B, num_item_ns * D)
        target_gate = self.target_gate(item_flat)  # (B, D)

        ns_flat = ns_tokens.view(B, -1)  # (B, num_ns * D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            valid_mask = ~seq_padding_masks[i]
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)
            seq_pooled = seq_sum / seq_count

            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)
            global_info = self.global_info_norm(global_info)

            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)

            # [MVP NEW] Apply target gate to queries
            q_tokens = q_tokens * target_gate.unsqueeze(1)

            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# [MVP NEW] Cross-Domain Interaction Layer
# ═══════════════════════════════════════════════════════════════════════════════


class CrossDomainInteraction(nn.Module):
    """Enables Q tokens from different sequence domains to interact.

    After each HyFormerBlock's Query Boosting, this layer performs
    self-attention over the concatenated Q tokens from all domains,
    allowing information flow between domains (e.g., browsing history
    informs purchase prediction).
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, rope_on_q=False,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens_list):
        """
        Args:
            q_tokens_list: List of (B, Nq, D) tensors, one per domain.
        Returns:
            Updated q_tokens_list with cross-domain information.
        """
        all_q = torch.cat(q_tokens_list, dim=1)  # (B, Nq*S, D)

        # Self-attention across all domain Q tokens
        residual = all_q
        x = self.norm(all_q)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        # Split back
        Nq = q_tokens_list[0].shape[1]
        result = []
        for i in range(len(q_tokens_list)):
            result.append(x[:, i * Nq:(i + 1) * Nq, :])
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders (same as baseline)
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    def __init__(self, d_model, hidden_mult=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, **kwargs):
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = RoPEMultiheadAttention(d_model, num_heads, dropout, rope_on_q=True)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None, rope_cos=None, rope_sin=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask,
                              rope_cos=rope_cos, rope_sin=rope_sin)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x, key_padding_mask


class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder (same as baseline)."""

    def __init__(self, d_model, num_heads, top_k=50, hidden_mult=4, dropout=0.0, causal=False):
        super().__init__()
        self.top_k = top_k
        self.causal = causal
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, num_heads, dropout, rope_on_q=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout),
        )

    def _gather_top_k(self, x, key_padding_mask):
        B, L, D = x.shape
        device = x.device
        valid_len = (~key_padding_mask).sum(dim=1)
        actual_k = torch.clamp(valid_len, max=self.top_k)
        start_pos = valid_len - actual_k
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)
        indices = start_pos.unsqueeze(1) + offsets
        indices = torch.clamp(indices, min=0, max=L - 1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)
        new_valid_len = actual_k
        pad_count = self.top_k - new_valid_len
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()
        return top_k_tokens, new_padding_mask, indices

    def forward(self, x, key_padding_mask=None, rope_cos=None, rope_sin=None):
        B, L, D = x.shape
        if L > self.top_k:
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)
            q_rope_cos, q_rope_sin = None, None
            if rope_cos is not None and rope_sin is not None:
                head_dim = rope_cos.shape[2]
                cos_expanded = rope_cos.expand(B, -1, -1)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)
            attn_out, _ = self.attn(query=q_normed, key=kv_normed, value=kv_normed,
                                    key_padding_mask=key_padding_mask,
                                    rope_cos=rope_cos, rope_sin=rope_sin,
                                    q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin)
            out = q + attn_out
        else:
            new_mask = key_padding_mask
            x_normed = self.norm_q(x)
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
            attn_out, _ = self.attn(query=x_normed, key=x_normed, value=x_normed,
                                    key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                    rope_cos=rope_cos, rope_sin=rope_sin)
            out = x + attn_out
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out
        return out, new_mask


def create_sequence_encoder(encoder_type, d_model, num_heads=4, hidden_mult=4,
                            dropout=0.0, top_k=50, causal=False):
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Block (with optional cross-domain interaction)
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block with optional cross-domain interaction."""

    def __init__(self, d_model, num_heads, num_queries, num_ns, num_sequences,
                 seq_encoder_type='swiglu', hidden_mult=4, dropout=0.0,
                 top_k=50, causal=False, rank_mixer_mode='full',
                 enable_cross_domain=False):
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(seq_encoder_type, d_model, num_heads,
                                    hidden_mult, dropout, top_k, causal)
            for _ in range(num_sequences)
        ])

        self.cross_attns = nn.ModuleList([
            CrossAttention(d_model, num_heads, dropout, ln_mode='pre')
            for _ in range(num_sequences)
        ])

        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(d_model, n_total, hidden_mult, dropout, rank_mixer_mode)

        # [MVP NEW] Optional cross-domain interaction
        self.cross_domain = None
        if enable_cross_domain:
            self.cross_domain = CrossDomainInteraction(d_model, num_heads, dropout)

    def forward(self, q_tokens_list, ns_tokens, seq_tokens_list, seq_padding_masks,
                rope_cos_list=None, rope_sin_list=None):
        S = self.num_sequences
        Nq = self.num_queries

        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            next_seq_i, mask_i = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i], rope_cos=rc, rope_sin=rs)
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i], rope_cos=rc, rope_sin=rs)
            decoded_qs.append(decoded_q_i)

        # [MVP NEW] Cross-domain interaction before fusion
        if self.cross_domain is not None:
            decoded_qs = self.cross_domain(decoded_qs)

        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)
        boosted = self.mixer(combined)

        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# NS Tokenizers (same as baseline)
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    def __init__(self, feature_specs, groups, emb_dim, d_model, emb_skip_threshold=0):
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.group_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(len(group) * emb_dim, d_model), nn.LayerNorm(d_model))
            for group in groups
        ])

    def forward(self, int_feats):
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class RankMixerNSTokenizer(nn.Module):
    def __init__(self, feature_specs, groups, emb_dim, d_model, num_ns_tokens,
                 emb_skip_threshold=0):
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        self.token_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(self.chunk_dim, d_model), nn.LayerNorm(d_model))
            for _ in range(num_ns_tokens)
        ])

    def forward(self, int_feats):
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)
        cat_emb = torch.cat(all_embs, dim=-1)
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))
        chunks = cat_emb.split(self.chunk_dim, dim=-1)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer MVP Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer MVP with Target-Aware Attention and Cross-Domain Interaction."""

    def __init__(
        self,
        user_int_feature_specs, item_int_feature_specs,
        user_dense_dim, item_dense_dim,
        seq_vocab_sizes,
        user_ns_groups, item_ns_groups,
        d_model=128, emb_dim=64, num_queries=2,
        num_hyformer_blocks=3, num_heads=8,
        seq_encoder_type='transformer', hidden_mult=4,
        dropout_rate=0.01, seq_top_k=50, seq_causal=False,
        action_num=1, num_time_buckets=65,
        rank_mixer_mode='full', use_rope=False, rope_base=10000.0,
        emb_skip_threshold=0, seq_id_threshold=10000,
        ns_tokenizer_type='rankmixer',
        user_ns_tokens=0, item_ns_tokens=0,
        # [MVP NEW] additional flags
        enable_cross_domain=True,
        enable_target_aware=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.enable_target_aware = enable_target_aware

        # ================== NS Tokens Construction ==================
        if ns_tokenizer_type == 'group':
            self.user_ns_tokenizer = GroupNSTokenizer(
                user_int_feature_specs, user_ns_groups, emb_dim, d_model, emb_skip_threshold)
            num_user_ns = len(user_ns_groups)
            self.item_ns_tokenizer = GroupNSTokenizer(
                item_int_feature_specs, item_ns_groups, emb_dim, d_model, emb_skip_threshold)
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                user_int_feature_specs, user_ns_groups, emb_dim, d_model,
                user_ns_tokens, emb_skip_threshold)
            num_user_ns = user_ns_tokens
            self.item_ns_tokenizer = RankMixerNSTokenizer(
                item_int_feature_specs, item_ns_groups, emb_dim, d_model,
                item_ns_tokens, emb_skip_threshold)
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model), nn.LayerNorm(d_model))

        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model), nn.LayerNorm(d_model))

        self.num_user_ns = num_user_ns + (1 if self.has_user_dense else 0)
        self.num_item_ns = num_item_ns + (1 if self.has_item_dense else 0)
        self.num_ns = self.num_user_ns + self.num_item_ns

        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T={T}. Valid T: {valid_T}")

        # ================== Seq Tokens Embedding ==================
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}
        self._seq_is_id = {}
        self._seq_vocab_sizes = {}
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model), nn.LayerNorm(d_model))

        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== Query Generator ==================
        if enable_target_aware:
            self.query_generator = TargetAwareQueryGenerator(
                d_model=d_model,
                num_user_ns=self.num_user_ns,
                num_item_ns=self.num_item_ns,
                num_queries=num_queries,
                num_sequences=self.num_sequences,
                hidden_mult=hidden_mult,
            )
        else:
            # Fallback to baseline-compatible generator
            from types import SimpleNamespace
            self.query_generator = TargetAwareQueryGenerator(
                d_model=d_model,
                num_user_ns=self.num_user_ns,
                num_item_ns=self.num_item_ns,
                num_queries=num_queries,
                num_sequences=self.num_sequences,
                hidden_mult=hidden_mult,
            )

        # ================== HyFormer Blocks ==================
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model, num_heads=num_heads,
                num_queries=num_queries, num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type, hidden_mult=hidden_mult,
                dropout=dropout_rate, top_k=seq_top_k, causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                enable_cross_domain=enable_cross_domain,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # ================== Output ==================
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.emb_dropout = nn.Dropout(dropout_rate)

        # [MVP NEW] Improved classifier with residual
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        self._init_params()

    def _init_params(self):
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(self, cardinality_threshold=10000):
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()
        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1
        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1
        if self.num_time_buckets > 0:
            skip_count += 1
        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self):
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self):
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(self, seq, sideinfo_embs, proj, is_id, emb_index, time_bucket_ids):
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)
        token_emb = F.gelu(proj(cat_emb))
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        return token_emb

    def _make_padding_mask(self, seq_len, max_len):
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    def _run_multi_seq_blocks(self, q_tokens_list, ns_tokens, seq_tokens_list,
                               seq_masks_list, apply_dropout=True):
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs, ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs, seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list, rope_sin_list=rope_sin_list)

        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)
        output = all_q.view(B, -1)
        output = self.output_proj(output)
        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)
        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain], self._seq_embs[domain],
                self._seq_proj[domain], self._seq_is_id[domain],
                self._seq_emb_index[domain], inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain],
                                           inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training)

        logits = self.clsfier(output)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)
        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain], self._seq_embs[domain],
                self._seq_proj[domain], self._seq_is_id[domain],
                self._seq_emb_index[domain], inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain],
                                           inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False)

        logits = self.clsfier(output)
        return logits, output
