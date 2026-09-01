"""
particle_transformer_v1.py

Particle Transformer V1 for ZMuonParT.

V1 uses pairwise attention bias from:
    - delta_eta
    - delta_phi

Invariant mass is intentionally excluded.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pairwise_features import compute_pairwise_features
from pairwise_attention_bias import PairwiseAttentionBias


class PairwiseSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attention_bias=None):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if attention_bias is not None:
            scores = scores + attention_bias

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, embed_dim)

        return self.out_proj(out)


class ParticleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PairwiseSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attention_bias=None):
        h = self.norm1(x)
        h = self.attn(
            h,
            key_padding_mask=key_padding_mask,
            attention_bias=attention_bias,
        )
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout2(h)

        return x


class ParticleTransformerV1(nn.Module):
    def __init__(
        self,
        n_features=6,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        pairwise_features=("delta_eta", "delta_phi"),
        pairwise_hidden_dim=64,
        use_pairwise_bias=True,
    ):
        super().__init__()

        self.pairwise_features = list(pairwise_features)
        self.use_pairwise_bias = use_pairwise_bias

        self.input_proj = nn.Linear(n_features, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pairwise_bias = PairwiseAttentionBias(
            pairwise_dim=len(self.pairwise_features),
            num_heads=num_heads,
            hidden_dim=pairwise_hidden_dim,
        )

        self.blocks = nn.ModuleList(
            [
                ParticleTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, X, mask):
        batch_size = X.size(0)

        h = self.input_proj(X)

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)

        cls_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=mask.device,
        )
        full_mask = torch.cat([cls_mask, mask], dim=1)
        key_padding_mask = ~full_mask

        seq_len = h.size(1)
        num_muons = X.size(1)
        num_heads = self.pairwise_bias.num_heads

        if self.use_pairwise_bias:
            pairwise = compute_pairwise_features(
                X,
                mask,
                feature_names=self.pairwise_features,
            )

            muon_pair_mask = mask[:, :, None] & mask[:, None, :]
            muon_bias = self.pairwise_bias(pairwise, muon_pair_mask)
        else:
            muon_bias = torch.zeros(
                batch_size,
                num_heads,
                num_muons,
                num_muons,
                dtype=h.dtype,
                device=h.device,
            )

        attention_bias = torch.zeros(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            dtype=h.dtype,
            device=h.device,
        )

        attention_bias[:, :, 1:, 1:] = muon_bias

        for block in self.blocks:
            h = block(
                h,
                key_padding_mask=key_padding_mask,
                attention_bias=attention_bias,
            )

        cls_output = h[:, 0]
        logits = self.classifier(cls_output).squeeze(-1)

        return logits