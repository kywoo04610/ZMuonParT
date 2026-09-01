"""
pairwise_attention_bias.py

Pairwise Attention Bias module.

This module converts pairwise muon features e_ij into attention bias B_ij.

Input:
    pairwise_features: (B, N, N, P)

Output:
    attention_bias: (B, num_heads, N, N)

This bias will later be added to the attention score:

    attention_score = QK^T / sqrt(d_head) + attention_bias
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseAttentionBias(nn.Module):
    """
    Convert pairwise features into attention bias.

    Parameters
    ----------
    pairwise_dim:
        Number of pairwise input features.
        For the first experiment, this is 2:
            [delta_eta, delta_phi]

    num_heads:
        Number of attention heads.

    hidden_dim:
        Hidden dimension of the small MLP.
    """

    def __init__(
        self,
        pairwise_dim: int,
        num_heads: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.pairwise_dim = pairwise_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(pairwise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        pairwise_features: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pairwise_features:
            Tensor with shape:

                (batch_size, num_muons, num_muons, pairwise_dim)

        pair_mask:
            Optional boolean tensor with shape:

                (batch_size, num_muons, num_muons)

            True means valid muon pair.
            False means padded pair.

        Returns
        -------
        torch.Tensor
            Attention bias with shape:

                (batch_size, num_heads, num_muons, num_muons)
        """

        if pairwise_features.ndim != 4:
            raise ValueError(
                "pairwise_features must have shape "
                "(batch_size, num_muons, num_muons, pairwise_dim), "
                f"but got {tuple(pairwise_features.shape)}"
            )

        if pairwise_features.size(-1) != self.pairwise_dim:
            raise ValueError(
                f"Expected pairwise_dim={self.pairwise_dim}, "
                f"but got {pairwise_features.size(-1)}"
            )

        bias = self.mlp(pairwise_features)

        # (B, N, N, H) -> (B, H, N, N)
        bias = bias.permute(0, 3, 1, 2).contiguous()

        if pair_mask is not None:
            if pair_mask.shape != pairwise_features.shape[:3]:
                raise ValueError(
                    f"pair_mask shape {tuple(pair_mask.shape)} does not match "
                    f"pairwise feature pair shape {tuple(pairwise_features.shape[:3])}"
                )

            pair_mask = pair_mask.bool()
            bias = bias.masked_fill(~pair_mask[:, None, :, :], 0.0)

        return bias