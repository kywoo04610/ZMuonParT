## pairwise_features.py

"""
pairwise_features.py

Utilities for computing pairwise muon features.

Current default feature set:
    - delta_eta
    - delta_phi

Additional candidate features are implemented but not used by default:
    - delta_r
    - abs_delta_dxy
    - abs_delta_dz

Important:
    Invariant mass is intentionally excluded from training features.
"""

from __future__ import annotations

import torch


# Feature order:
# [pt, eta, phi, isolation, dxy, dz]
PT_IDX = 0
ETA_IDX = 1
PHI_IDX = 2
ISO_IDX = 3
DXY_IDX = 4
DZ_IDX = 5


DEFAULT_PAIRWISE_FEATURES = [
    "delta_eta",
    "delta_phi",
]


AVAILABLE_PAIRWISE_FEATURES = [
    "delta_eta",
    "delta_phi",
    "delta_r",
    "abs_delta_dxy",
    "abs_delta_dz",
]


def wrap_delta_phi(dphi: torch.Tensor) -> torch.Tensor:
    """
    Wrap delta phi into the range [-pi, pi].

    Parameters
    ----------
    dphi:
        Raw phi difference.

    Returns
    -------
    torch.Tensor
        Periodic delta phi.
    """
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))


def compute_pairwise_features(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    feature_names: list[str] | None = None,
) -> torch.Tensor:
    """
    Compute selected pairwise muon features.

    Parameters
    ----------
    x:
        Muon feature tensor with shape:

            (batch_size, num_muons, num_features)

        The assumed feature order is:

            [pt, eta, phi, isolation, dxy, dz]

    mask:
        Boolean tensor with shape:

            (batch_size, num_muons)

        True means a valid muon.
        False means a padded muon.

    feature_names:
        List of pairwise features to compute.

        If None, the default is:

            ["delta_eta", "delta_phi"]

        Available features:

            - "delta_eta"
            - "delta_phi"
            - "delta_r"
            - "abs_delta_dxy"
            - "abs_delta_dz"

    Returns
    -------
    torch.Tensor
        Pairwise feature tensor with shape:

            (batch_size, num_muons, num_muons, num_pairwise_features)

        For example, with the default feature set:

            (B, N, N, 2)

    Notes
    -----
    Invariant mass is not included on purpose.
    It should be used later only for physics validation, not as a training input.
    """

    if x.ndim != 3:
        raise ValueError(
            f"x must have shape (batch_size, num_muons, num_features), "
            f"but got shape {tuple(x.shape)}"
        )

    if x.size(-1) <= DZ_IDX:
        raise ValueError(
            f"x must contain at least 6 features in the order "
            f"[pt, eta, phi, isolation, dxy, dz], but got {x.size(-1)} features"
        )

    if feature_names is None:
        feature_names = DEFAULT_PAIRWISE_FEATURES

    unknown_features = set(feature_names) - set(AVAILABLE_PAIRWISE_FEATURES)
    if unknown_features:
        raise ValueError(
            f"Unknown pairwise feature(s): {sorted(unknown_features)}. "
            f"Available features are: {AVAILABLE_PAIRWISE_FEATURES}"
        )

    if mask is not None:
        if mask.ndim != 2:
            raise ValueError(
                f"mask must have shape (batch_size, num_muons), "
                f"but got shape {tuple(mask.shape)}"
            )

        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match "
                f"x batch/muon shape {tuple(x.shape[:2])}"
            )

        mask = mask.bool()

    eta = x[:, :, ETA_IDX]
    phi = x[:, :, PHI_IDX]
    dxy = x[:, :, DXY_IDX]
    dz = x[:, :, DZ_IDX]

    eta_i = eta[:, :, None]
    eta_j = eta[:, None, :]

    phi_i = phi[:, :, None]
    phi_j = phi[:, None, :]

    dxy_i = dxy[:, :, None]
    dxy_j = dxy[:, None, :]

    dz_i = dz[:, :, None]
    dz_j = dz[:, None, :]

    delta_eta = eta_i - eta_j
    delta_phi = wrap_delta_phi(phi_i - phi_j)
    delta_r = torch.sqrt(delta_eta**2 + delta_phi**2 + 1.0e-12)

    abs_delta_dxy = torch.abs(dxy_i - dxy_j)
    abs_delta_dz = torch.abs(dz_i - dz_j)

    feature_map = {
        "delta_eta": delta_eta,
        "delta_phi": delta_phi,
        "delta_r": delta_r,
        "abs_delta_dxy": abs_delta_dxy,
        "abs_delta_dz": abs_delta_dz,
    }

    pairwise_features = torch.stack(
        [feature_map[name] for name in feature_names],
        dim=-1,
    )

    if mask is not None:
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pairwise_features = pairwise_features * pair_mask[..., None]

    return pairwise_features