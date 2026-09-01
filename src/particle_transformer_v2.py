"""
particle_transformer_v2.py

Particle Transformer V2 for ZMuonParT.

V2 uses pairwise attention bias from:
    - delta_eta
    - delta_phi
    - delta_r

Invariant mass is intentionally excluded.
"""

from particle_transformer_v1 import ParticleTransformerV1


class ParticleTransformerV2(ParticleTransformerV1):
    def __init__(
        self,
        n_features=6,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        pairwise_features=("delta_eta", "delta_phi", "delta_r"),
        pairwise_hidden_dim=64,
        use_pairwise_bias=True,
    ):
        super().__init__(
            n_features=n_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pairwise_features=pairwise_features,
            pairwise_hidden_dim=pairwise_hidden_dim,
            use_pairwise_bias=use_pairwise_bias,
        )