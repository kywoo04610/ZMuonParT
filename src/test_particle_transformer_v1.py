import torch

from particle_transformer_v1 import ParticleTransformerV1


def main():
    batch_size = 2
    num_muons = 8
    num_features = 6

    X = torch.randn(batch_size, num_muons, num_features)
    mask = torch.ones(batch_size, num_muons, dtype=torch.bool)
    mask[1, 5:] = False

    model = ParticleTransformerV1(
        n_features=6,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        pairwise_features=("delta_eta", "delta_phi"),
    )

    logits = model(X, mask)

    print("logits shape:", logits.shape)
    assert logits.shape == (batch_size,)

    print("ParticleTransformerV1 forward test passed.")


if __name__ == "__main__":
    main()