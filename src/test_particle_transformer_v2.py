import torch

from particle_transformer_v2 import ParticleTransformerV2


def main():
    batch_size = 2
    num_muons = 8
    num_features = 6

    X = torch.randn(batch_size, num_muons, num_features)
    X[:, :, 2] = torch.rand(batch_size, num_muons) * 2 * torch.pi - torch.pi

    mask = torch.ones(batch_size, num_muons, dtype=torch.bool)
    mask[1, 5:] = False

    model = ParticleTransformerV2()

    logits = model(X, mask)

    print("logits shape:", logits.shape)
    assert logits.shape == (batch_size,)

    print("ParticleTransformerV2 forward test passed.")


if __name__ == "__main__":
    main()