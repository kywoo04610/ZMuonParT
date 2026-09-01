import torch

from particle_transformer_v3 import ParticleTransformerV3


def main():
    batch_size = 2
    num_muons = 8
    num_features = 6

    X = torch.randn(batch_size, num_muons, num_features)

    # phi를 [-pi, pi] 범위로 설정
    X[:, :, 2] = torch.rand(batch_size, num_muons) * 2 * torch.pi - torch.pi

    mask = torch.ones(batch_size, num_muons, dtype=torch.bool)
    mask[1, 5:] = False

    model = ParticleTransformerV3()

    logits = model(X, mask)

    print("logits shape:", logits.shape)
    assert logits.shape == (batch_size,)

    print("ParticleTransformerV3 forward test passed.")


if __name__ == "__main__":
    main()