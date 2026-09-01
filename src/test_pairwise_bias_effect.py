import torch

from particle_transformer_v1 import ParticleTransformerV1


def main():
    torch.manual_seed(42)

    batch_size = 2
    num_muons = 8
    num_features = 6

    X = torch.randn(batch_size, num_muons, num_features)

    # phi를 [-pi, pi] 범위로 설정
    X[:, :, 2] = torch.rand(batch_size, num_muons) * 2 * torch.pi - torch.pi

    mask = torch.ones(batch_size, num_muons, dtype=torch.bool)
    mask[1, 5:] = False

    model_without_bias = ParticleTransformerV1(
        use_pairwise_bias=False,
    )

    model_with_bias = ParticleTransformerV1(
        use_pairwise_bias=True,
    )

    # 두 모델의 weight를 동일하게 맞춘다.
    model_with_bias.load_state_dict(model_without_bias.state_dict())

    model_without_bias.eval()
    model_with_bias.eval()

    with torch.no_grad():
        logits_without_bias = model_without_bias(X, mask)
        logits_with_bias = model_with_bias(X, mask)

    print("logits without pairwise bias:")
    print(logits_without_bias)

    print()
    print("logits with pairwise bias:")
    print(logits_with_bias)

    difference = torch.abs(logits_with_bias - logits_without_bias)

    print()
    print("absolute difference:")
    print(difference)

    assert torch.any(difference > 1.0e-6)

    print()
    print("Pairwise bias changes the model output.")
    print("Test passed.")


if __name__ == "__main__":
    main()