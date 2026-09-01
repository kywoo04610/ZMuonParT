import torch

from pairwise_features import compute_pairwise_features


def main():
    batch_size = 2
    num_muons = 8
    num_features = 6

    x = torch.randn(batch_size, num_muons, num_features)

    # phi는 대략 [-pi, pi] 범위로 만들어준다
    x[:, :, 2] = torch.rand(batch_size, num_muons) * 2 * torch.pi - torch.pi

    mask = torch.ones(batch_size, num_muons, dtype=torch.bool)

    # 두 번째 event의 마지막 3개 muon은 padding이라고 가정
    mask[1, 5:] = False

    pairwise = compute_pairwise_features(
        x,
        mask,
        feature_names=["delta_eta", "delta_phi"],
    )

    print("pairwise shape:", pairwise.shape)

    assert pairwise.shape == (batch_size, num_muons, num_muons, 2)

    # diagonal check
    diagonal = pairwise[:, torch.arange(num_muons), torch.arange(num_muons), :]
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)

    # padding mask check
    assert torch.all(pairwise[1, 5:, :, :] == 0)
    assert torch.all(pairwise[1, :, 5:, :] == 0)

    print("All pairwise feature tests passed.")


if __name__ == "__main__":
    main()