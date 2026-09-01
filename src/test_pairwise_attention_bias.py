import torch

from pairwise_features import compute_pairwise_features
from pairwise_attention_bias import PairwiseAttentionBias


def main():
    batch_size = 2
    num_muons = 8
    num_features = 6
    num_heads = 4

    x = torch.randn(batch_size, num_muons, num_features)
    x[:, :, 2] = torch.rand(batch_size, num_muons) * 2 * torch.pi - torch.pi

    mask = torch.ones(batch_size, num_muons, dtype=torch.bool)
    mask[1, 5:] = False

    pairwise_features = compute_pairwise_features(
        x,
        mask,
        feature_names=["delta_eta", "delta_phi"],
    )

    pair_mask = mask[:, :, None] & mask[:, None, :]

    bias_layer = PairwiseAttentionBias(
        pairwise_dim=2,
        num_heads=num_heads,
        hidden_dim=64,
    )

    attention_bias = bias_layer(pairwise_features, pair_mask)

    print("pairwise_features shape:", pairwise_features.shape)
    print("attention_bias shape:", attention_bias.shape)

    assert attention_bias.shape == (
        batch_size,
        num_heads,
        num_muons,
        num_muons,
    )

    # Padding pair bias should be zero
    assert torch.all(attention_bias[1, :, 5:, :] == 0)
    assert torch.all(attention_bias[1, :, :, 5:] == 0)

    print("All pairwise attention bias tests passed.")


if __name__ == "__main__":
    main()