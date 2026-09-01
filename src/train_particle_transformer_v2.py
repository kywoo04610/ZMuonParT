## train_particle_transformer_v2.py
## Particle Transformer V2 학습 코드입니다.
## V2 pairwise features: delta_eta, delta_phi, delta_r
## 실행 예:
## python train_particle_transformer_v2.py --epochs 5 --batch-size 4096 --lr 1e-4 --num-workers 4

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from feature_transform import transform_features
from particle_transformer_v2 import ParticleTransformerV2


DATASET_PATHS = {
    "train": "../processed/train_dataset.npz",
    "valid": "../processed/valid_dataset.npz",
}

NORMALIZATION_PATH = "../processed/normalization.npz"


class MuonDataset(Dataset):
    def __init__(self, path, norm_path):
        data = np.load(path)
        norm = np.load(norm_path)

        X = data["X"]
        mask = data["mask"]
        y = data["y"]
        features = data["features"]

        if not np.array_equal(features, norm["features"]):
            raise ValueError("Feature mismatch between dataset and normalization.")

        X = transform_features(X, features)
        X = (X - norm["mean"]) / norm["std"]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="../models")
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, mask, y in loader:
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(X, mask)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            pred = probs > 0.5

            correct += (pred == y.bool()).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.numel()

    return total_loss / total, correct / total


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = MuonDataset(DATASET_PATHS["train"], NORMALIZATION_PATH)
    valid_dataset = MuonDataset(DATASET_PATHS["valid"], NORMALIZATION_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    n_pos = float((train_dataset.y == 1).sum())
    n_neg = float((train_dataset.y == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)

    print("Train signal:", int(n_pos))
    print("Train background:", int(n_neg))
    print("pos_weight:", pos_weight.item())

    model_config = {
        "n_features": 6,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 256,
        "dropout": 0.1,
        "pairwise_features": ("delta_eta", "delta_phi", "delta_r"),
        "pairwise_hidden_dim": 64,
        "use_pairwise_bias": True,
    }

    print("Model: ParticleTransformerV2")
    print("Pairwise features:", model_config["pairwise_features"])

    model = ParticleTransformerV2(**model_config).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_valid_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0

        for X, mask, y in train_loader:
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X, mask)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total += y.numel()

        train_loss = total_loss / total
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.6f} | "
            f"valid loss {valid_loss:.6f} | "
            f"valid acc {valid_acc:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            output_path = os.path.join(
                args.output_dir,
                "particle_transformer_v2_best.pt",
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "valid_loss": valid_loss,
                    "epoch": epoch,
                    "model_config": model_config,
                },
                output_path,
            )

            print("Saved best model:", output_path)


if __name__ == "__main__":
    main()