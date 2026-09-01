## train_transformer.py는 학습된 transformer 모델을 학습하는 코드입니다.
## 명령어는 python train_transformer.py --epochs <number_of_epochs> --batch-size <batch_size> --lr <learning_rate> --num-workers <number_of_workers> --output-dir <output_directory> 형태로 실행할 수 있습니다.

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from feature_transform import transform_features


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


class SimpleMuonTransformer(nn.Module):
    def __init__(
        self,
        n_features=6,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, X, mask):
        batch_size = X.size(0)

        h = self.input_proj(X)

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)

        cls_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=mask.device,
        )
        full_mask = torch.cat([cls_mask, mask], dim=1)

        key_padding_mask = ~full_mask

        h = self.encoder(
            h,
            src_key_padding_mask=key_padding_mask,
        )

        cls_output = h[:, 0]

        logits = self.classifier(cls_output).squeeze(-1)
        return logits


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

    model = SimpleMuonTransformer(
        n_features=6,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
    ).to(device)

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
            output_path = os.path.join(args.output_dir, "transformer_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "valid_loss": valid_loss,
                    "epoch": epoch,
                    "model_config": {
                        "n_features": 6,
                        "embed_dim": 128,
                        "num_heads": 4,
                        "num_layers": 2,
                        "ff_dim": 256,
                        "dropout": 0.1,
                    },
                },
                output_path,
            )
            print("Saved best model:", output_path)


if __name__ == "__main__":
    main()