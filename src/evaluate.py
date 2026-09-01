## evaluate.py는 학습된 모델을 평가하는 코드입니다.
## 명령어는 python evaluate.py --model <model_name> --batch-size <batch_size> --num-workers <number_of_workers> --threshold <threshold_value> 형태로 실행할 수 있습니다.
import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from feature_transform import transform_features
from train import BaselineMuonNet
from train_transformer import SimpleMuonTransformer

from particle_transformer_v1 import ParticleTransformerV1
from particle_transformer_v2 import ParticleTransformerV2
from particle_transformer_v3 import ParticleTransformerV3
from particle_transformer_v4 import ParticleTransformerV4

DATASET_PATH = "../processed/test_dataset.npz"
NORMALIZATION_PATH = "../processed/normalization.npz"

MODEL_PATHS = {
    "baseline": "../models/baseline_best.pt",
    "transformer": "../models/transformer_best.pt",
    "particle_transformer_v1": "../models/particle_transformer_v1_best.pt",
    "particle_transformer_v2": "../models/particle_transformer_v2_best.pt",
    "particle_transformer_v3": "../models/particle_transformer_v3_best.pt",
    "particle_transformer_v4": "../models/particle_transformer_v4_best.pt",
}

MODEL_CLASSES = {
    "baseline": BaselineMuonNet,
    "transformer": SimpleMuonTransformer,
    "particle_transformer_v1": ParticleTransformerV1,
    "particle_transformer_v2": ParticleTransformerV2,
    "particle_transformer_v3": ParticleTransformerV3,
    "particle_transformer_v4": ParticleTransformerV4,
}

class MuonDataset(Dataset):
    def __init__(self, path, norm_path):
        data = np.load(path)
        norm = np.load(norm_path)

        X = data["X"]
        mask = data["mask"]
        y = data["y"]
        features = data["features"]

        if not np.array_equal(features, norm["features"]):
            raise ValueError("Feature mismatch.")

        X = transform_features(X, features)
        X = (X - norm["mean"]) / norm["std"]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


def compute_roc_auc(y_true, y_score):
    order = np.argsort(-y_score)
    y = y_true[order]

    n_signal = np.sum(y == 1)
    n_background = np.sum(y == 0)

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    signal_eff = tps / n_signal
    background_eff = fps / n_background

    signal_eff = np.concatenate([[0.0], signal_eff])
    background_eff = np.concatenate([[0.0], background_eff])

    auc = np.trapezoid(signal_eff, background_eff)

    return background_eff, signal_eff, auc


def print_threshold_scan(y_true, y_score):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\nThreshold scan")
    print("threshold | sig_eff | bkg_eff | bkg_rej | accuracy")

    for threshold in thresholds:
        y_pred = y_score > threshold

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        sig_eff = tp / (tp + fn)
        bkg_eff = fp / (fp + tn)
        bkg_rej = 1.0 / bkg_eff if bkg_eff > 0 else np.inf
        acc = (tp + tn) / len(y_true)

        print(
            f"{threshold:8.2f} | "
            f"{sig_eff:7.4f} | "
            f"{bkg_eff:7.4f} | "
            f"{bkg_rej:7.2f} | "
            f"{acc:8.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="baseline",
        choices=list(MODEL_PATHS.keys()),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def build_model(model_name, checkpoint):
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = MODEL_CLASSES[model_name]

    if model_name == "baseline":
        return model_class(n_features=6)

    config = checkpoint.get("model_config")

    if config is None:
        raise ValueError(
            f"Checkpoint for {model_name} does not contain model_config."
        )

    return model_class(**config)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", args.model)

    dataset = MuonDataset(DATASET_PATH, NORMALIZATION_PATH)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_path = MODEL_PATHS[args.model]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(args.model, checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true_all = []
    y_score_all = []

    with torch.no_grad():
        for X, mask, y in loader:
            X = X.to(device)
            mask = mask.to(device)

            logits = model(X, mask)
            probs = torch.sigmoid(logits)

            y_true_all.append(y.numpy())
            y_score_all.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_all).astype(np.int64)
    y_score = np.concatenate(y_score_all)

    y_pred = y_score > args.threshold

    accuracy = np.mean(y_pred == y_true)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    signal_eff = tp / (tp + fn)
    background_eff = fp / (fp + tn)
    background_rejection = 1.0 / background_eff if background_eff > 0 else np.inf

    bkg_eff_curve, sig_eff_curve, auc = compute_roc_auc(y_true, y_score)

    print("\nEvaluation result")
    print("Total events:", len(y_true))
    print("Accuracy:", accuracy)
    print("AUC:", auc)

    print("\nConfusion matrix")
    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)

    print("\nPhysics-style metrics at threshold", args.threshold)
    print("Signal efficiency:", signal_eff)
    print("Background efficiency:", background_eff)
    print("Background rejection:", background_rejection)

    print_threshold_scan(y_true, y_score)

    output_path = f"../models/{args.model}_test_scores.npz"
    np.savez(
        output_path,
        y_true=y_true,
        y_score=y_score,
        background_eff=bkg_eff_curve,
        signal_eff=sig_eff_curve,
        auc=auc,
    )

    print("\nScores and ROC saved:", output_path)


if __name__ == "__main__":
    main()