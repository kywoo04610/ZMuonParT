## real_inference_fsr.py
## FSR corrected real dataset에 학습된 Particle Transformer를 적용하여 score를 저장합니다.
##
## 입력:
##   real_singlemuon_dataset_fsr.npz
##
## 출력:
##   real_scores_particle_transformer_v4_fsr.npz
##
## 실행 예:
## python real_inference_fsr.py \
##     --input ../processed/real_singlemuon_dataset_fsr.npz \
##     --normalization ../processed/normalization.npz \
##     --checkpoint ../models/particle_transformer_v4_best.pt \
##     --output ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --batch-size 4096 \
##     --num-workers 4

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from feature_transform import transform_features
from particle_transformer_v4 import ParticleTransformerV4


class RealMuonDatasetFSR(Dataset):
    def __init__(self, input_path, norm_path):
        self.data = np.load(input_path, mmap_mode="r")
        self.norm = np.load(norm_path)

        self.X = self.data["X"]
        self.mask = self.data["mask"]
        self.m_mumu = self.data["m_mumu"]
        self.m_mumu_fsr = self.data["m_mumu_fsr"]
        self.has_os_pair = self.data["has_os_pair"]
        self.features = self.data["features"]

        if not np.array_equal(self.features, self.norm["features"]):
            raise ValueError("Feature mismatch between real dataset and normalization.")

        self.mean = self.norm["mean"]
        self.std = self.norm["std"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        mask = self.mask[idx]

        X = transform_features(X, self.features)
        X = (X - self.mean) / self.std

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            np.float32(self.m_mumu[idx]),
            np.float32(self.m_mumu_fsr[idx]),
            bool(self.has_os_pair[idx]),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../processed/real_singlemuon_dataset_fsr.npz",
    )
    parser.add_argument(
        "--normalization",
        default="../processed/normalization.npz",
    )
    parser.add_argument(
        "--checkpoint",
        default="../models/particle_transformer_v4_best.pt",
    )
    parser.add_argument(
        "--output",
        default="../processed/real_scores_particle_transformer_v4_fsr.npz",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_model(checkpoint, device):
    config = checkpoint["model_config"]
    model = ParticleTransformerV4(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Input:", args.input)
    print("Normalization:", args.normalization)
    print("Checkpoint:", args.checkpoint)
    print("Output:", args.output)

    dataset = RealMuonDatasetFSR(args.input, args.normalization)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint, device)

    n_events = len(dataset)

    scores = np.empty(n_events, dtype=np.float32)
    m_mumu = np.empty(n_events, dtype=np.float32)
    m_mumu_fsr = np.empty(n_events, dtype=np.float32)
    has_os_pair = np.empty(n_events, dtype=bool)

    offset = 0

    with torch.no_grad():
        for batch_idx, (X, mask, mass, mass_fsr, os_pair) in enumerate(
            loader,
            start=1,
        ):
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(X, mask)
            prob = torch.sigmoid(logits)

            batch_size = X.size(0)
            start = offset
            end = offset + batch_size

            scores[start:end] = prob.cpu().numpy()
            m_mumu[start:end] = mass.numpy()
            m_mumu_fsr[start:end] = mass_fsr.numpy()
            has_os_pair[start:end] = os_pair.numpy().astype(bool)

            offset = end

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx:06d} | "
                    f"processed {offset}/{n_events}"
                )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        score=scores,
        m_mumu=m_mumu,
        m_mumu_fsr=m_mumu_fsr,
        has_os_pair=has_os_pair,
    )

    print("\nSaved:", args.output)
    print("Total events:", n_events)
    print("Events with OS pair:", int(has_os_pair.sum()))
    print("Score min:", float(np.min(scores)))
    print("Score max:", float(np.max(scores)))
    print("Score mean:", float(np.mean(scores)))

    valid_fsr = has_os_pair & np.isfinite(m_mumu_fsr)
    print("Events with finite FSR mass:", int(valid_fsr.sum()))


if __name__ == "__main__":
    main()