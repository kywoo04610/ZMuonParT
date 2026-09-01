## normalization.py는 npz 파일에 저장된 feature들의 정규화(normalization) 상수를 계산하는 코드입니다.
## 명령어는 python normalization.py --dataset <dataset_name> --output <output_path> 형태로 실행할 수 있습니다.
import argparse
import os

import numpy as np

from feature_transform import transform_features


DATASET_PATH = "../processed/train_dataset.npz"
OUTPUT_PATH = "../processed/normalization.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization constants after feature transform."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    data = np.load(args.dataset)

    X = data["X"]
    mask = data["mask"]
    features = data["features"]

    print("Dataset:", args.dataset)
    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("features:", features)

    X_transformed = transform_features(X, features)
    values = X_transformed[mask]

    mean = values.mean(axis=0)
    std = values.std(axis=0)

    std = np.where(std < 1e-8, 1.0, std)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        features=features,
        transform=np.array("log1p_pt_iso_signedlog1p_dxy_dz"),
    )

    print("\nNormalization saved:", args.output)

    print("\nFeature transform + normalization constants")
    for feature, m, s in zip(features, mean, std):
        print(f"{feature}: mean = {m:.6g}, std = {s:.6g}")


if __name__ == "__main__":
    main()