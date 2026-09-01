## check_normalization.py는 npz 파일에 저장된 feature들의 정규화 상태를 확인하는 코드입니다.
## 명령어는 python check_normalization.py --dataset <dataset_name> 입니다.
import argparse
import os

import numpy as np

from feature_transform import transform_features


DATASET_PATHS = {
    "train": "../processed/train_dataset.npz",
    "valid": "../processed/valid_dataset.npz",
    "test": "../processed/test_dataset.npz",
}

NORMALIZATION_PATH = "../processed/normalization.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check normalized feature distributions."
    )
    parser.add_argument(
        "--dataset",
        default="train",
        choices=DATASET_PATHS.keys(),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = DATASET_PATHS[args.dataset]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not os.path.exists(NORMALIZATION_PATH):
        raise FileNotFoundError(f"Normalization file not found: {NORMALIZATION_PATH}")

    data = np.load(dataset_path)
    norm = np.load(NORMALIZATION_PATH)

    X = data["X"]
    mask = data["mask"]
    features = data["features"]

    if not np.array_equal(features, norm["features"]):
        raise ValueError("Feature order mismatch.")

    print("Dataset:", args.dataset)
    print("Normalization:", NORMALIZATION_PATH)
    print("X shape:", X.shape)
    print("Mask shape:", mask.shape)

    # -------------------------------
    # Feature transform
    # -------------------------------
    X = transform_features(X, features)

    # -------------------------------
    # Mean / std normalization
    # -------------------------------
    mean = norm["mean"]
    std = norm["std"]

    X = (X - mean) / std

    values = X[mask]

    print("Valid muons:", len(values))

    percentiles = [0, 1, 5, 50, 90, 95, 99, 99.9, 99.99, 100]

    for i, feature in enumerate(features):

        x = values[:, i]

        print("\n" + "=" * 60)
        print(feature)
        print("mean :", np.mean(x))
        print("std  :", np.std(x))

        qs = np.percentile(x, percentiles)

        for p, q in zip(percentiles, qs):
            print(f"p{p:<5}: {q}")


if __name__ == "__main__":
    main()