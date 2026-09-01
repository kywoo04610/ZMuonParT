## check_feature_stats.py는 npz 파일에 저장된 feature들의 통계치를 확인하는 코드입니다.
## 명령어는 python check_feature_stats.py --dataset <dataset_name> 입니다.
import argparse
import os

import numpy as np


DATASET_PATHS = {
    "train": "../processed/train_dataset.npz",
    "valid": "../processed/valid_dataset.npz",
    "test": "../processed/test_dataset.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Check feature percentiles.")
    parser.add_argument("--dataset", default="train", choices=DATASET_PATHS.keys())
    return parser.parse_args()


def main():
    args = parse_args()
    path = DATASET_PATHS[args.dataset]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)

    X = data["X"]
    mask = data["mask"]
    features = data["features"]

    values = X[mask]

    percentiles = [0, 1, 5, 50, 90, 95, 99, 99.9, 99.99, 100]

    print("Dataset:", args.dataset)
    print("Path:", path)
    print("Values shape:", values.shape)

    for i, feature in enumerate(features):
        x = values[:, i]

        print("\n" + "=" * 60)
        print(feature)
        print("mean:", np.mean(x))
        print("std :", np.std(x))

        qs = np.percentile(x, percentiles)

        for p, q in zip(percentiles, qs):
            print(f"p{p}: {q}")


if __name__ == "__main__":
    main()