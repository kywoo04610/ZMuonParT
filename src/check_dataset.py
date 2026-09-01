### check_dataset.py는 ParT 모델 학습에 필요한 dataset을 확인하는 코드입니다.
## 명령어는 python check_dataset.py --dataset <dataset_name> --n-events <number_of_events>

import argparse
import os

import numpy as np


DATASET_PATHS = {
    "DYJetsToLL": "../processed/DYJetsToLL_dataset.npz",
    "TTbar": "../processed/TTbar_dataset.npz",
    "train": "../processed/train_dataset.npz",
    "valid": "../processed/valid_dataset.npz",
    "test": "../processed/test_dataset.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Check processed dataset npz file.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_PATHS.keys(),
        help="Dataset name to check.",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=3,
        help="Number of events to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    path = DATASET_PATHS[args.dataset]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path)

    X = data["X"]
    mask = data["mask"]
    y = data["y"]
    features = data["features"]

    print("Dataset:", args.dataset)
    print("Path:", path)

    print("\nKeys:", data.files)

    print("\nShapes")
    print("X:", X.shape, X.dtype)
    print("mask:", mask.shape, mask.dtype)
    print("y:", y.shape, y.dtype)
    print("features:", features)

    print("\nLabel summary")
    print("Total events:", len(y))
    print("Signal y=1:", int(np.sum(y == 1)))
    print("Background y=0:", int(np.sum(y == 0)))
    print("Signal fraction:", float(np.mean(y)))

    print("\nMuon multiplicity from mask")
    n_muons = mask.sum(axis=1)
    unique, counts = np.unique(n_muons, return_counts=True)

    for n, c in zip(unique, counts):
        print(f"nMuon = {n}: {c}")

    print("\nFirst events")
    n_print = min(args.n_events, len(y))

    for i in range(n_print):
        print("\n" + "=" * 60)
        print(f"Event {i}")
        print("label y:", y[i])
        print("mask:", mask[i])
        print("nMuon:", int(mask[i].sum()))

        for imu in range(X.shape[1]):
            if not mask[i, imu]:
                continue

            print(f"\n  Muon {imu}")
            for ifeature, feature in enumerate(features):
                print(f"    {feature}: {X[i, imu, ifeature]}")


if __name__ == "__main__":
    main()