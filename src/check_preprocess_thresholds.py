## check_preprocess_thresholds.py는 npz 파일에 저장된 feature들의 전처리(preprocessing) 임계값(thresholds)을 확인하는 코드입니다.
## 명령어는 python check_preprocess_thresholds.py --dataset <dataset_name> 입니다.
import argparse
import os

import numpy as np


DATASET_PATHS = {
    "train": "../processed/train_dataset.npz",
    "valid": "../processed/valid_dataset.npz",
    "test": "../processed/test_dataset.npz",
}


THRESHOLDS = {
    "Muon_pt": [200, 500, 1000, 5000, 10000],
    "Muon_eta": [2.4, 2.5, 3.0, 4.0],
    "Muon_pfRelIso04_all": [1, 5, 10, 20, 50],
    "Muon_dxy": [0.05, 0.1, 0.2, 0.5, 1.0],
    "Muon_dz": [0.1, 0.5, 1.0, 5.0, 10.0],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study preprocessing threshold effects on signal/background."
    )
    parser.add_argument("--dataset", default="train", choices=DATASET_PATHS.keys())
    return parser.parse_args()


def affected_event_mask(values, mask, feature, threshold):
    if feature in ["Muon_eta", "Muon_dxy", "Muon_dz"]:
        return np.any(mask & (np.abs(values) > threshold), axis=1)

    return np.any(mask & (values > threshold), axis=1)


def main():
    args = parse_args()
    path = DATASET_PATHS[args.dataset]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)

    X = data["X"]
    mask = data["mask"]
    y = data["y"]
    features = data["features"]

    feature_to_index = {feature: i for i, feature in enumerate(features)}

    n_signal = np.sum(y == 1)
    n_background = np.sum(y == 0)

    print("Dataset:", args.dataset)
    print("Path:", path)
    print("Total events:", len(y))
    print("Signal:", int(n_signal))
    print("Background:", int(n_background))

    for feature, thresholds in THRESHOLDS.items():
        if feature not in feature_to_index:
            continue

        idx = feature_to_index[feature]
        values = X[:, :, idx]

        print("\n" + "=" * 70)
        print(feature)

        for threshold in thresholds:
            affected = affected_event_mask(values, mask, feature, threshold)

            affected_signal = np.sum(affected & (y == 1))
            affected_background = np.sum(affected & (y == 0))

            signal_frac = affected_signal / n_signal if n_signal > 0 else 0
            background_frac = affected_background / n_background if n_background > 0 else 0

            print(
                f"threshold {threshold:>8}: "
                f"signal affected = {affected_signal:>8} ({signal_frac:.6f}), "
                f"background affected = {affected_background:>8} ({background_frac:.6f})"
            )


if __name__ == "__main__":
    main()