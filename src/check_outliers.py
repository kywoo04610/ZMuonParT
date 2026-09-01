## check_outliers.py는 npz 파일에 저장된 feature들의 이상치(outlier) 여부를 확인하는 코드입니다.
## 명령어는 python check_outliers.py --dataset <dataset_name> --feature <feature_name> --threshold <threshold_value> --n-events <number_of_events> 입니다.
import argparse
import os

import numpy as np


DATASET_PATHS = {
    "train": "../processed/train_dataset.npz",
    "valid": "../processed/valid_dataset.npz",
    "test": "../processed/test_dataset.npz",
}


DEFAULT_THRESHOLDS = {
    "Muon_pt": 1000.0,
    "Muon_eta": 3.0,
    "Muon_phi": 3.2,
    "Muon_pfRelIso04_all": 10.0,
    "Muon_dxy": 1.0,
    "Muon_dz": 10.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect outlier muons in processed datasets.")
    parser.add_argument("--dataset", default="train", choices=DATASET_PATHS.keys())
    parser.add_argument("--feature", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n-events", type=int, default=10)
    return parser.parse_args()


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

    if args.feature not in feature_to_index:
        raise ValueError(f"Unknown feature: {args.feature}. Available: {list(features)}")

    threshold = args.threshold
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS.get(args.feature)

    if threshold is None:
        raise ValueError("No threshold provided and no default threshold exists.")

    feature_idx = feature_to_index[args.feature]
    values = X[:, :, feature_idx]

    if args.feature in ["Muon_eta", "Muon_phi", "Muon_dxy", "Muon_dz"]:
        outlier_muon_mask = mask & (np.abs(values) > threshold)
    else:
        outlier_muon_mask = mask & (values > threshold)

    outlier_event_mask = np.any(outlier_muon_mask, axis=1)
    outlier_event_indices = np.where(outlier_event_mask)[0]

    print("Dataset:", args.dataset)
    print("Path:", path)
    print("Feature:", args.feature)
    print("Threshold:", threshold)
    print("Total events:", len(y))
    print("Outlier events:", len(outlier_event_indices))
    print("Outlier event fraction:", len(outlier_event_indices) / len(y))

    print("\nOutlier label summary")
    if len(outlier_event_indices) > 0:
        y_out = y[outlier_event_indices]
        print("Signal y=1:", int(np.sum(y_out == 1)))
        print("Background y=0:", int(np.sum(y_out == 0)))
        print("Signal fraction:", float(np.mean(y_out)))

    n_print = min(args.n_events, len(outlier_event_indices))

    for k in range(n_print):
        iev = outlier_event_indices[k]

        print("\n" + "=" * 70)
        print(f"Event index: {iev}")
        print("label y:", y[iev])
        print("mask:", mask[iev])
        print("nMuon:", int(mask[iev].sum()))

        for imu in range(X.shape[1]):
            if not mask[iev, imu]:
                continue

            is_outlier = outlier_muon_mask[iev, imu]
            tag = " <-- OUTLIER" if is_outlier else ""

            print(f"\n  Muon {imu}{tag}")
            for ifeature, feature in enumerate(features):
                print(f"    {feature}: {X[iev, imu, ifeature]}")


if __name__ == "__main__":
    main()