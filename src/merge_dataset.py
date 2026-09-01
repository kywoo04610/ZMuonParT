## dataset.py를 이용해 각 sample별로 dataset을 만들고, 
## merge_dataset.py를 이용해 train/valid/test split을 수행하는 코드입니다.
## 명령어는 python merge_dataset.py --train-frac <train_fraction> --valid-frac <valid_fraction> --test-frac <test_fraction> --seed <random_seed> 형태로 실행할 수 있습니다.
import argparse
import os

import numpy as np


DATASETS = {
    "DYJetsToLL": "../processed/DYJetsToLL_dataset.npz",
    "TTbar": "../processed/TTbar_dataset.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge datasets using stratified train/valid/test split."
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(path):
    data = np.load(path)
    return data["X"], data["mask"], data["y"], data["features"]


def split_indices(indices, train_frac, valid_frac):
    n = len(indices)

    n_train = int(train_frac * n)
    n_valid = int(valid_frac * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return train_idx, valid_idx, test_idx


def save_split(output_dir, name, X, mask, y, features):
    output_path = os.path.join(output_dir, f"{name}_dataset.npz")

    np.savez(
        output_path,
        X=X,
        mask=mask,
        y=y,
        features=features,
    )

    print(f"\n{name} saved:", output_path)
    print("  X shape:", X.shape)
    print("  mask shape:", mask.shape)
    print("  y shape:", y.shape)
    print("  signal:", int(np.sum(y == 1)))
    print("  background:", int(np.sum(y == 0)))
    print("  signal fraction:", float(np.mean(y)))


def main():
    args = parse_args()

    total_frac = args.train_frac + args.valid_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError("train-frac + valid-frac + test-frac must be 1.")

    X_list = []
    mask_list = []
    y_list = []
    features_ref = None

    print("Loading datasets")

    for sample, path in DATASETS.items():
        print("\nSample:", sample)
        print("Path:", path)

        X, mask, y, features = load_dataset(path)

        print("  X shape:", X.shape)
        print("  mask shape:", mask.shape)
        print("  y shape:", y.shape)
        print("  signal:", int(np.sum(y == 1)))
        print("  background:", int(np.sum(y == 0)))

        if features_ref is None:
            features_ref = features
        else:
            if not np.array_equal(features_ref, features):
                raise ValueError(f"Feature mismatch in {sample}")

        X_list.append(X)
        mask_list.append(mask)
        y_list.append(y)

    print("\nMerging")
    X_all = np.concatenate(X_list, axis=0)
    mask_all = np.concatenate(mask_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    n_total = len(y_all)

    print("Total events:", n_total)
    print("Total signal:", int(np.sum(y_all == 1)))
    print("Total background:", int(np.sum(y_all == 0)))
    print("Total signal fraction:", float(np.mean(y_all)))

    rng = np.random.default_rng(args.seed)

    signal_idx = np.where(y_all == 1)[0]
    background_idx = np.where(y_all == 0)[0]

    rng.shuffle(signal_idx)
    rng.shuffle(background_idx)

    sig_train, sig_valid, sig_test = split_indices(
        signal_idx,
        args.train_frac,
        args.valid_frac,
    )
    bkg_train, bkg_valid, bkg_test = split_indices(
        background_idx,
        args.train_frac,
        args.valid_frac,
    )

    train_idx = np.concatenate([sig_train, bkg_train])
    valid_idx = np.concatenate([sig_valid, bkg_valid])
    test_idx = np.concatenate([sig_test, bkg_test])

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)

    output_dir = "../processed"
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving stratified splits")

    save_split(
        output_dir,
        "train",
        X_all[train_idx],
        mask_all[train_idx],
        y_all[train_idx],
        features_ref,
    )

    save_split(
        output_dir,
        "valid",
        X_all[valid_idx],
        mask_all[valid_idx],
        y_all[valid_idx],
        features_ref,
    )

    save_split(
        output_dir,
        "test",
        X_all[test_idx],
        mask_all[test_idx],
        y_all[test_idx],
        features_ref,
    )

    print("\nMerge finished.")


if __name__ == "__main__":
    main()