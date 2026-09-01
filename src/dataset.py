## dataset.py는 ROOT 파일에서 데이터를 읽어와 ParT 모델 학습에 필요한 dataset을 만드는 코드입니다.
## 명령어는 python dataset.py --sample <sample_name> --max-files <number_of_files> --max-muons <number_of_muons> 형태로 실행할 수 있습니다.
import argparse
import glob
import os

import awkward as ak
import numpy as np
import uproot

from config import SAMPLES, TREE_NAME, STEP_SIZE
from truth import has_truth_z_mumu_pair


INPUT_FEATURES = [
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_pfRelIso04_all",
    "Muon_dxy",
    "Muon_dz",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build ParT dataset from ROOT files.")
    parser.add_argument("--sample", required=True, choices=SAMPLES.keys())
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-muons", type=int, default=8)
    return parser.parse_args()


def get_root_files(sample, max_files=None):
    data_dir = SAMPLES[sample]["data_dir"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.root")))

    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found in {data_dir}")

    if max_files is not None:
        files = files[:max_files]

    return files


def has_opposite_charge_pair(charges):
    pairs = ak.combinations(charges, 2, fields=["q1", "q2"])
    opposite = pairs.q1 * pairs.q2 < 0
    return ak.any(opposite, axis=1)


def build_chunk(arrays, max_muons):
    # Reco-level preselection
    reco_mask = has_opposite_charge_pair(arrays["Muon_charge"])
    selected = arrays[reco_mask]

    if len(selected["nMuon"]) == 0:
        return None, None, None

    # Truth-based event label
    truth_label = has_truth_z_mumu_pair(selected)
    y = ak.to_numpy(truth_label).astype(np.int64)

    # Sort muons by descending pT
    order = ak.argsort(selected["Muon_pt"], axis=1, ascending=False)

    feature_arrays = []

    for feature in INPUT_FEATURES:
        values = selected[feature][order]
        values = ak.pad_none(values, max_muons, axis=1)
        values = values[:, :max_muons]
        values = ak.fill_none(values, 0)
        feature_arrays.append(ak.to_numpy(values))

    X = np.stack(feature_arrays, axis=-1).astype(np.float32)

    n_muons = ak.num(selected["Muon_pt"])
    n_muons_clipped = np.minimum(ak.to_numpy(n_muons), max_muons)
    mask = np.arange(max_muons)[None, :] < n_muons_clipped[:, None]

    return X, mask.astype(bool), y


def main():
    args = parse_args()

    sample = args.sample
    root_files = get_root_files(sample, args.max_files)

    branches = [
        "nMuon",
        "Muon_charge",
        "Muon_genPartIdx",
        "GenPart_pdgId",
        "GenPart_genPartIdxMother",
    ] + INPUT_FEATURES

    X_all = []
    mask_all = []
    y_all = []

    total_events = 0
    selected_events = 0

    print("Sample:", sample)
    print("Number of ROOT files:", len(root_files))
    print("Max muons:", args.max_muons)

    for root_file in root_files:
        print("\nProcessing:", root_file)

        file = uproot.open(root_file)
        events = file[TREE_NAME]

        for arrays in events.iterate(branches, step_size=STEP_SIZE):
            total_events += len(arrays["nMuon"])

            X, mask, y = build_chunk(arrays, args.max_muons)

            if X is None:
                continue

            selected_events += len(y)

            X_all.append(X)
            mask_all.append(mask)
            y_all.append(y)

    X_all = np.concatenate(X_all)
    mask_all = np.concatenate(mask_all)
    y_all = np.concatenate(y_all)

    output_dir = "../processed"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{sample}_dataset.npz")

    np.savez(
        output_path,
        X=X_all,
        mask=mask_all,
        y=y_all,
        features=np.array(INPUT_FEATURES),
    )

    n_signal = int(np.sum(y_all == 1))
    n_background = int(np.sum(y_all == 0))

    avg_muons = mask_all.sum(axis=1).mean()
    max_muons_observed = mask_all.sum(axis=1).max()

    summary_path = os.path.join(output_dir, f"{sample}_summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"Sample: {sample}\n")
        f.write("Label definition: truth Z->mumu pair exists\n")
        f.write(f"Number of ROOT files: {len(root_files)}\n")
        f.write(f"Total events read: {total_events}\n")
        f.write(f"Reco preselected events: {selected_events}\n")
        f.write(f"Signal events y=1: {n_signal}\n")
        f.write(f"Background events y=0: {n_background}\n")
        f.write(f"Signal fraction: {n_signal / selected_events:.6f}\n")
        f.write(f"X shape: {X_all.shape}\n")
        f.write(f"mask shape: {mask_all.shape}\n")
        f.write(f"y shape: {y_all.shape}\n")
        f.write(f"Max muons used: {args.max_muons}\n")
        f.write(f"Average real muons/event: {avg_muons:.3f}\n")
        f.write(f"Max real muons/event after truncation: {max_muons_observed}\n")
        f.write("\nInput features:\n")

        for feature in INPUT_FEATURES:
            f.write(f"- {feature}\n")

    print("\nDataset saved:", output_path)
    print("Summary saved:", summary_path)
    print("X shape:", X_all.shape)
    print("mask shape:", mask_all.shape)
    print("y shape:", y_all.shape)
    print("Signal y=1:", n_signal)
    print("Background y=0:", n_background)
    print("Signal fraction:", n_signal / selected_events)


if __name__ == "__main__":
    main()