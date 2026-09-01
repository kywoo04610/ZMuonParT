## eda.py는 ROOT 파일에서 데이터를 읽어와 EDA를 수행하고, 히스토그램을 그리는 코드입니다.
## 명령어는 python eda.py --sample <sample_name> --max-files <number_of_files> 형태로 실행할 수 있습니다.
import argparse
import glob
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

from config import (
    SAMPLES,
    TREE_NAME,
    STEP_SIZE,
    FEATURES,
    CONTINUOUS_FEATURES,
    BOOLEAN_FEATURES,
    HIST_CONFIG,
)

from truth import has_truth_z_mumu_pair


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA for CMS NanoAOD samples.")
    parser.add_argument(
        "--sample",
        required=True,
        choices=SAMPLES.keys(),
        help="Sample name to analyze.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of ROOT files to process. If omitted, all files are used.",
    )
    return parser.parse_args()


def get_root_files(sample, max_files=None):
    files = sorted(glob.glob(os.path.join(SAMPLES[sample]["data_dir"], "*.root")))

    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found for sample: {sample}")

    if max_files is not None:
        files = files[:max_files]

    return files


def has_opposite_charge_pair(charges):
    pairs = ak.combinations(charges, 2, fields=["q1", "q2"])
    opposite = pairs.q1 * pairs.q2 < 0
    return ak.any(opposite, axis=1)


def print_continuous_statistics(name, values):
    print(f"\n{name}")
    print(" entries:", len(values))
    print(" min:", np.min(values))
    print(" max:", np.max(values))
    print(" mean:", np.mean(values))
    print(" median:", np.percentile(values, 50))
    print(" p90:", np.percentile(values, 90))
    print(" p95:", np.percentile(values, 95))
    print(" p99:", np.percentile(values, 99))
    print(" p99.9:", np.percentile(values, 99.9))


def print_boolean_statistics(name, values):
    n_total = len(values)
    n_true = np.sum(values)
    n_false = n_total - n_true

    print(f"\n{name}")
    print(" total:", n_total)
    print(" true:", n_true)
    print(" false:", n_false)
    print(" true fraction:", n_true / n_total)


def save_histogram(sample, name, values, output_dir):
    config = HIST_CONFIG[name]

    plot_range = config["range"]
    bins = config["bins"]
    xlabel = config["xlabel"]

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins, range=plot_range)
    plt.xlabel(xlabel)
    plt.ylabel("Number of muons")
    plt.title(f"{sample}: {name} after preselection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_{name}_linear.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"hist_{name}_linear.pdf"))
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins, range=plot_range)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Number of muons")
    plt.title(f"{sample}: {name} after preselection (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_{name}_log.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"hist_{name}_log.pdf"))
    plt.close()


def save_boolean_histogram(sample, name, values, output_dir):
    n_true = np.sum(values)
    n_false = len(values) - n_true

    plt.figure(figsize=(5, 5))
    plt.bar(["False", "True"], [n_false, n_true])
    plt.xlabel(name)
    plt.ylabel("Number of muons")
    plt.title(f"{sample}: {name} after preselection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bar_{name}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"bar_{name}.pdf"))
    plt.close()


def analyze_file(root_file, branches, feature_values, selected_nmuons):
    file = uproot.open(root_file)
    events = file[TREE_NAME]

    total_events = 0
    count_nmuon_ge2 = 0
    selected_events = 0
    truth_z_events = 0

    for arrays in events.iterate(branches, step_size=STEP_SIZE):
        total_events += len(arrays["nMuon"])

        mask_nmuon_ge2 = arrays["nMuon"] >= 2
        count_nmuon_ge2 += int(ak.sum(mask_nmuon_ge2))

        mask_opposite_pair = has_opposite_charge_pair(arrays["Muon_charge"])

        truth_z_label = has_truth_z_mumu_pair(arrays)
        truth_z_events += int(ak.sum(mask_opposite_pair & truth_z_label))

        selected = arrays[mask_opposite_pair]
        selected_events += len(selected["nMuon"])

        selected_nmuons.append(ak.to_numpy(selected["nMuon"]))

        for feature in FEATURES:
            flat = ak.flatten(selected[feature])
            feature_values[feature].append(ak.to_numpy(flat))

    return total_events, count_nmuon_ge2, selected_events, truth_z_events


def main():
    args = parse_args()

    sample = args.sample
    root_files = get_root_files(sample, args.max_files)

    output_dir = f"../plots/eda/{sample}"
    os.makedirs(output_dir, exist_ok=True)

    print("Sample:", sample)
    print("Number of ROOT files:", len(root_files))

    branches = [
        "nMuon",
        "Muon_charge",
        "Muon_genPartIdx",
        "GenPart_pdgId",
        "GenPart_genPartIdxMother",
    ] + FEATURES

    total_events_all = 0
    count_nmuon_ge2_all = 0
    selected_events_all = 0
    truth_z_events_all = 0

    selected_nmuons = []
    feature_values = {feature: [] for feature in FEATURES}

    for root_file in root_files:
        print("\nProcessing:", root_file)

        total_events, count_nmuon_ge2, selected_events, truth_z_events = analyze_file(
            root_file=root_file,
            branches=branches,
            feature_values=feature_values,
            selected_nmuons=selected_nmuons,
        )

        total_events_all += total_events
        count_nmuon_ge2_all += count_nmuon_ge2
        selected_events_all += selected_events
        truth_z_events_all += truth_z_events

    print("\nPreselection summary")
    print("Total events:", total_events_all)
    print("nMuon >= 2:", count_nmuon_ge2_all)
    print("Selected events:", selected_events_all)
    print("Truth Z->mumu events after preselection:", truth_z_events_all)

    print("\nEfficiencies")
    print("nMuon >= 2 efficiency:", count_nmuon_ge2_all / total_events_all)
    print("preselection efficiency:", selected_events_all / total_events_all)

    if selected_events_all > 0:
        print(
            "truth Z fraction after preselection:",
            truth_z_events_all / selected_events_all,
        )

    selected_nmuons = np.concatenate(selected_nmuons)

    print("\nSelected nMuon distribution")
    unique, counts = np.unique(selected_nmuons, return_counts=True)

    for n, c in zip(unique, counts):
        print(f"nMuon = {n}: {c}")

    print("Max nMuon:", selected_nmuons.max())

    for feature in CONTINUOUS_FEATURES:
        values = np.concatenate(feature_values[feature])

        print_continuous_statistics(feature, values)
        save_histogram(
            sample=sample,
            name=feature,
            values=values,
            output_dir=output_dir,
        )

    for feature in BOOLEAN_FEATURES:
        values = np.concatenate(feature_values[feature])

        print_boolean_statistics(feature, values)
        save_boolean_histogram(
            sample=sample,
            name=feature,
            values=values,
            output_dir=output_dir,
        )

    print("\nPlots saved to:", output_dir)


if __name__ == "__main__":
    main()