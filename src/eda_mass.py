## eda_mass.py는 ROOT 파일에서 dimuon mass를 계산하고, 히스토그램을 그리는 코드입니다.
## 명령어는 python eda_mass.py --sample <sample_name> --max-files <number_of_files> 형태로 실행할 수 있습니다.
import argparse
import glob
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

from config import SAMPLES, TREE_NAME, STEP_SIZE
from candidate import build_opposite_charge_pairs, compute_mumu_mass


def parse_args():
    parser = argparse.ArgumentParser(description="Make dimuon mass histogram.")
    parser.add_argument(
        "--sample",
        required=True,
        choices=SAMPLES.keys(),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
    )
    return parser.parse_args()


def get_root_files(sample, max_files=None):
    data_dir = SAMPLES[sample]["data_dir"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.root")))

    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found in {data_dir}")

    if max_files is not None:
        files = files[:max_files]

    return files


def make_muon_record(arrays):
    return ak.zip(
        {
            "pt": arrays["Muon_pt"],
            "eta": arrays["Muon_eta"],
            "phi": arrays["Muon_phi"],
            "mass": arrays["Muon_mass"],
            "charge": arrays["Muon_charge"],
        }
    )


def save_mass_histogram(sample, masses, output_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(masses, bins=120, range=(0, 200))
    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Number of pairs")
    plt.title(f"{sample}: opposite-charge dimuon mass")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "hist_mumu_mass_linear.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "hist_mumu_mass_linear.pdf"))
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(masses, bins=120, range=(0, 200))
    plt.yscale("log")
    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Number of pairs")
    plt.title(f"{sample}: opposite-charge dimuon mass (log scale)")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "hist_mumu_mass_log.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "hist_mumu_mass_log.pdf"))
    plt.close()


def main():
    args = parse_args()

    sample = args.sample
    root_files = get_root_files(sample, args.max_files)

    output_dir = f"../plots/eda/{sample}"
    os.makedirs(output_dir, exist_ok=True)

    branches = [
        "nMuon",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mass",
        "Muon_charge",
    ]

    all_masses = []
    total_events = 0
    selected_events = 0
    total_pairs = 0

    print("Sample:", sample)
    print("Number of ROOT files:", len(root_files))

    for root_file in root_files:
        print("\nProcessing:", root_file)

        file = uproot.open(root_file)
        events = file[TREE_NAME]

        for arrays in events.iterate(branches, step_size=STEP_SIZE):
            total_events += len(arrays["nMuon"])

            muons = make_muon_record(arrays)

            pairs = build_opposite_charge_pairs(muons)
            masses = compute_mumu_mass(pairs)

            has_pair = ak.num(masses) > 0
            selected_events += int(ak.sum(has_pair))
            total_pairs += int(ak.sum(ak.num(masses)))

            flat_masses = ak.flatten(masses)
            all_masses.append(ak.to_numpy(flat_masses))

    all_masses = np.concatenate(all_masses)

    print("\nDimuon mass summary")
    print("Total events:", total_events)
    print("Events with OS muon pair:", selected_events)
    print("Total OS pairs:", total_pairs)
    print("Mass entries:", len(all_masses))
    print("Mass min:", np.min(all_masses))
    print("Mass max:", np.max(all_masses))
    print("Mass mean:", np.mean(all_masses))
    print("Mass median:", np.percentile(all_masses, 50))
    print("Mass p90:", np.percentile(all_masses, 90))
    print("Mass p99:", np.percentile(all_masses, 99))

    save_mass_histogram(sample, all_masses, output_dir)

    print("\nPlots saved to:", output_dir)


if __name__ == "__main__":
    main()