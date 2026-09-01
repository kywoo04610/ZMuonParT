## plot_real_mass.py
## 실제 데이터에서 ParT score threshold별 m_mumu 분포를 비교하는 코드입니다.
##
## 실행 예:
## python plot_real_mass.py \
##     --input ../processed/real_scores_particle_transformer_v4.npz \
##     --output ../plots/real/real_mumu_mass_thresholds_v4.png

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../processed/real_scores_particle_transformer_v4.npz",
    )
    parser.add_argument(
        "--output",
        default="../plots/real/real_mumu_mass_thresholds_v4.png",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
    )
    parser.add_argument("--mass-min", type=float, default=50.0)
    parser.add_argument("--mass-max", type=float, default=130.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--log-y", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    data = np.load(args.input)

    score = data["score"]
    m_mumu = data["m_mumu"]
    has_os_pair = data["has_os_pair"]

    mass_window = (
        has_os_pair
        & np.isfinite(m_mumu)
        & (m_mumu >= args.mass_min)
        & (m_mumu <= args.mass_max)
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.figure(figsize=(9, 6))

    bins = np.linspace(args.mass_min, args.mass_max, args.bins + 1)

    print("Threshold summary")
    print("threshold | selected events | in mass window")

    for threshold in args.thresholds:
        selected = mass_window & (score > threshold)
        selected_mass = m_mumu[selected]

        print(
            f"{threshold:8.2f} | "
            f"{np.sum(score > threshold):15d} | "
            f"{len(selected_mass):14d}"
        )

        plt.hist(
            selected_mass,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=f"score > {threshold}",
        )

    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Events")
    plt.title("Real SingleMuon data: dimuon mass after ParT score selection")
    plt.legend()
    plt.grid(alpha=0.3)

    if args.log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(args.output)
    print("\nSaved:", args.output)

    if args.output.endswith(".png"):
        pdf_output = args.output.replace(".png", ".pdf")
        plt.savefig(pdf_output)
        print("Saved:", pdf_output)


if __name__ == "__main__":
    main()