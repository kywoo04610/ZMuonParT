## plot_real_mass_presentation.py
## 발표용 FSR-corrected dimuon invariant mass 그래프를 생성합니다.
##
## 예시 1: threshold 여러 개 비교
## python plot_real_mass_presentation.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --thresholds 0.10 0.50 0.90 0.95 \
##     --output ../plots/real/presentation/mass_threshold_comparison.png
##
## 예시 2: matched threshold 하나만
## python plot_real_mass_presentation.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --thresholds 0.96786046 \
##     --output ../plots/real/presentation/mass_matched_threshold.png

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import (
    set_presentation_style,
    add_cms_label,
    save_figure,
    CMS_BLUE,
    CMS_RED,
    CMS_BLACK,
    CMS_GRAY,
)


COLORS = [
    CMS_BLACK,
    CMS_BLUE,
    CMS_RED,
    CMS_GRAY,
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../processed/real_scores_particle_transformer_v4_fsr.npz",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.50, 0.90, 0.95],
    )
    parser.add_argument(
        "--output",
        default="../plots/real/presentation/mass_threshold_comparison.png",
    )

    parser.add_argument("--mass-min", type=float, default=60.0)
    parser.add_argument("--mass-max", type=float, default=120.0)
    parser.add_argument("--bins", type=int, default=120)

    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis.",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize histograms to unit area for shape comparison.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_presentation_style()

    data = np.load(args.input)

    score = data["score"]
    mass = data["m_mumu_fsr"]
    has_os_pair = data["has_os_pair"]

    mass_base = (
        has_os_pair
        & np.isfinite(mass)
        & (mass >= args.mass_min)
        & (mass <= args.mass_max)
    )

    bins = np.linspace(args.mass_min, args.mass_max, args.bins + 1)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    for i, threshold in enumerate(args.thresholds):
        selected = mass_base & (score > threshold)
        selected_mass = mass[selected]

        color = COLORS[i % len(COLORS)]

        weights = None
        ylabel = "Events / 0.5 GeV"

        if args.normalize:
            if len(selected_mass) > 0:
                weights = np.ones_like(selected_mass) / len(selected_mass)
            ylabel = "Normalized events"

        label = (
            rf"score $>$ {threshold:.2f}"
            + f"  (N={len(selected_mass)/1e6:.2f}M)"
        )

        ax.hist(
            selected_mass,
            bins=bins,
            histtype="step",
            linewidth=2.8,
            color=color,
            label=label,
            weights=weights,
        )

    ax.axvline(
        91.1876,
        color=CMS_RED,
        linestyle="--",
        linewidth=2.0,
        label=r"PDG $m_Z$",
    )

    ax.set_xlabel(r"$m_{\mu\mu}^{\mathrm{FSR}}$ [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_title(r"FSR-corrected dimuon invariant mass")

    if args.log_y:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.25)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.82),
        fontsize=13,
    )

    add_cms_label(ax, right_label="")

    fig.subplots_adjust(
        left=0.13,
        bottom=0.14,
        right=0.98,
        top=0.90,
    )

    save_figure(fig, args.output)

    plt.close(fig)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()