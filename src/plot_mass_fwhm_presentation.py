## plot_mass_fwhm_presentation.py
## fitting 없이 histogram으로 peak position과 FWHM을 구하고
## 발표용 CMS 스타일 mass plot을 생성합니다.
##
## 실행 예:
## python plot_mass_fwhm_presentation.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --threshold 0.90 \
##     --output ../plots/real/presentation/mass_fwhm_threshold_090.png

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import (
    set_presentation_style,
    save_figure,
    CMS_BLUE,
    CMS_RED,
    CMS_BLACK,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../processed/real_scores_particle_transformer_v4_fsr.npz",
    )

    parser.add_argument("--threshold", type=float, default=0.90)

    parser.add_argument("--mass-min", type=float, default=70.0)
    parser.add_argument("--mass-max", type=float, default=110.0)
    parser.add_argument("--bins", type=int, default=160)

    parser.add_argument(
        "--output",
        default="../plots/real/presentation/mass_fwhm_threshold_090.png",
    )

    return parser.parse_args()


def compute_fwhm(bin_centers, counts):
    peak_idx = int(np.argmax(counts))
    peak_height = counts[peak_idx]
    peak_position = bin_centers[peak_idx]
    half_max = peak_height / 2.0

    left_cross = np.nan
    for i in range(peak_idx, 0, -1):
        if counts[i - 1] <= half_max <= counts[i]:
            x1 = bin_centers[i - 1]
            x2 = bin_centers[i]
            y1 = counts[i - 1]
            y2 = counts[i]

            if y2 != y1:
                left_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                left_cross = x1
            break

    right_cross = np.nan
    for i in range(peak_idx, len(counts) - 1):
        if counts[i] >= half_max >= counts[i + 1]:
            x1 = bin_centers[i]
            x2 = bin_centers[i + 1]
            y1 = counts[i]
            y2 = counts[i + 1]

            if y2 != y1:
                right_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                right_cross = x2
            break

    if np.isfinite(left_cross) and np.isfinite(right_cross):
        fwhm = right_cross - left_cross
    else:
        fwhm = np.nan

    return {
        "peak_position": peak_position,
        "peak_height": peak_height,
        "half_max": half_max,
        "left_cross": left_cross,
        "right_cross": right_cross,
        "fwhm": fwhm,
    }


def add_cms_text(ax):
    ax.text(
        0.04,
        0.94,
        "CMS Open Data",
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
        color=CMS_BLACK,
    )

    ax.text(
        0.04,
        0.88,
        "Run2016H SingleMuon",
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
        va="top",
        color=CMS_BLACK,
    )

    ax.text(
        0.96,
        0.94,
        "Particle Transformer v4",
        transform=ax.transAxes,
        fontsize=13,
        ha="right",
        va="top",
        color=CMS_BLACK,
    )


def main():
    args = parse_args()

    set_presentation_style()

    data = np.load(args.input)

    score = data["score"]
    mass = data["m_mumu_fsr"]
    has_os_pair = data["has_os_pair"]

    selected = (
        (score > args.threshold)
        & has_os_pair
        & np.isfinite(mass)
        & (mass >= args.mass_min)
        & (mass <= args.mass_max)
    )

    selected_mass = mass[selected]

    bins = np.linspace(args.mass_min, args.mass_max, args.bins + 1)
    bin_width = bins[1] - bins[0]

    counts, edges = np.histogram(
        selected_mass,
        bins=bins,
        range=(args.mass_min, args.mass_max),
    )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    yerr = np.sqrt(np.maximum(counts, 1.0))

    result = compute_fwhm(bin_centers, counts)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    ax.errorbar(
        bin_centers,
        counts,
        yerr=yerr,
        fmt="o",
        markersize=4.2,
        color=CMS_BLACK,
        ecolor=CMS_BLACK,
        elinewidth=1.0,
        capsize=1.5,
        label="Data",
    )

    ax.axvline(
        result["peak_position"],
        color=CMS_BLUE,
        linestyle="-",
        linewidth=2.2,
        label="Peak bin center",
    )

    ax.axvline(
        91.1876,
        color=CMS_RED,
        linestyle="--",
        linewidth=2.2,
        label=r"PDG $m_Z$",
    )

    if np.isfinite(result["fwhm"]):
        ax.hlines(
            result["half_max"],
            result["left_cross"],
            result["right_cross"],
            color=CMS_BLUE,
            linewidth=3.0,
            label="FWHM",
        )

        ax.vlines(
            [result["left_cross"], result["right_cross"]],
            ymin=0,
            ymax=result["half_max"],
            color=CMS_BLUE,
            linestyle=":",
            linewidth=2.0,
        )

    ax.set_xlabel(r"$m_{\mu\mu}^{\mathrm{FSR}}$ [GeV]")
    ax.set_ylabel(f"Events / {bin_width:.2f} GeV")
    ax.set_title(r"FSR-corrected dimuon invariant mass")

    ax.set_xlim(args.mass_min, args.mass_max)

    ax.grid(True, alpha=0.22)

    add_cms_text(ax)

    info_text = (
        rf"score $>$ {args.threshold:.2f}" + "\n"
        rf"$N = {len(selected_mass)/1e6:.2f}$M" + "\n"
        rf"peak = {result['peak_position']:.3f} GeV" + "\n"
        rf"FWHM = {result['fwhm']:.3f} GeV"
    )

    ax.text(
        0.05,
        0.62,
        info_text,
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=CMS_BLACK,
            alpha=0.88,
        ),
    )

    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.82),
        fontsize=12,
    )

    fig.subplots_adjust(
        left=0.13,
        bottom=0.15,
        right=0.97,
        top=0.88,
    )

    save_figure(fig, args.output)

    plt.close(fig)

    print("Saved:", args.output)
    print("Selected events:", len(selected_mass))
    print("Peak position:", result["peak_position"])
    print("FWHM:", result["fwhm"])
    print("FWHM left:", result["left_cross"])
    print("FWHM right:", result["right_cross"])


if __name__ == "__main__":
    main()