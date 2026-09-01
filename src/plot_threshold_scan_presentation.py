## plot_threshold_scan_presentation.py
## 발표용 threshold scan 그래프를 생성합니다.

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
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../processed/real_mass_summary_particle_transformer_v4_fsr.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="../plots/real/presentation",
    )
    return parser.parse_args()


def load_csv(path):
    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )


def plot_selected_events(data, output_dir):
    fig, ax = plt.subplots()

    ax.plot(
        data["threshold"],
        data["selected_events"] / 1e6,
        marker="o",
        color=CMS_RED,
        label="Selected events",
    )

    ax.set_xlabel("Particle Transformer score threshold")
    ax.set_ylabel("Selected events [million]")
    ax.set_title("Selected events vs threshold")

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    add_cms_label(ax)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "selected_events_vs_threshold.png"))
    plt.close(fig)


def plot_signal_fraction(data, output_dir):
    fig, ax = plt.subplots()

    ax.plot(
        data["threshold"],
        data["signal_window_fraction"],
        marker="o",
        color=CMS_BLUE,
        label=r"Fraction in $80 < m_{\mu\mu}^{FSR} < 100$ GeV",
    )

    ax.set_xlabel("Particle Transformer score threshold")
    ax.set_ylabel("Signal-window fraction")
    ax.set_title("Signal-window fraction vs threshold")
    ax.set_ylim(0.80, 1.00)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    add_cms_label(ax)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "signal_fraction_vs_threshold.png"))
    plt.close(fig)


def plot_fwhm(data, output_dir):
    fig, ax = plt.subplots()

    ax.plot(
        data["threshold"],
        data["fwhm"],
        marker="o",
        color=CMS_RED,
        label="Histogram FWHM",
    )

    ax.set_xlabel("Particle Transformer score threshold")
    ax.set_ylabel("FWHM [GeV]")
    ax.set_title(r"$m_{\mu\mu}^{FSR}$ peak width vs threshold")

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    add_cms_label(ax)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "fwhm_vs_threshold.png"))
    plt.close(fig)


def plot_peak_position(data, output_dir):
    fig, ax = plt.subplots()

    ax.plot(
        data["threshold"],
        data["peak_position"],
        marker="o",
        color=CMS_BLACK,
        label="Peak bin center",
    )

    ax.axhline(
        91.1876,
        linestyle="--",
        color=CMS_RED,
        linewidth=2.0,
        label=r"PDG $m_Z$",
    )

    ax.set_xlabel("Particle Transformer score threshold")
    ax.set_ylabel("Peak position [GeV]")
    ax.set_title(r"$m_{\mu\mu}^{FSR}$ peak position vs threshold")

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    add_cms_label(ax)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "peak_position_vs_threshold.png"))
    plt.close(fig)


def main():
    args = parse_args()

    set_presentation_style()

    data = load_csv(args.input)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_selected_events(data, args.output_dir)
    plot_signal_fraction(data, args.output_dir)
    plot_fwhm(data, args.output_dir)
    plot_peak_position(data, args.output_dir)

    print("Saved presentation plots to:", args.output_dir)


if __name__ == "__main__":
    main()