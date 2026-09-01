## plot_voigt_fit_presentation.py
## CMS 스타일의 발표용 Voigt fit 그래프를 생성합니다.
##
## 중요:
## scipy.special.voigt_profile은 area=1로 정규화된 PDF이므로,
## histogram count와 비교하기 위해 bin_width를 곱합니다.
##
## 실행 예:
## python plot_voigt_fit_presentation.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --threshold 0.90 \
##     --output ../plots/real/presentation/voigt_fit_threshold_090_cms_style.png

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

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

    parser.add_argument("--fit-min", type=float, default=85.0)
    parser.add_argument("--fit-max", type=float, default=97.0)

    parser.add_argument("--plot-min", type=float, default=70.0)
    parser.add_argument("--plot-max", type=float, default=110.0)

    parser.add_argument("--bins", type=int, default=160)

    parser.add_argument(
        "--output",
        default="../plots/real/presentation/voigt_fit_threshold_090_cms_style.png",
    )

    return parser.parse_args()


def voigt_signal(m, amplitude, m0, gamma, sigma, bin_width):
    x = m - m0

    # voigt_profile is normalized as a probability density.
    # Multiplying by bin_width converts density to expected events/bin.
    profile = voigt_profile(x, sigma, gamma / 2.0)

    return amplitude * profile * bin_width


def fit_function(m, amplitude, m0, gamma, sigma, slope, intercept, bin_width):
    return (
        voigt_signal(m, amplitude, m0, gamma, sigma, bin_width)
        + slope * m
        + intercept
    )


def voigt_fwhm_approx(gamma, sigma):
    gaussian_fwhm = 2.354820045 * sigma

    return (
        0.5346 * gamma
        + np.sqrt(0.2166 * gamma**2 + gaussian_fwhm**2)
    )


def estimate_initial_parameters(bin_centers, counts):
    peak_idx = int(np.argmax(counts))
    peak_position = bin_centers[peak_idx]
    peak_height = counts[peak_idx]

    # Since voigt_signal includes bin_width internally,
    # amplitude roughly corresponds to total signal yield.
    amplitude = float(max(np.sum(counts) * 0.8, 1.0))

    m0 = float(peak_position)
    gamma = 2.5
    sigma = 1.0

    n_side = max(5, len(counts) // 10)
    left_mean = np.mean(counts[:n_side])
    right_mean = np.mean(counts[-n_side:])

    intercept = float(0.5 * (left_mean + right_mean))
    slope = 0.0

    return [amplitude, m0, gamma, sigma, slope, intercept]


def fit_mass(mass, fit_min, fit_max, bins):
    counts, edges = np.histogram(
        mass,
        bins=bins,
        range=(fit_min, fit_max),
    )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    sigma_counts = np.sqrt(np.maximum(counts, 1.0))

    p0 = estimate_initial_parameters(bin_centers, counts)

    lower_bounds = [
        0.0,
        85.0,
        0.5,
        0.01,
        -np.inf,
        -np.inf,
    ]

    upper_bounds = [
        np.inf,
        95.0,
        10.0,
        10.0,
        np.inf,
        np.inf,
    ]

    popt, pcov = curve_fit(
        lambda m, amplitude, m0, gamma, sigma, slope, intercept:
            fit_function(
                m,
                amplitude,
                m0,
                gamma,
                sigma,
                slope,
                intercept,
                bin_width,
            ),
        bin_centers,
        counts,
        p0=p0,
        sigma=sigma_counts,
        absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )

    perr = np.sqrt(np.diag(pcov))

    return popt, perr, bin_width


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
        & (mass >= args.plot_min)
        & (mass <= args.plot_max)
    )

    selected_mass = mass[selected]

    fit_mask = (
        (selected_mass >= args.fit_min)
        & (selected_mass <= args.fit_max)
    )

    fit_mass_values = selected_mass[fit_mask]

    popt, perr, fit_bin_width = fit_mass(
        fit_mass_values,
        args.fit_min,
        args.fit_max,
        args.bins,
    )

    amplitude, m0, gamma, sigma, slope, intercept = popt
    amp_err, m0_err, gamma_err, sigma_err, slope_err, intercept_err = perr

    fwhm_voigt = voigt_fwhm_approx(gamma, sigma)

    bins = np.linspace(args.plot_min, args.plot_max, args.bins + 1)
    plot_bin_width = bins[1] - bins[0]

    counts, edges = np.histogram(
        selected_mass,
        bins=bins,
        range=(args.plot_min, args.plot_max),
    )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    yerr = np.sqrt(np.maximum(counts, 1.0))

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

    x_fit = np.linspace(args.fit_min, args.fit_max, 1200)

    y_fit = fit_function(
        x_fit,
        amplitude,
        m0,
        gamma,
        sigma,
        slope,
        intercept,
        fit_bin_width,
    )

    ax.plot(
        x_fit,
        y_fit,
        color=CMS_BLUE,
        linewidth=3.0,
        label="Voigt fit",
    )

    ax.axvline(
        91.1876,
        color=CMS_RED,
        linestyle="--",
        linewidth=2.2,
        label=r"PDG $m_Z$",
    )

    ax.set_xlabel(r"$m_{\mu\mu}^{\mathrm{FSR}}$ [GeV]")
    ax.set_ylabel(f"Events / {plot_bin_width:.2f} GeV")
    ax.set_title(r"FSR-corrected dimuon invariant mass")

    ax.set_xlim(args.plot_min, args.plot_max)

    ax.grid(True, alpha=0.22)

    add_cms_text(ax)

    info_text = (
        rf"score $>$ {args.threshold:.2f}" + "\n"
        rf"$N = {len(selected_mass)/1e6:.2f}$M" + "\n"
        rf"$m_0 = {m0:.3f}\pm{m0_err:.3f}$ GeV" + "\n"
        rf"FWHM $\approx {fwhm_voigt:.3f}$ GeV"
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
    print("Fit range:", args.fit_min, args.fit_max)
    print("m0:", m0, "+/-", m0_err)
    print("gamma:", gamma, "+/-", gamma_err)
    print("sigma:", sigma, "+/-", sigma_err)
    print("FWHM Voigt:", fwhm_voigt)


if __name__ == "__main__":
    main()