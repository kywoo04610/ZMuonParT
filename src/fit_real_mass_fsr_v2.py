## fit_real_mass_fsr_v2.py
## FSR corrected dimuon mass에 대해 threshold별 Voigt + linear background fit을 수행합니다.
##
## Voigt profile = Breit-Wigner natural width ⊗ Gaussian detector resolution
##
## 실행 예:
## python fit_real_mass_fsr_v2.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --output-csv ../processed/real_mass_voigt_fit_particle_transformer_v4_fsr.csv \
##     --output-dir ../plots/real/voigt_fit_fsr

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile


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
        default=[
            0.10, 0.15, 0.20, 0.25, 0.30,
            0.35, 0.40, 0.45, 0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95,
        ],
    )

    parser.add_argument("--fit-min", type=float, default=80.0)
    parser.add_argument("--fit-max", type=float, default=100.0)
    parser.add_argument("--bins", type=int, default=200)

    parser.add_argument(
        "--output-csv",
        default="../processed/real_mass_voigt_fit_particle_transformer_v4_fsr.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="../plots/real/voigt_fit_fsr",
    )

    return parser.parse_args()


def voigt_signal(m, amplitude, m0, gamma, sigma):
    """
    Voigt signal model.

    Parameters
    ----------
    amplitude:
        Overall signal normalization.

    m0:
        Peak center.

    gamma:
        Lorentzian FWHM parameter.
        In the Z context, this is related to the natural Breit-Wigner width.

    sigma:
        Gaussian detector resolution parameter.
    """

    x = m - m0

    # scipy.special.voigt_profile(x, sigma, gamma_lorentz)
    # The third argument is Lorentzian HWHM, so use gamma / 2.
    profile = voigt_profile(x, sigma, gamma / 2.0)

    return amplitude * profile


def fit_function(m, amplitude, m0, gamma, sigma, slope, intercept):
    return voigt_signal(m, amplitude, m0, gamma, sigma) + slope * m + intercept


def voigt_fwhm_approx(gamma, sigma):
    """
    Approximate Voigt FWHM.

    Olivero-Longbothum approximation:
    FWHM ≈ 0.5346 * gamma + sqrt(0.2166 * gamma^2 + gaussian_fwhm^2)

    where gaussian_fwhm = 2.3548 * sigma.
    """

    gaussian_fwhm = 2.354820045 * sigma

    return (
        0.5346 * gamma
        + np.sqrt(0.2166 * gamma**2 + gaussian_fwhm**2)
    )


def estimate_initial_parameters(bin_centers, counts, bin_width):
    peak_idx = int(np.argmax(counts))
    peak_position = bin_centers[peak_idx]
    peak_height = counts[peak_idx]

    # Area-like amplitude. Since voigt_profile is normalized,
    # amplitude roughly corresponds to signal area times bin width scale.
    amplitude = float(max(np.sum(counts) * bin_width * 0.8, 1.0))

    m0 = float(peak_position)
    gamma = 2.5
    sigma = 1.2

    n_side = max(5, len(counts) // 10)
    left_mean = np.mean(counts[:n_side])
    right_mean = np.mean(counts[-n_side:])

    intercept = float(0.5 * (left_mean + right_mean))
    slope = 0.0

    return [amplitude, m0, gamma, sigma, slope, intercept]


def fit_threshold(score, mass, has_os_pair, threshold, fit_min, fit_max, bins):
    selected = (
        (score > threshold)
        & has_os_pair
        & np.isfinite(mass)
        & (mass >= fit_min)
        & (mass <= fit_max)
    )

    selected_mass = mass[selected]

    counts, edges = np.histogram(
        selected_mass,
        bins=bins,
        range=(fit_min, fit_max),
    )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    sigma_counts = np.sqrt(np.maximum(counts, 1.0))

    p0 = estimate_initial_parameters(bin_centers, counts, bin_width)

    lower_bounds = [
        0.0,      # amplitude
        85.0,     # m0
        0.5,      # gamma
        0.01,     # sigma
        -np.inf,  # slope
        -np.inf,  # intercept
    ]

    upper_bounds = [
        np.inf,   # amplitude
        95.0,     # m0
        10.0,     # gamma
        10.0,     # sigma
        np.inf,   # slope
        np.inf,   # intercept
    ]

    try:
        popt, pcov = curve_fit(
            fit_function,
            bin_centers,
            counts,
            p0=p0,
            sigma=sigma_counts,
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
            maxfev=100000,
        )

        perr = np.sqrt(np.diag(pcov))

        fit_success = True
        error_message = ""

    except Exception as e:
        popt = np.full(6, np.nan)
        perr = np.full(6, np.nan)
        fit_success = False
        error_message = str(e)

    amplitude, m0, gamma, sigma, slope, intercept = popt
    amp_err, m0_err, gamma_err, sigma_err, slope_err, intercept_err = perr

    fwhm_voigt = voigt_fwhm_approx(gamma, sigma)

    return {
        "threshold": threshold,
        "selected_events": int(np.sum(score > threshold)),
        "fit_window_events": int(len(selected_mass)),
        "fit_success": fit_success,
        "error_message": error_message,
        "amplitude": float(amplitude),
        "amplitude_err": float(amp_err),
        "m0": float(m0),
        "m0_err": float(m0_err),
        "gamma": float(gamma),
        "gamma_err": float(gamma_err),
        "sigma": float(sigma),
        "sigma_err": float(sigma_err),
        "fwhm_voigt": float(fwhm_voigt),
        "slope": float(slope),
        "slope_err": float(slope_err),
        "intercept": float(intercept),
        "intercept_err": float(intercept_err),
        "bin_centers": bin_centers,
        "counts": counts,
        "sigma_counts": sigma_counts,
        "popt": popt,
        "fit_min": fit_min,
        "fit_max": fit_max,
        "bin_width": bin_width,
    }


def plot_fit(result, output_dir):
    threshold = result["threshold"]
    bin_centers = result["bin_centers"]
    counts = result["counts"]
    sigma_counts = result["sigma_counts"]
    popt = result["popt"]

    fit_min = result["fit_min"]
    fit_max = result["fit_max"]

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        bin_centers,
        counts,
        yerr=sigma_counts,
        fmt="o",
        markersize=3,
        linewidth=1,
        label="Data",
    )

    x_fit = np.linspace(fit_min, fit_max, 1000)

    if result["fit_success"]:
        y_fit = fit_function(x_fit, *popt)
        y_sig = voigt_signal(x_fit, popt[0], popt[1], popt[2], popt[3])
        y_bkg = popt[4] * x_fit + popt[5]

        plt.plot(x_fit, y_fit, label="Voigt + linear bkg")
        plt.plot(x_fit, y_bkg, linestyle="--", label="Linear bkg")
        plt.plot(x_fit, y_sig, linestyle=":", label="Voigt component")

        text = (
            f"threshold = {threshold:.2f}\n"
            f"m0 = {result['m0']:.3f} ± {result['m0_err']:.3f} GeV\n"
            f"gamma = {result['gamma']:.3f} ± {result['gamma_err']:.3f} GeV\n"
            f"sigma = {result['sigma']:.3f} ± {result['sigma_err']:.3f} GeV\n"
            f"FWHM(Voigt) ≈ {result['fwhm_voigt']:.3f} GeV"
        )

        plt.text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.15),
        )
    else:
        plt.text(
            0.05,
            0.95,
            "Fit failed",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )

    plt.xlabel(r"$m_{\mu\mu}^{\mathrm{FSR}}$ [GeV]")
    plt.ylabel("Events / bin")
    plt.title(f"FSR corrected dimuon mass fit, score > {threshold:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_png = os.path.join(
        output_dir,
        f"voigt_fit_fsr_threshold_{threshold:.2f}.png",
    )
    output_pdf = output_png.replace(".png", ".pdf")

    plt.savefig(output_png)
    plt.savefig(output_pdf)
    plt.close()

    print("Saved:", output_png)
    print("Saved:", output_pdf)


def save_csv(results, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    columns = [
        "threshold",
        "selected_events",
        "fit_window_events",
        "fit_success",
        "m0",
        "m0_err",
        "gamma",
        "gamma_err",
        "sigma",
        "sigma_err",
        "fwhm_voigt",
        "amplitude",
        "amplitude_err",
        "slope",
        "slope_err",
        "intercept",
        "intercept_err",
        "error_message",
    ]

    with open(output_csv, "w") as f:
        f.write(",".join(columns) + "\n")

        for result in results:
            row = [str(result[col]) for col in columns]
            f.write(",".join(row) + "\n")

    print("\nSaved:", output_csv)


def print_summary(results):
    print("\nVoigt fit summary")
    print(
        "threshold | selected | fit events | success | "
        "m0 [GeV] | gamma [GeV] | sigma [GeV] | FWHM_Voigt [GeV]"
    )

    for result in results:
        print(
            f"{result['threshold']:8.2f} | "
            f"{result['selected_events']:8d} | "
            f"{result['fit_window_events']:10d} | "
            f"{str(result['fit_success']):7s} | "
            f"{result['m0']:8.3f} ± {result['m0_err']:.3f} | "
            f"{result['gamma']:8.3f} ± {result['gamma_err']:.3f} | "
            f"{result['sigma']:8.3f} ± {result['sigma_err']:.3f} | "
            f"{result['fwhm_voigt']:8.3f}"
        )


def main():
    args = parse_args()

    data = np.load(args.input)

    score = data["score"]
    mass = data["m_mumu_fsr"]
    has_os_pair = data["has_os_pair"]

    results = []

    for threshold in args.thresholds:
        result = fit_threshold(
            score=score,
            mass=mass,
            has_os_pair=has_os_pair,
            threshold=threshold,
            fit_min=args.fit_min,
            fit_max=args.fit_max,
            bins=args.bins,
        )

        results.append(result)
        plot_fit(result, args.output_dir)

    print_summary(results)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()