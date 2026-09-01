## analysis_real_mass_fsr.py
## FSR corrected dimuon mass를 이용해 ParT threshold별 Z peak를 정량 분석합니다.

import argparse
import os

import numpy as np


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
    parser.add_argument("--mass-min", type=float, default=50.0)
    parser.add_argument("--mass-max", type=float, default=130.0)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--signal-min", type=float, default=80.0)
    parser.add_argument("--signal-max", type=float, default=100.0)
    parser.add_argument(
        "--output",
        default="../processed/real_mass_summary_particle_transformer_v4_fsr.csv",
    )
    return parser.parse_args()


def compute_fwhm(bin_centers, counts):
    if len(counts) == 0 or np.max(counts) <= 0:
        return np.nan, np.nan, np.nan

    peak_idx = int(np.argmax(counts))
    half_max = counts[peak_idx] / 2.0

    left_idx = None
    for i in range(peak_idx, 0, -1):
        if counts[i - 1] < half_max <= counts[i]:
            left_idx = i
            break

    right_idx = None
    for i in range(peak_idx, len(counts) - 1):
        if counts[i] >= half_max > counts[i + 1]:
            right_idx = i
            break

    if left_idx is None or right_idx is None:
        return np.nan, np.nan, np.nan

    x1 = bin_centers[left_idx - 1]
    x2 = bin_centers[left_idx]
    y1 = counts[left_idx - 1]
    y2 = counts[left_idx]

    if y2 == y1:
        left = x2
    else:
        left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    x1 = bin_centers[right_idx]
    x2 = bin_centers[right_idx + 1]
    y1 = counts[right_idx]
    y2 = counts[right_idx + 1]

    if y2 == y1:
        right = x1
    else:
        right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    return right - left, left, right


def analyze_threshold(
    score,
    mass,
    has_os_pair,
    threshold,
    mass_min,
    mass_max,
    bins,
    signal_min,
    signal_max,
):
    selected = score > threshold

    valid_mass = (
        selected
        & has_os_pair
        & np.isfinite(mass)
        & (mass >= mass_min)
        & (mass <= mass_max)
    )

    selected_mass = mass[valid_mass]

    counts, edges = np.histogram(
        selected_mass,
        bins=bins,
        range=(mass_min, mass_max),
    )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    if len(selected_mass) == 0 or np.max(counts) == 0:
        return {
            "threshold": threshold,
            "selected_events": int(np.sum(selected)),
            "mass_window_events": 0,
            "peak_position": np.nan,
            "peak_height": 0,
            "fwhm": np.nan,
            "fwhm_left": np.nan,
            "fwhm_right": np.nan,
            "signal_window_events": 0,
            "signal_window_fraction": np.nan,
        }

    peak_idx = int(np.argmax(counts))
    peak_position = bin_centers[peak_idx]
    peak_height = counts[peak_idx]

    fwhm, fwhm_left, fwhm_right = compute_fwhm(bin_centers, counts)

    signal_window = (
        valid_mass
        & (mass >= signal_min)
        & (mass <= signal_max)
    )

    signal_window_events = int(np.sum(signal_window))
    mass_window_events = int(len(selected_mass))

    signal_window_fraction = (
        signal_window_events / mass_window_events
        if mass_window_events > 0
        else np.nan
    )

    return {
        "threshold": threshold,
        "selected_events": int(np.sum(selected)),
        "mass_window_events": mass_window_events,
        "peak_position": float(peak_position),
        "peak_height": int(peak_height),
        "fwhm": float(fwhm),
        "fwhm_left": float(fwhm_left),
        "fwhm_right": float(fwhm_right),
        "signal_window_events": signal_window_events,
        "signal_window_fraction": float(signal_window_fraction),
    }


def print_summary(rows):
    print("\nFSR corrected real mass analysis summary")
    print(
        "threshold | selected | mass window | peak [GeV] | "
        "peak height | FWHM [GeV] | signal frac"
    )

    for row in rows:
        print(
            f"{row['threshold']:8.2f} | "
            f"{row['selected_events']:8d} | "
            f"{row['mass_window_events']:11d} | "
            f"{row['peak_position']:10.3f} | "
            f"{row['peak_height']:11d} | "
            f"{row['fwhm']:10.3f} | "
            f"{row['signal_window_fraction']:11.4f}"
        )


def save_csv(rows, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)

    columns = [
        "threshold",
        "selected_events",
        "mass_window_events",
        "peak_position",
        "peak_height",
        "fwhm",
        "fwhm_left",
        "fwhm_right",
        "signal_window_events",
        "signal_window_fraction",
    ]

    with open(output, "w") as f:
        f.write(",".join(columns) + "\n")

        for row in rows:
            values = [str(row[col]) for col in columns]
            f.write(",".join(values) + "\n")

    print("\nSaved:", output)


def main():
    args = parse_args()

    data = np.load(args.input)

    score = data["score"]
    mass = data["m_mumu_fsr"]
    has_os_pair = data["has_os_pair"]

    rows = []

    for threshold in args.thresholds:
        row = analyze_threshold(
            score=score,
            mass=mass,
            has_os_pair=has_os_pair,
            threshold=threshold,
            mass_min=args.mass_min,
            mass_max=args.mass_max,
            bins=args.bins,
            signal_min=args.signal_min,
            signal_max=args.signal_max,
        )
        rows.append(row)

    print_summary(rows)
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()