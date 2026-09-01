## plot_model_comparison_presentation.py
## 발표용 모델 성능 비교 그래프를 생성합니다.
##
## 생성:
##   - model_accuracy_comparison.png/pdf/svg
##   - model_auc_comparison.png/pdf/svg
##   - model_background_rejection_comparison.png/pdf/svg
##
## 실행:
## python plot_model_comparison_presentation.py \
##     --output-dir ../plots/ml/presentation

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


MODELS = [
    "MLP",
    "Transformer",
    "ParT v1",
    "ParT v2",
    "ParT v3",
    "ParT v4",
]

ACCURACY = np.array([
    0.9183807887,
    0.9277767385,
    0.9276697192,
    0.9305770779,
    0.9324020621,
    0.9320118330,
])

AUC = np.array([
    0.9511889174,
    0.9734508136,
    0.9748033921,
    0.9754034209,
    0.9744991659,
    0.9760299948,
])

BKG_REJ = np.array([
    6.52,
    11.34,
    12.38,
    12.02,
    11.41,
    11.98,
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../plots/ml/presentation",
    )
    return parser.parse_args()


def plot_line(
    values,
    ylabel,
    title,
    output_path,
    color,
    ylim=None,
    value_format="{:.4f}",
):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    x = np.arange(len(MODELS))

    ax.plot(
        x,
        values,
        color=color,
        marker="o",
        linewidth=2.8,
        markersize=8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.25)

    y_offset = 0.00035
    if ylim is not None:
        y_offset = 0.02 * (ylim[1] - ylim[0])

    for xi, yi in zip(x, values):
        ax.text(
            xi,
            yi + y_offset,
            value_format.format(yi),
            ha="center",
            va="bottom",
            fontsize=12,
            color=CMS_BLACK,
        )

    fig.subplots_adjust(
        left=0.13,
        bottom=0.15,
        right=0.98,
        top=0.90,
    )

    save_figure(fig, output_path)
    plt.close(fig)


def plot_bar(
    values,
    ylabel,
    title,
    output_path,
    color,
    ylim=None,
    value_format="{:.2f}",
):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    x = np.arange(len(MODELS))

    bars = ax.bar(
        x,
        values,
        color=color,
        edgecolor=CMS_BLACK,
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    y_offset = 0.08
    if ylim is not None:
        y_offset = 0.015 * (ylim[1] - ylim[0])

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + y_offset,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=12,
            color=CMS_BLACK,
        )

    fig.subplots_adjust(
        left=0.14,
        bottom=0.15,
        right=0.98,
        top=0.90,
    )

    save_figure(fig, output_path)
    plt.close(fig)


def main():
    args = parse_args()

    set_presentation_style()
    os.makedirs(args.output_dir, exist_ok=True)

    plot_line(
        values=ACCURACY,
        ylabel="Accuracy",
        title="Model accuracy comparison",
        output_path=os.path.join(
            args.output_dir,
            "model_accuracy_comparison.png",
        ),
        color=CMS_BLUE,
        ylim=(0.915, 0.935),
        value_format="{:.4f}",
    )

    plot_line(
        values=AUC,
        ylabel="AUC",
        title="Model AUC comparison",
        output_path=os.path.join(
            args.output_dir,
            "model_auc_comparison.png",
        ),
        color=CMS_RED,
        ylim=(0.950, 0.978),
        value_format="{:.4f}",
    )

    plot_bar(
        values=BKG_REJ,
        ylabel="Background rejection",
        title="Background rejection comparison (threshold = 0.5)",
        output_path=os.path.join(
            args.output_dir,
            "model_background_rejection_comparison.png",
        ),
        color=CMS_BLACK,
        ylim=(5.0, 13.0),
        value_format="{:.2f}",
    )

    print("Saved presentation plots to:", args.output_dir)


if __name__ == "__main__":
    main()