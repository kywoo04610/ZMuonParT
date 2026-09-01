## plot_style.py
## 발표용 그래프 스타일을 통일하는 코드입니다.

import os
import matplotlib.pyplot as plt


CMS_BLUE = "#1F77B4"
CMS_RED = "#D62728"
CMS_BLACK = "#000000"
CMS_GRAY = "#7F7F7F"


def set_presentation_style():
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 5.2),
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "lines.linewidth": 2.6,
            "lines.markersize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_cms_label(ax, right_label="Particle Transformer v4"):
    ax.text(
        0.02,
        0.96,
        "CMS Open Data",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color=CMS_BLACK,
    )

    ax.text(
        0.02,
        0.89,
        "Run2016H SingleMuon",
        transform=ax.transAxes,
        fontsize=13,
        va="top",
        ha="left",
        color=CMS_BLACK,
    )

    ax.text(
        0.98,
        0.96,
        right_label,
        transform=ax.transAxes,
        fontsize=13,
        va="top",
        ha="right",
        color=CMS_BLACK,
    )


def save_figure(fig, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    if output_path.endswith(".png"):
        fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
        fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")