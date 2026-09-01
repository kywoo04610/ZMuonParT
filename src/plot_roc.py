import os
import numpy as np
import matplotlib.pyplot as plt


SCORE_FILES = {
    "Baseline": "../models/baseline_test_scores.npz",
    "Transformer": "../models/transformer_test_scores.npz",
}


def main():
    output_dir = "../plots/ml"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))

    for label, path in SCORE_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file not found: {path}")

        data = np.load(path)

        bkg_eff = data["background_eff"]
        sig_eff = data["signal_eff"]
        auc = float(data["auc"])

        plt.plot(
            bkg_eff,
            sig_eff,
            label=f"{label} AUC = {auc:.4f}",
        )

    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.title("ROC curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(output_dir, "roc_baseline_vs_transformer.png")
    pdf_path = os.path.join(output_dir, "roc_baseline_vs_transformer.pdf")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print("ROC plot saved:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()