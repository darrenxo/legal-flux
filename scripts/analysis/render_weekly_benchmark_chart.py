from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(
    "reports/legal_flux/weekly_2026-08-22_28_assets/benchmark_accuracy.png"
)


def main() -> None:
    datasets = [
        "AnnoCaseLaw\n(n=394)",
        "Realistic LJP Facts\n(n=1,509)",
        "IL-TUR / CJPE\n(n=1,517)",
    ]
    direct = np.array([50.0000, 52.5514, 70.2044])
    structured = np.array([50.5076, 51.8887, 69.7429])
    majority = np.array([46.9543, 50.2982, 50.2307])

    x = np.arange(len(datasets))
    width = 0.31

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    direct_bars = ax.bar(
        x - width / 2,
        direct,
        width,
        label="Direct",
        color="#1F4E79",
        edgecolor="white",
        linewidth=0.8,
    )
    structured_bars = ax.bar(
        x + width / 2,
        structured,
        width,
        label="Structured (one-shot IRAC)",
        color="#2A9D8F",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.scatter(
        x,
        majority,
        marker="D",
        s=48,
        color="#6B7280",
        label="Majority baseline",
        zorder=4,
    )

    for bars in (direct_bars, structured_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#111827",
            )

    ax.set_title("Direct vs. Structured Accuracy on Three Legal Benchmarks", pad=16)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.grid(axis="y", color="#D7DEE7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.015,
        "Qwen3.5-9B BF16 on Delta; full evaluation (6,840 generations, 0 errors).",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
