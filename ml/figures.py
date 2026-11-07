"""Draws the report figures from results/*.json into docs/figures.

    python -m ml.figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import RESULTS
from .core import CLASSES, ROOT

OUT = ROOT / "docs" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#a9c8ee", "#2a78d6", "#0f3f7a"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def confusion(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for ax, key, title in zip(axes, ("cnn", "rf"), ("1D CNN (int8)", "Random Forest")):
        m = np.array(res["beats"]["All test patients"][key]["confusion"], float)
        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap=RAMP, vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{share[i, j] * 100:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(6), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(6), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True class")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Where beats from unseen patients end up (% of each true class)", x=0.01, ha="left", fontsize=13)
    fig.savefig(OUT / "confusion.png", dpi=160)


def sqa(res):
    rows = res["sqa_ab"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True, constrained_layout=True)
    for ax, noise, title in zip(axes, ("em", "ma"), ("Electrode motion noise", "Muscle artefact noise")):
        r = [x for x in rows if x["noise"] == noise]
        snr = [x["snr_db"] for x in r]
        off = [x["sqa_off"]["false_alarms_per_hour"] for x in r]
        on = [x["sqa_on"]["false_alarms_per_hour"] for x in r]
        ax.plot(snr, off, color=ORANGE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
        ax.plot(snr, on, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
        ax.annotate("SQA off", (snr[-1], off[-1]), xytext=(-8, 8), textcoords="offset points", ha="right", color=INK)
        ax.annotate("SQA on", (snr[-1], on[-1]), xytext=(-8, 8), textcoords="offset points", ha="right", color=INK)
        ax.invert_xaxis()
        ax.set_xticks(snr, [f"{s} dB" for s in snr])
        ax.set_xlabel("Signal-to-noise ratio (noisier to the right)")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("False alarms per hour of noise")
    fig.suptitle("Signal quality gating removes most noise-driven false alarms", x=0.01, ha="left", fontsize=13)
    fig.savefig(OUT / "sqa_false_alarms.png", dpi=160)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((RESULTS / "evaluation.json").read_text())
    confusion(res)
    sqa(res)


if __name__ == "__main__":
    main()
