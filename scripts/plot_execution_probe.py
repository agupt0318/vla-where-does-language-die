"""Plot canonical analysis_seedN.json with episode-bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(path, output=None):
    data = json.loads(path.read_text())
    if "sites" not in data or not all("delta" in v for v in data["sites"].values()):
        raise ValueError("Expected canonical analysis_seedN.json from analyze_execution_probe")
    names = sorted(data["sites"], key=lambda x: (x.split(".")[0], int(x.split(".")[1][1:])))
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.3), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    yy = np.arange(len(names))
    for key, offset, color, label in [
        ("named", -0.13, "#245d94", "Named destination"),
        ("selected", 0.13, "#bc6132", "Motion-derived destination"),
    ]:
        means = np.array([data["sites"][n][key]["value"] for n in names])
        lo = np.array([data["sites"][n][key]["lo"] for n in names])
        hi = np.array([data["sites"][n][key]["hi"] for n in names])
        axes[0].errorbar(
            means,
            yy + offset,
            xerr=[means - lo, hi - means],
            fmt="o",
            ms=4,
            capsize=2,
            color=color,
            label=label,
        )
    delta = np.array([data["sites"][n]["delta"]["value"] for n in names])
    lo = np.array([data["sites"][n]["delta"]["lo"] for n in names])
    hi = np.array([data["sites"][n]["delta"]["hi"] for n in names])
    axes[1].errorbar(delta, yy, xerr=[delta - lo, hi - delta], fmt="o", color="#385e49", capsize=2)
    axes[1].axvline(0, color="gray", ls="--", lw=1)
    axes[0].set_yticks(yy, [n.replace(".resid_post", "") for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.04, 1.04)
    axes[0].set_xlabel("Probe agreement")
    axes[1].set_xlabel("Motion agreement minus named agreement")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Novel pairings: n={data['n_valid_novel']} trials, "
        f"{data['n_test_episodes']} held-out episodes",
        fontsize=12,
    )
    fig.tight_layout()
    output = output or path.parent / "execution_probe.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    print(plot(a.analysis, a.output))
