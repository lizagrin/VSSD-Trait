"""Plotting helpers used by scripts/make_figures.py and the notebook."""
from __future__ import annotations

import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]


def _save(fig: plt.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_modality_ablation(table: pd.DataFrame, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(TRAITS)); w = 0.18
    for i, name in enumerate(table.index):
        ax.bar(x + (i - 1.5) * w, table.loc[name, TRAITS].values, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(TRAITS)
    ax.set_ylabel("mACC"); ax.set_title("Modality ablation: per-trait mACC")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=2, frameon=False)
    return _save(fig, out_path)


def plot_window_attention(table: pd.DataFrame, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(table.index, table["Средний вес α"], color="#1f77b4", alpha=0.8)
    ax.plot(table.index, table["Средний вес α"], "o-", color="#d62728")
    ax.set_xlabel("Window index (15-second clip)")
    ax.set_ylabel("Mean attention weight α")
    ax.set_title("Temporal saliency of clip windows")
    return _save(fig, out_path)


def plot_fusion_heatmap(table: pd.DataFrame, out_path: str) -> str:
    W = table[["w_V (видео)", "w_A (аудио)", "w_T (текст)"]].values
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(W, cmap="YlGnBu", vmin=0.15, vmax=0.55, aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["w_V", "w_A", "w_T"])
    ax.set_yticks(range(len(table))); ax.set_yticklabels(table.index)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.3f}", ha="center", va="center",
                    color=("white" if W[i, j] > 0.4 else "black"), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Late-fusion modality weights per trait")
    return _save(fig, out_path)


def plot_seed_reproducibility(table: pd.DataFrame, out_path: str,
                              n_seeds: int = 5) -> str:
    metrics: List[str] = list(table.index)
    mu = table["Среднее"].values.astype(float)
    sd = table["Std"].values.astype(float)
    t4 = 2.776  # t_{n-1=4}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(metrics)), mu,
                yerr=t4 * sd / math.sqrt(n_seeds),
                fmt="o", color="#2ca02c", capsize=4, lw=2)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.split()[0] for m in metrics], rotation=15)
    ax.set_ylabel("Среднее ± 95% CI")
    ax.set_title("5-seed reproducibility")
    return _save(fig, out_path)
