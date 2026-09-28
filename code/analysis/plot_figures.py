#!/usr/bin/env python3
"""Redraw the paper's data figures from the shipped result files.

Figure 1 (instance CED and writing quality against target length) and Figure 3 (cost per story against
target length) share one axis style here, so a change to the look applies to every panel at once.

Every value is computed from results/, never transcribed, so the figures cannot drift away from the
data they describe. Run from the package root:

    python code/analysis/plot_figures.py [out_dir]
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/figures"
# Figure 1 palette of the paper: NstAgent is the orange series, RollSum the blue one.
NST_COLOR, RS_COLOR = "#DB7043", "#4176CF"
GRID_COLOR, AXIS_COLOR, FIT_COLOR = "#E8E7E4", "#7F7F7F", "#8C8C8C"
# USD prices behind Table 4: uncached input, cached input, output, per million tokens.
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.22, 0.007, 0.66
LENGTHS = [("10K", "10k"), ("20K", "20k"), ("50K", "50k"), ("100K", "100k")]
SUBTYPES = [
    "characterization_memory_contradictions", "characterization_knowledge_contradictions",
    "characterization_skill_power_fluctuations", "characterization_forgotten_abilities",
    "factual_detail_appearance_mismatches", "factual_detail_nomenclature_confusions",
    "factual_detail_quantitative_mismatches", "narrative_style_perspective_confusions",
    "narrative_style_tone_inconsistencies", "narrative_style_style_shifts",
    "timeline_plot_absolute_time_contradictions", "timeline_plot_duration_timeline_contradictions",
    "timeline_plot_simultaneity_contradictions", "timeline_plot_causeless_effects",
    "timeline_plot_causal_logic_violations", "timeline_plot_abandoned_plot_elements",
    "world_building_core_rules_violations", "world_building_social_norms_violations",
    "world_building_geographical_contradictions",
]
csv.field_size_limit(sys.maxsize)


def instance_ced(arm: str) -> float:
    path = ROOT / f"results/scores/constory/deepseek/{arm}.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("evaluation_status") == "completed" and r.get("criteria_completed") in ("5", "5.0")]
    return st.fmean(sum(len(json.loads(r.get(k) or "[]")) for k in SUBTYPES) / float(r["checked_words"]) * 1e4
                    for r in rows)


def writing_quality(arm: str) -> float:
    path = ROOT / f"results/scores/writingbench/deepseek/{arm}.jsonl"
    means = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("evaluation_status") != "completed" or r.get("criteria_completed") != 5:
            continue
        scores = []
        for v in (r.get("scores") or {}).values():
            if isinstance(v, list):
                done = [x for x in v if isinstance(x, dict) and x.get("status", "completed") == "completed"
                        and isinstance(x.get("score"), (int, float))]
                v = done[-1]["score"] if done else None
            elif isinstance(v, dict):
                v = v.get("score")
            if isinstance(v, (int, float)):
                scores.append(float(v))
        if scores:
            means.append(st.fmean(scores))
    return st.fmean(means)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def style_axes(ax, xlabel, ylabel):
    """Shared look: horizontal gridlines, x and y axis lines, no top or right frame."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color=GRID_COLOR, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(axis="both", length=2.2, width=0.6, color=AXIS_COLOR, pad=3)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=5)


def panel(values, ylabel, stem, ylim, yticks, lower_is_better):
    """One Figure 1 panel; only the final point is labelled, in bold for our method."""
    fig, ax = plt.subplots(figsize=(3.12, 2.28), constrained_layout=True)
    xs = list(range(len(LENGTHS)))
    series = (("NstAgent (ours)", "NstAgent", NST_COLOR, "o", "-", 2.4, 6.0),
              ("RollSum", "RollSum", RS_COLOR, "s", (0, (4, 2)), 1.4, 4.4))
    for label, key, color, marker, dash, width, msize in series:
        ax.plot(xs, values[key], label=label, color=color, marker=marker, linestyle=dash,
                linewidth=width, markersize=msize, markeredgecolor="white", markeredgewidth=0.6,
                zorder=3, clip_on=False)
    for label, key, color, _, _, _, _ in series:
        y = values[key][-1]
        ax.annotate(f"{y:.2f}", xy=(xs[-1], y), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", color=color, fontsize=8.0,
                    fontweight="bold" if key == "NstAgent" else "normal")
    style_axes(ax, "Target length (words)",
               ylabel + (r" $\downarrow$" if lower_is_better else r" $\uparrow$"))
    ax.set_xticks(xs, [label for label, _ in LENGTHS])
    ax.set_yticks(yticks)
    ax.set_xlim(-0.25, len(LENGTHS) - 0.55)
    ax.set_ylim(*ylim)
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                       handlelength=2.4, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0)
    legend.get_texts()[0].set_fontweight("bold")
    save(fig, stem)
    return {m: [round(v, 3) for v in values[m]] for m in values}


def nstagent_costs():
    """Cost per story in USD for each length, recomputed from the token counts in results/tables."""
    rows = {}
    for line in open(ROOT / "results/tables/cost.md", encoding="utf-8"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or cells[0] != "NstAgent":
            continue
        num = lambda c: float(c.split("(")[0].split("（")[0].replace(",", ""))
        length, total_in, cached, out = cells[1], num(cells[4]), num(cells[5]), num(cells[6])
        rows[length] = ((total_in - cached) * PRICE_IN + cached * PRICE_CACHED + out * PRICE_OUT) / 1e6
    return [rows[label] for label, _ in LENGTHS]


def cost_figure(stem="nstagent_cost_scaling"):
    """Figure 3: cost against target length, with the least-squares line the text quotes."""
    costs = nstagent_costs()
    xs = [10_000, 20_000, 50_000, 100_000]
    n = len(xs)
    mx, my = sum(xs) / n, sum(costs) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, costs)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, costs))
    ss_tot = sum((y - my) ** 2 for y in costs)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(2.95, 2.45), constrained_layout=True)
    fit_x = [0, 108_000]
    ax.plot(fit_x, [slope * x + intercept for x in fit_x], color=FIT_COLOR, linestyle="--",
            linewidth=1.2, zorder=2)
    ax.plot(xs, costs, color=NST_COLOR, marker="o", linewidth=2.4, markersize=6.0,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3, clip_on=False)
    ax.annotate(f"{costs[-1]:.2f}", xy=(xs[-1], costs[-1]), xytext=(0, 10),
                textcoords="offset points", ha="center", color=NST_COLOR, fontsize=8.0,
                fontweight="bold")
    ax.annotate(f"linear fit\n($R^2 = {r2:.3f}$)", xy=(0.06, 0.80), xycoords="axes fraction",
                color=FIT_COLOR, fontsize=8.0, va="top")
    style_axes(ax, "Target length (words)", "Cost per story (USD)")
    ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000], ["0", "25K", "50K", "75K", "100K"])
    ax.set_xlim(-2_000, 112_000)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])
    ax.set_ylim(0, 1.55)
    save(fig, stem)
    return {f"{label}": round(c, 3) for (label, _), c in zip(LENGTHS, costs)} | {"R2": round(r2, 4)}


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=600 if suffix == "png" else None)
    plt.close(fig)


def main():
    configure_style()
    ced = {"NstAgent": [instance_ced(f"nstagent_{k}") for _, k in LENGTHS],
           "RollSum": [instance_ced(f"rollsum_{k}") for _, k in LENGTHS]}
    wq = {"NstAgent": [writing_quality(f"nstagent_{k}") for _, k in LENGTHS],
          "RollSum": [writing_quality(f"rollsum_{k}") for _, k in LENGTHS]}
    print("Figure 1 instance CED:", panel(ced, "Instance CED", "deepseek_nst_rollsum_instance_ced",
                                          (5.6, 12.6), [6, 8, 10, 12], True))
    print("Figure 1 writing quality:", panel(wq, "Writing Quality", "deepseek_nst_rollsum_writing_quality",
                                             (8.88, 9.32), [8.9, 9.0, 9.1, 9.2, 9.3], False))
    print("Figure 3 cost:", cost_figure())


if __name__ == "__main__":
    main()
