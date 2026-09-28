#!/usr/bin/env python3
"""Recompute the RL checkpoint table of the appendix from the shipped judge output.

Usage: python code/analysis/rl_table.py [package_root]

Each checkpoint wrote the same 20 ConStory-Bench prompts (data/rl/eval_prompts_20.jsonl) at 10K words
from outlines generated once by the initial model. ConStory-Bench judged every chapter
(prompts_terminal_scope with the terminal marker before chapter 0), and WritingBench scored the same
stories. CED is computed per story as errors per 10K checked words and then averaged, as in the main
tables; Writing Quality is the mean of the five criterion scores.
"""
import ast
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
RL = ROOT / "results/rl/scores"
CHECKPOINTS = [("initial", "Initial (Qwen3.5-4B)"), ("step16", "Step 16"), ("step32", "Step 32"),
               ("step48", "Step 48")]
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


def ced(path):
    sub, ins = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("evaluation_status") != "completed" or r.get("criteria_completed") not in ("5", "5.0"):
                continue
            counts = [len(json.loads(r.get(k) or "[]")) for k in SUBTYPES]
            w = float(r["checked_words"])
            sub.append(sum(c > 0 for c in counts) / w * 1e4)
            ins.append(sum(counts) / w * 1e4)
    return st.fmean(sub), st.fmean(ins), len(sub)


def writing_quality(path):
    last = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("evaluation_status") != "completed":
            continue
        scores = r["scores"]
        scores = ast.literal_eval(scores) if isinstance(scores, str) else scores
        vals = [x["score"] for v in scores.values() for x in (v if isinstance(v, list) else [v])
                if isinstance(x, dict) and isinstance(x.get("score"), (int, float))]
        if vals:
            last[r["index"]] = st.fmean(vals)  # a resumed run appends; keep the last record per story
    return st.fmean(last.values()), len(last)


md = ["# RL checkpoints of NstAgent on Qwen3.5-4B (10K words, 20 prompts, shared outlines)", "",
      "Recomputed by `code/analysis/rl_table.py` from `results/rl/scores/`.", "",
      "| Checkpoint | n | Subtype CED | Instance CED | Writing Quality |", "|---|---:|---:|---:|---:|"]
for key, label in CHECKPOINTS:
    s, i, n = ced(RL / f"constory/{key}_10k.csv")
    wq, m = writing_quality(RL / f"writingbench/{key}_10k.jsonl")
    md.append(f"| {label} | {n}/{m} | {s:.3f} | {i:.3f} | {wq:.2f} |")
md.append("")
(ROOT / "results/tables/rl.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
