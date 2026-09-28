#!/usr/bin/env python3
"""Recompute the paper's statistics from the stored per-story result files.

Reports dispersion (mean ± SE, SD) and paired significance tests against NstAgent, the length trend
on one fixed prompt subset, the split of the global error subtype from window-local ones, and Instance
CED by error category and subtype. Nothing is re-judged: all statistics come from the stored per-story
ConStory CSVs and WritingBench JSONL files.

Outputs: paper_numbers.md (readable tables) and paper_numbers.json (machine-readable).
"""
import csv
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

from scipy import stats

csv.field_size_limit(sys.maxsize)
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
# Source paths below use the original run layout; in this repository the same per-story files live
# under results/scores/<benchmark>/<arm>, and resolve() maps each path to that copy by basename.
PACKAGE = (ROOT / "results/scores/constory").is_dir()
OUT = ROOT / ("results/tables" if PACKAGE else "eval_reports/20260923_paper_numbers")
MD_NAME = "paper_numbers.md"
PACKAGE_ARMS = {
    "ds100_deepseek_v4_flash_nstagent_10k": "deepseek/nstagent_10k",
    "deepseek_v4_flash_nstagent_10k": "deepseek/nstagent_10k",
    "deepseek_v4_flash_nstagent_20k_last7": "deepseek/nstagent_20k",
    "deepseek_v4_flash_nstagent_20k": "deepseek/nstagent_20k",
    "deepseek_v4_flash_nstagent_50k_autodl_corrected": "deepseek/nstagent_50k",
    "deepseek_v4_flash_nstagent_50k": "deepseek/nstagent_50k",
    "deepseek_v4_flash_nstagent_100k_50en": "deepseek/nstagent_100k",
    "deepseek_v4_flash_rolling_summary_unbounded16k_10k_100en": "deepseek/rollsum_10k",
    "deepseek_v4_flash_rolling_summary_unbounded16k_20k_100en_last8": "deepseek/rollsum_20k",
    "deepseek_v4_flash_rolling_summary_unbounded16k_20k_100en": "deepseek/rollsum_20k",
    "deepseek_v4_flash_rolling_summary_unbounded16k_50k_100en": "deepseek/rollsum_50k",
    "deepseek_v4_flash_rolling_summary_unbounded16k_100k_50en": "deepseek/rollsum_100k",
    "deepseek_v4_flash_rolling_summary_unbounded16k_100k_50en_rep2": "deepseek/rollsum_100k",
    "deepseek_v4_flash_direct_10k_100en": "deepseek/direct_10k",
    "deepseek_v4_flash_storywriter_10k_100en": "deepseek/storywriter_10k",
    "six20_deepseek_v4_flash_dome_10k": "deepseek/dome_10k_20prompts",
    "deepseek_v4_flash_old_dome_10k": "deepseek/dome_10k_20prompts",
    "deepseek_v4_flash_nstagent_ablation_no_state_20k_100en": "deepseek/ablation_no_state_20k",
    "deepseek_v4_flash_nstagent_ablation_no_lookback_20k_100en": "deepseek/ablation_no_lookback_20k",
    "luna100en_gpt56_luna_nstagent_10k": "luna/nstagent_10k",
    "gpt56_luna_nstagent_10k_100en": "luna/nstagent_10k",
    "gpt56_luna_nstagent_20k_last7": "luna/nstagent_20k",
    "gpt56_luna_nstagent_20k_100en": "luna/nstagent_20k",
    "gpt56_luna_nstagent_50k_100en": "luna/nstagent_50k",
    "gpt56_luna_nstagent_100k_50en": "luna/nstagent_100k",
    "luna100en_gpt56_luna_rolling_summary_unbounded16k_10k": "luna/rollsum_10k",
    "gpt56_luna_rolling_summary_unbounded16k_10k_100en": "luna/rollsum_10k",
    "gpt56_luna_rolling_summary_unbounded16k_20k_last7": "luna/rollsum_20k",
    "gpt56_luna_rolling_summary_unbounded16k_20k_100en": "luna/rollsum_20k",
    "gpt56_luna_rolling_summary_unbounded16k_50k_100en": "luna/rollsum_50k",
    "gpt56_luna_rolling_summary_unbounded16k_100k_50en": "luna/rollsum_100k",
    "gpt56_luna_direct_10k_100en": "luna/direct_10k",
    "gpt56_luna_storywriter_10k_100en": "luna/storywriter_10k",
    "six20_gpt56_luna_dome_10k": "luna/dome_10k_20prompts",
    "gpt56_luna_dome_10k": "luna/dome_10k_20prompts",
}


def resolve(path):
    """Original run path, or its shipped copy under results/scores/ in this repository."""
    if not PACKAGE:
        return ROOT / path
    stem = Path(path).stem
    arm = PACKAGE_ARMS.get(stem)
    if arm is None:
        return ROOT / path  # e.g. the -FutureReqs ablation, which the package does not ship
    sub = "writingbench" if path.endswith(".jsonl") else "constory"
    ext = ".jsonl" if sub == "writingbench" else ".csv"
    return ROOT / f"results/scores/{sub}/{arm}{ext}"
R = "eval/reruns/"
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
GLOBAL_SUBTYPE = "timeline_plot_abandoned_plot_elements"
SIX = R + "20260915_six_methods_100en_10k_v4pro/"
FM = R + "20260914_10k_fullmarker_reeval_v4pro/constory_improved/"
K100 = R + "20260915_100k_50en_2methods_v4pro/"
DS100 = R + "20260911_deepseek100_nst_rollsum_scaling_v4pro/"
DSU = R + "20260915_deepseek100_rollsum_unbounded16k_v4pro/"
LUNA = R + "20260913_gpt56_luna_100en_unbounded_rollsum_v4pro/"
LUNA_L7 = R + "20260915_luna100en_20k_last7_v4pro/constory_improved/"
DS_NST50 = R + "20260916_ds100_nst50k_blsc_reeval_autodl_v4pro/constory_improved/deepseek_v4_flash_nstagent_50k_autodl_corrected.csv"
DSU100 = R + "20260916_ds_rollsum_unbounded16k_100k_50en_v4pro/"
ABL = R + "20260915_nstagent_ablation_a1a2a7_v4pro/"


def wb_scores(path):
    out = {}
    p = resolve(path)
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "index" not in r or r.get("evaluation_status") != "completed" or r.get("criteria_completed") != 5:
            continue
        vals = []
        for v in (r.get("scores") or {}).values():
            if isinstance(v, list):
                done = [x for x in v if isinstance(x, dict) and x.get("status", "completed") == "completed"
                        and isinstance(x.get("score"), (int, float))]
                v = done[-1]["score"] if done else None
            elif isinstance(v, dict):
                v = v.get("score")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[str(r["index"])] = st.fmean(vals)
    return out


def ced_scores(path, window=None):
    """Per-story Subtype/Instance CED-original, plus the split into window-local and global errors."""
    out = {}
    p = resolve(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("id") or r.get("evaluation_status") != "completed" \
                    or r.get("criteria_completed") not in ("5", "5.0"):
                continue
            if window is not None and r.get("target_chapter_count") != str(window):
                continue
            counts = {k: len(json.loads(r.get(k) or "[]")) for k in SUBTYPES}
            w = float(r["checked_words"])
            ins = sum(counts.values())
            sub = sum(v > 0 for v in counts.values())
            g_ins = counts[GLOBAL_SUBTYPE]
            out[str(r["id"])] = dict(
                W=w, E_ins=ins, E_sub=sub,
                sub=sub / w * 1e4, ins=ins / w * 1e4,
                ins_local=(ins - g_ins) / w * 1e4, ins_global=g_ins / w * 1e4,
                sub_local=(sub - (1 if g_ins else 0)) / w * 1e4,
                by_sub={k: v / w * 1e4 for k, v in counts.items()},
            )
    return out


def desc(values):
    n = len(values)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), se=float("nan"))
    sd = st.stdev(values) if n > 1 else 0.0
    return dict(n=n, mean=st.fmean(values), sd=sd, se=sd / math.sqrt(n) if n else float("nan"))


def paired(a, b, ids=None):
    common = sorted((set(a) & set(b)) if ids is None else (set(a) & set(b) & set(ids)), key=int)
    if len(common) < 5:
        return None
    x, y = [a[i] for i in common], [b[i] for i in common]
    d = [u - v for u, v in zip(x, y)]
    rng = random.Random(20260923)
    boot = sorted(st.fmean(rng.choices(d, k=len(d))) for _ in range(10000))
    nz = any(v != 0 for v in d)
    return dict(n=len(common), diff=st.fmean(d), lo=boot[250], hi=boot[9749],
                t=stats.ttest_rel(x, y).pvalue if nz else 1.0,
                w=stats.wilcoxon(x, y).pvalue if nz else 1.0)


def holm(results, key="t"):
    """Holm-Bonferroni per metric: the family is the set of comparisons sharing one metric.

    Keys are (comparison, metric) tuples, so correcting within a metric adjusts across the variants or
    lengths being compared, which is the family a reader of one column actually scans.
    """
    by_metric = {}
    for k, v in results.items():
        if v:
            by_metric.setdefault(k[1], []).append((k, v))
    for metric, items in by_metric.items():
        prev = 0.0
        for rank, (k, v) in enumerate(sorted(items, key=lambda kv: kv[1][key])):
            v["p_holm"] = max(prev, min(1.0, v[key] * (len(items) - rank)))
            prev = v["p_holm"]
    return results


def col(d, k):
    return {i: v[k] for i, v in d.items()}


def arm(label, wb_path, ced_path, window=None):
    return dict(label=label, wb=wb_scores(wb_path), ced=ced_scores(ced_path, window))


# ---------------------------------------------------------------- arms
tenk = {
    "DeepSeek-V4-Flash": [
        arm("NstAgent", DS100 + "writingbench/deepseek_v4_flash_nstagent_10k.jsonl",
            FM + "ds100_deepseek_v4_flash_nstagent_10k.csv"),
        arm("RollSum", DSU + "writingbench/deepseek_v4_flash_rolling_summary_unbounded16k_10k_100en.jsonl",
            DSU + "constory_improved/deepseek_v4_flash_rolling_summary_unbounded16k_10k_100en.csv"),
        arm("Direct", SIX + "writingbench/deepseek_v4_flash_direct_10k_100en.jsonl",
            SIX + "constory_improved/deepseek_v4_flash_direct_10k_100en.csv"),
        arm("StoryWriter", SIX + "writingbench/deepseek_v4_flash_storywriter_10k_100en.jsonl",
            SIX + "constory_improved/deepseek_v4_flash_storywriter_10k_100en.csv"),
        arm("DOME (20)", R + "20260910_old_deepseek_six_methods_current_v4pro/writingbench/deepseek_v4_flash_old_dome_10k.jsonl",
            FM + "six20_deepseek_v4_flash_dome_10k.csv"),
    ],
    "GPT-5.6 Luna": [
        arm("NstAgent", LUNA + "writingbench/gpt56_luna_nstagent_10k_100en.jsonl",
            FM + "luna100en_gpt56_luna_nstagent_10k.csv"),
        arm("RollSum", LUNA + "writingbench/gpt56_luna_rolling_summary_unbounded16k_10k_100en.jsonl",
            FM + "luna100en_gpt56_luna_rolling_summary_unbounded16k_10k.csv"),
        arm("Direct", SIX + "writingbench/gpt56_luna_direct_10k_100en.jsonl",
            SIX + "constory_improved/gpt56_luna_direct_10k_100en.csv"),
        arm("StoryWriter", SIX + "writingbench/gpt56_luna_storywriter_10k_100en.jsonl",
            SIX + "constory_improved/gpt56_luna_storywriter_10k_100en.csv"),
        arm("DOME (20)", R + "20260910_gpt56_luna_six_methods_v4pro/writingbench/gpt56_luna_dome_10k.jsonl",
            FM + "six20_gpt56_luna_dome_10k.csv"),
    ],
}
scaling = {
    "DeepSeek-V4-Flash": [
        ("10K", arm("NstAgent", DS100 + "writingbench/deepseek_v4_flash_nstagent_10k.jsonl",
                    FM + "ds100_deepseek_v4_flash_nstagent_10k.csv"),
         arm("RollSum", DSU + "writingbench/deepseek_v4_flash_rolling_summary_unbounded16k_10k_100en.jsonl",
             DSU + "constory_improved/deepseek_v4_flash_rolling_summary_unbounded16k_10k_100en.csv")),
        ("20K", arm("NstAgent", DS100 + "writingbench/deepseek_v4_flash_nstagent_20k.jsonl",
                    R + "20260915_ds100_nst20k_last7_v4pro/constory_improved/deepseek_v4_flash_nstagent_20k_last7.csv", 7),
         arm("RollSum (last 8)", DSU + "writingbench/deepseek_v4_flash_rolling_summary_unbounded16k_20k_100en.jsonl",
             DSU + "constory_improved/deepseek_v4_flash_rolling_summary_unbounded16k_20k_100en_last8.csv", 8)),
        ("50K", arm("NstAgent", DS100 + "writingbench/deepseek_v4_flash_nstagent_50k.jsonl", DS_NST50, 5),
         arm("RollSum", DSU + "writingbench/deepseek_v4_flash_rolling_summary_unbounded16k_50k_100en.jsonl",
             DSU + "constory_improved/deepseek_v4_flash_rolling_summary_unbounded16k_50k_100en.csv", 5)),
        ("100K", arm("NstAgent", K100 + "writingbench/deepseek_v4_flash_nstagent_100k_50en.jsonl",
                     K100 + "constory_improved/deepseek_v4_flash_nstagent_100k_50en.csv", 4),
         arm("RollSum", DSU100 + "writingbench/deepseek_v4_flash_rolling_summary_unbounded16k_100k_50en_rep2.jsonl",
             DSU100 + "constory_improved/deepseek_v4_flash_rolling_summary_unbounded16k_100k_50en.csv", 4)),
    ],
    "GPT-5.6 Luna": [
        ("10K", arm("NstAgent", LUNA + "writingbench/gpt56_luna_nstagent_10k_100en.jsonl",
                    FM + "luna100en_gpt56_luna_nstagent_10k.csv"),
         arm("RollSum", LUNA + "writingbench/gpt56_luna_rolling_summary_unbounded16k_10k_100en.jsonl",
             FM + "luna100en_gpt56_luna_rolling_summary_unbounded16k_10k.csv")),
        ("20K", arm("NstAgent", LUNA + "writingbench/gpt56_luna_nstagent_20k_100en.jsonl",
                    LUNA_L7 + "gpt56_luna_nstagent_20k_last7.csv", 7),
         arm("RollSum", LUNA + "writingbench/gpt56_luna_rolling_summary_unbounded16k_20k_100en.jsonl",
             LUNA_L7 + "gpt56_luna_rolling_summary_unbounded16k_20k_last7.csv", 7)),
        ("50K", arm("NstAgent", LUNA + "writingbench/gpt56_luna_nstagent_50k_100en.jsonl",
                    LUNA + "constory_improved/gpt56_luna_nstagent_50k_100en.csv", 5),
         arm("RollSum", LUNA + "writingbench/gpt56_luna_rolling_summary_unbounded16k_50k_100en.jsonl",
             LUNA + "constory_improved/gpt56_luna_rolling_summary_unbounded16k_50k_100en.csv", 5)),
        ("100K", arm("NstAgent", K100 + "writingbench/gpt56_luna_nstagent_100k_50en.jsonl",
                     K100 + "constory_improved/gpt56_luna_nstagent_100k_50en.csv", 4),
         arm("RollSum", K100 + "writingbench/gpt56_luna_rolling_summary_unbounded16k_100k_50en.jsonl",
             K100 + "constory_improved/gpt56_luna_rolling_summary_unbounded16k_100k_50en.csv", 4)),
    ],
}
ablation = [
    arm("NstAgent", DS100 + "writingbench/deepseek_v4_flash_nstagent_20k.jsonl",
        R + "20260915_ds100_nst20k_last7_v4pro/constory_improved/deepseek_v4_flash_nstagent_20k_last7.csv", 7),
    arm("-State", ABL + "writingbench/deepseek_v4_flash_nstagent_ablation_no_state_20k_100en.jsonl",
        ABL + "constory_improved/deepseek_v4_flash_nstagent_ablation_no_state_20k_100en.csv", 7),
    arm("-Lookback", ABL + "writingbench/deepseek_v4_flash_nstagent_ablation_no_lookback_20k_100en.jsonl",
        ABL + "constory_improved/deepseek_v4_flash_nstagent_ablation_no_lookback_20k_100en.csv", 7),
    arm("-FutureReqs", ABL + "writingbench/deepseek_v4_flash_nstagent_ablation_no_future_requirements_20k_100en.jsonl",
        ABL + "constory_improved/deepseek_v4_flash_nstagent_ablation_no_future_requirements_20k_100en.csv", 7),
    arm("RollSum (last 8)", DSU + "writingbench/deepseek_v4_flash_rolling_summary_unbounded16k_20k_100en.jsonl",
        DSU + "constory_improved/deepseek_v4_flash_rolling_summary_unbounded16k_20k_100en_last8.csv", 8),
]

# ---------------------------------------------------------------- report
subset_ids = set(scaling["DeepSeek-V4-Flash"][3][1]["ced"]) | set(scaling["GPT-5.6 Luna"][3][1]["ced"])
md = ["# Paper numbers: dispersion, paired tests, fixed subset, and global-subtype split", "",
      f"Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}. Recomputed from the stored per-story results; nothing is re-judged.",
      "CED is CED-original = E/W×10⁴. SE is the standard error of the mean; [ ] after a paired difference is the 95% bootstrap CI,",
      "p is from a paired t-test, and `Holm` is the Holm-corrected p within one metric (across variants or lengths).", ""]
payload = {}


def stat_line(label, values):
    d = desc(values)
    return f"{d['mean']:.3f} ± {d['se']:.3f} (SD {d['sd']:.3f}, n={d['n']})"


md += ["## 1. 10K: mean ± SE and paired tests against NstAgent", ""]
for model, arms in tenk.items():
    md += [f"### {model}", "", "| Method | Subtype CED | Instance CED | Writing Quality |", "|---|---|---|---|"]
    for a in arms:
        md.append(f"| {a['label']} | {stat_line('', list(col(a['ced'], 'sub').values()))} "
                  f"| {stat_line('', list(col(a['ced'], 'ins').values()))} "
                  f"| {stat_line('', list(a['wb'].values()))} |")
    ref = arms[0]
    fam = {}
    for a in arms[1:]:
        if a["label"].startswith("DOME"):
            continue
        for metric, getter in (("sub", lambda x: col(x["ced"], "sub")),
                               ("ins", lambda x: col(x["ced"], "ins")),
                               ("wb", lambda x: x["wb"])):
            fam[(a["label"], metric)] = paired(getter(ref), getter(a))
    holm(fam)
    md += ["", "| Pair (NstAgent − baseline) | Metric | Difference [95% CI] | p (t) | Holm |", "|---|---|---|---|---|"]
    for (lab, metric), r in fam.items():
        if r:
            md.append(f"| NstAgent − {lab} | {metric} | {r['diff']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                      f"| {r['t']:.2g} | {r['p_holm']:.2g} |")
    payload[f"10k::{model}"] = {f"{k[0]}::{k[1]}": v for k, v in fam.items()}
    md.append("")

md += ["## 2. 10K–100K: mean ± SE, paired tests, and global-subtype split", ""]
for model, rows in scaling.items():
    md += [f"### {model}", "",
           "| Length | Method | Subtype CED | Instance CED | Writing Quality | Global subtype share | Local-only Instance CED |",
           "|---|---|---|---|---|---|---|"]
    fam = {}
    for length, nst, roll in rows:
        for a in (nst, roll):
            g = st.fmean(col(a["ced"], "ins_global").values())
            loc = st.fmean(col(a["ced"], "ins_local").values())
            tot = st.fmean(col(a["ced"], "ins").values())
            md.append(f"| {length} | {a['label']} | {stat_line('', list(col(a['ced'], 'sub').values()))} "
                      f"| {stat_line('', list(col(a['ced'], 'ins').values()))} "
                      f"| {stat_line('', list(a['wb'].values()))} "
                      f"| {g:.3f} ({g / tot * 100:.1f}%) | {loc:.3f} |")
        for metric, getter in (("sub", lambda x: col(x["ced"], "sub")),
                               ("ins", lambda x: col(x["ced"], "ins")),
                               ("ins_local", lambda x: col(x["ced"], "ins_local")),
                               ("wb", lambda x: x["wb"])):
            fam[(length, metric)] = paired(getter(nst), getter(roll))
    holm(fam)
    md += ["", "| Length | Metric | NstAgent − RollSum [95% CI] | p (t) | p (Wilcoxon) | Holm |", "|---|---|---|---|---|---|"]
    for (length, metric), r in fam.items():
        if r:
            md.append(f"| {length} | {metric} | {r['diff']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                      f"| {r['t']:.2g} | {r['w']:.2g} | {r['p_holm']:.2g} | ")
    payload[f"scaling::{model}"] = {f"{k[0]}::{k[1]}": v for k, v in fam.items()}
    md.append("")

md += ["## 3. Length trend on the fixed 50-prompt subset (the 100K sample)", "",
       f"Subset size: {len(subset_ids)} ids. At every length only stories in this subset are kept, so the sample composition does not change.", ""]
for model, rows in scaling.items():
    md += [f"### {model}", "", "| Length | Method | n | Subtype CED | Instance CED | Writing Quality |", "|---|---|---:|---|---|---|"]
    sub_fam = {}
    for length, nst, roll in rows:
        for a in (nst, roll):
            ids = [i for i in a["ced"] if i in subset_ids]
            wb_ids = [i for i in a["wb"] if i in subset_ids]
            md.append(f"| {length} | {a['label']} | {len(ids)} "
                      f"| {stat_line('', [a['ced'][i]['sub'] for i in ids])} "
                      f"| {stat_line('', [a['ced'][i]['ins'] for i in ids])} "
                      f"| {stat_line('', [a['wb'][i] for i in wb_ids])} |")
        for metric, getter in (("sub", lambda x: col(x["ced"], "sub")),
                               ("ins", lambda x: col(x["ced"], "ins")),
                               ("wb", lambda x: x["wb"])):
            sub_fam[(length, metric)] = paired(getter(nst), getter(roll), ids=subset_ids)
    holm(sub_fam)
    md += ["", "| Length | Metric | NstAgent − RollSum [95% CI] | p (t) | Holm |", "|---|---|---|---|---|"]
    for (length, metric), r in sub_fam.items():
        if r:
            md.append(f"| {length} | {metric} | {r['diff']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                      f"| {r['t']:.2g} | {r['p_holm']:.2g} |")
    payload[f"subset::{model}"] = {f"{k[0]}::{k[1]}": v for k, v in sub_fam.items()}
    md.append("")

md += ["## 4. 20K ablation (DeepSeek-V4-Flash, terminal window of 7 chapters; RollSum with 8 chapters as reference)", "",
       "| Variant | Subtype CED | Instance CED | Writing Quality |", "|---|---|---|---|"]
for a in ablation:
    if not a["ced"]:  # a variant whose per-story files this copy does not ship
        continue
    md.append(f"| {a['label']} | {stat_line('', list(col(a['ced'], 'sub').values()))} "
              f"| {stat_line('', list(col(a['ced'], 'ins').values()))} "
              f"| {stat_line('', list(a['wb'].values()))} |")
ref = ablation[0]
fam = {}
for a in ablation[1:]:
    for metric, getter in (("sub", lambda x: col(x["ced"], "sub")),
                           ("ins", lambda x: col(x["ced"], "ins")),
                           ("wb", lambda x: x["wb"])):
        fam[(a["label"], metric)] = paired(getter(a), getter(ref))  # variant - full, positive = worse
holm(fam)
md += ["", "| Pair (variant − full) | Metric | Difference [95% CI] | p (t) | Holm |", "|---|---|---|---|---|"]
for (lab, metric), r in fam.items():
    if r:
        md.append(f"| {lab} − NstAgent | {metric} | {r['diff']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                  f"| {r['t']:.2g} | {r['p_holm']:.2g} |")
payload["ablation"] = {f"{k[0]}::{k[1]}": v for k, v in fam.items()}
md.append("")

# ------------------------------------------------ errors by category and subtype (appendix C.1)
CATEGORIES = ["characterization", "factual_detail", "narrative_style", "timeline_plot", "world_building"]


def by_type(a):
    """Mean instance CED per subtype and per category over the arm's stories (per-story density, then mean)."""
    stories = [v["by_sub"] for v in a["ced"].values()]
    subs = {k: st.fmean(s[k] for s in stories) for k in SUBTYPES}
    cats = {c: st.fmean(sum(v for k, v in s.items() if k.startswith(c + "_")) for s in stories)
            for c in CATEGORIES}
    return subs, cats, len(stories)


def lowest_count(values_by_method, ours="NstAgent"):
    """Subtypes where NstAgent is strictly lowest on unrounded values (all-zero rows skipped)."""
    n = 0
    for k in SUBTYPES:
        vals = {m: v[k] for m, v in values_by_method.items()}
        if max(vals.values()) > 0 and all(vals[ours] < v for m, v in vals.items() if m != ours):
            n += 1
    return n


md += ["## 5. Instance CED by error category and subtype (Appendix C.1)", "",
       "Computed per story as errors / checked words × 10⁴, then averaged; the category values sum to the Instance CED of Tables 2 and 3.", ""]
payload["by_type"] = {}
ORDER_10K = ["Direct", "DOME (20)", "StoryWriter", "RollSum", "NstAgent"]
for model, arms_ in tenk.items():
    got = {a["label"]: by_type(a) for a in arms_}
    md += [f"### 10K, {model}", "", "| Error category | " + " | ".join(ORDER_10K) + " |", "|---" * (len(ORDER_10K) + 1) + "|"]
    for c in CATEGORIES:
        md.append(f"| {c} | " + " | ".join(f"{got[m][1][c]:.2f}" for m in ORDER_10K) + " |")
    md.append("| Total | " + " | ".join(f"{sum(got[m][1].values()):.3f}" for m in ORDER_10K) + " |")
    md += ["", "| Subtype | " + " | ".join(ORDER_10K) + " |", "|---" * (len(ORDER_10K) + 1) + "|"]
    for k in SUBTYPES:
        md.append(f"| {k} | " + " | ".join(f"{got[m][0][k]:.2f}" for m in ORDER_10K) + " |")
    n = lowest_count({m: got[m][0] for m in ORDER_10K})
    md += ["", f"NstAgent is the lowest of the five methods on {n} of the 19 subtypes (unrounded values).", ""]
    payload["by_type"][f"10k::{model}"] = {m: {"categories": got[m][1], "subtypes": got[m][0], "n": got[m][2]}
                                           for m in ORDER_10K}
for model, rows in scaling.items():
    lengths = [(length, nst, rs) for length, nst, rs in rows if length != "10K"]
    got = {length: (by_type(nst), by_type(rs)) for length, nst, rs in lengths}
    head = " | ".join(f"{length} NstAgent | {length} RollSum" for length, _, _ in lengths)
    md += [f"### 20K–100K, {model}", "", f"| Error category | {head} |", "|---" * (2 * len(lengths) + 1) + "|"]
    for c in CATEGORIES:
        md.append(f"| {c} | " + " | ".join(f"{got[l][0][1][c]:.2f} | {got[l][1][1][c]:.2f}" for l, _, _ in lengths) + " |")
    md += ["", f"| Subtype | {head} |", "|---" * (2 * len(lengths) + 1) + "|"]
    for k in SUBTYPES:
        md.append(f"| {k} | " + " | ".join(f"{got[l][0][0][k]:.2f} | {got[l][1][0][k]:.2f}" for l, _, _ in lengths) + " |")
    wins = [lowest_count({"NstAgent": got[l][0][0], "RollSum": got[l][1][0]}) for l, _, _ in lengths]
    cat_wins = sum(got[l][0][1][c] < got[l][1][1][c] for l, _, _ in lengths for c in CATEGORIES)
    md += ["", f"Subtypes where NstAgent is lower (20K / 50K / 100K): {' / '.join(map(str, wins))}; "
           f"error categories where it is lower: {cat_wins} / {len(CATEGORIES) * len(lengths)}.", ""]
    payload["by_type"][f"scaling::{model}"] = {
        l: {"NstAgent": {"categories": got[l][0][1], "subtypes": got[l][0][0]},
            "RollSum": {"categories": got[l][1][1], "subtypes": got[l][1][0]}} for l, _, _ in lengths}

abl = [a for a in ablation if a["ced"]]  # the package does not ship the -FutureReqs arm
got = {a["label"]: by_type(a) for a in abl}
names = [a["label"] for a in abl]
md += ["### 20K ablation (DeepSeek-V4-Flash)", "", "| Error category | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
for c in CATEGORIES:
    md.append(f"| {c} | " + " | ".join(f"{got[m][1][c]:.2f}" for m in names) + " |")
md.append("| Total | " + " | ".join(f"{sum(got[m][1].values()):.3f}" for m in names) + " |")
md += ["", "| Subtype | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
for k in SUBTYPES:
    md.append(f"| {k} | " + " | ".join(f"{got[m][0][k]:.2f}" for m in names) + " |")
md += ["", f"NstAgent is the lowest on {lowest_count({m: got[m][0] for m in names})} of the 19 subtypes (unrounded values).", ""]
payload["by_type"]["ablation"] = {m: {"categories": got[m][1], "subtypes": got[m][0], "n": got[m][2]} for m in names}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / MD_NAME).write_text("\n".join(md), encoding="utf-8")
(OUT / "paper_numbers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(md))
