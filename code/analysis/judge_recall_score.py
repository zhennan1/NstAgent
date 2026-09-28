#!/usr/bin/env python3
"""Score the judge-recall calibration: did the judge report each injected contradiction?

Every injection carries a unique QX-XXXXXXXX marker that occurs exactly twice in its own story, so a
detection is an exact string match inside the judge's reported error objects — no fuzzy grading. We
report three increasingly permissive criteria (target subtype / target category / any category), the
control false-trigger rate, and recall as a function of the antecedent-to-manifestation distance.
"""
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from scipy import stats

csv.field_size_limit(sys.maxsize)
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
# Inside the released repository the manifest, judge output and report sit under results/.
PACKAGE = (ROOT / "results/scores/constory/judge_recall").is_dir()
REPORT = ROOT / ("results/tables" if PACKAGE else "eval_reports/20260922_judge_recall_injection_v4pro")
RUN = ROOT / ("results/scores/constory/judge_recall" if PACKAGE
              else "eval/reruns/20260922_judge_recall_injection_v4pro/constory")
MANIFEST = REPORT / ("judge_recall_manifest.jsonl" if PACKAGE else "manifest.jsonl")
RUN_CSV = "{label}.csv" if PACKAGE else "recall_{label}.csv"
REPORT_MD = "judge_recall.md"
DETECTIONS = "judge_recall_detections.jsonl" if PACKAGE else "detections.jsonl"
LENGTHS = ["10k", "20k", "50k", "100k"]
DEPTHS = ["near", "mid", "far"]
CATEGORY_COLUMNS = {
    "characterization": ["characterization_memory_contradictions", "characterization_knowledge_contradictions",
                         "characterization_skill_power_fluctuations", "characterization_forgotten_abilities"],
    "factual_detail": ["factual_detail_appearance_mismatches", "factual_detail_nomenclature_confusions",
                       "factual_detail_quantitative_mismatches"],
    "narrative_style": ["narrative_style_perspective_confusions", "narrative_style_tone_inconsistencies",
                        "narrative_style_style_shifts"],
    "timeline_plot": ["timeline_plot_absolute_time_contradictions", "timeline_plot_duration_timeline_contradictions",
                      "timeline_plot_simultaneity_contradictions", "timeline_plot_causeless_effects",
                      "timeline_plot_causal_logic_violations", "timeline_plot_abandoned_plot_elements"],
    "world_building": ["world_building_core_rules_violations", "world_building_social_norms_violations",
                       "world_building_geographical_contradictions"],
}
TARGET_SUBTYPE = {
    "characterization": "characterization_knowledge_contradictions",
    "factual_detail": "factual_detail_appearance_mismatches",
    "narrative_style": "narrative_style_style_shifts",
    "timeline_plot": "timeline_plot_duration_timeline_contradictions",
    "world_building": "world_building_core_rules_violations",
}
ALL_COLUMNS = [c for cols in CATEGORY_COLUMNS.values() for c in cols]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k, n):
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n * 100:.1f}% [{lo * 100:.0f}, {hi * 100:.0f}]"


manifest = [json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()]
results = {}
for label in LENGTHS:
    path = RUN / RUN_CSV.format(label=label)
    if not path.exists():
        continue
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("id"):
                results[(label, str(r["id"]))] = r

rows = []
for m in manifest:
    r = results.get((m["length"], str(m["story_id"])))
    if r is None or r.get("evaluation_status") != "completed" or r.get("criteria_completed") not in ("5", "5.0"):
        continue
    token = m["token"]
    hit_cols = [c for c in ALL_COLUMNS if token in (r.get(c) or "")]
    rows.append(dict(
        m,
        evaluated=True,
        in_target_subtype=TARGET_SUBTYPE[m["category"]] in hit_cols,
        in_target_category=any(c in CATEGORY_COLUMNS[m["category"]] for c in hit_cols),
        in_any_category=bool(hit_cols),
        checked_words=float(r.get("checked_words") or 0),
        hit_columns=hit_cols,
    ))
pos = [r for r in rows if r["positive"]]
ctl = [r for r in rows if not r["positive"]]
# Any inserted paragraph — contradictory or not — reads as a foreign block, so the judge flags it as a
# style shift, and an inserted object that never returns reads as an abandoned plot element. Those two
# channels measure the injection method, not the judge's contradiction detection, so control
# false-triggers are reported both raw and with these two artifact channels excluded.
ARTIFACT = {"narrative_style_style_shifts", "timeline_plot_abandoned_plot_elements"}
non_artifact = lambda r: bool(set(r["hit_columns"]) - ARTIFACT)

md = ["# Judge recall of the ConStory judge (DeepSeek-V4-Pro) under controlled error injection", "",
      f"Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}; evaluated {len(rows)}/{len(manifest)}"
      f" ({len(pos)} positives, {len(ctl)} consistent controls).", "",
      "**Question**: a lower CED at 100K could reflect the judge's recall decaying over a ~90K-word prefix rather than better stories.",
      "**Method**: a pair of explicit, self-contained statements is injected into real NstAgent stories; the later statement **always lands inside the terminal window**"
      " (whole story at 10K; last 7/5/4 chapters at 20K/50K/100K), and the earlier one is placed at one of three prefix depths. The judge and prompt are identical to the main evaluation.",
      "Each injection carries a unique marker `QX-XXXXXXXX` that occurs exactly twice in its story, so detection is an exact string match.", "",
      "## 1. By length (main result)", "",
      "| Length | Target-subtype recall | Target-category recall | Detected in any category | Controls reported as target type | Controls in any category |",
      "|---|---|---|---|---|---|"]
for label in LENGTHS:
    p = [r for r in pos if r["length"] == label]
    c = [r for r in ctl if r["length"] == label]
    md.append(f"| {label.upper()} | {rate(sum(r['in_target_subtype'] for r in p), len(p))} "
              f"| {rate(sum(r['in_target_category'] for r in p), len(p))} "
              f"| {rate(sum(r['in_any_category'] for r in p), len(p))} "
              f"| {rate(sum(r['in_target_category'] for r in c), len(c))} "
              f"| {rate(sum(r['in_any_category'] for r in c), len(c))} |")

md += ["", "## 2. By prefix depth (within each length)", "",
       "Depth is the number of words between the earlier and the later statement: `near` is just before the window, `far` is at the start of the story.", "",
       "| Length | Depth | Mean gap (words) | Target-category recall | Detected in any category |", "|---|---|---:|---|---|"]
for label in LENGTHS:
    for depth in DEPTHS:
        p = [r for r in pos if r["length"] == label and r["depth"] == depth]
        if not p:
            continue
        gap = sum(r["gap_words"] for r in p) / len(p)
        md.append(f"| {label.upper()} | {depth} | {gap:,.0f} "
                  f"| {rate(sum(r['in_target_category'] for r in p), len(p))} "
                  f"| {rate(sum(r['in_any_category'] for r in p), len(p))} |")

md += ["", "## 3. Sources of control false triggers", "",
       "Controls inject two **mutually consistent** statements, so any reported error is a false trigger. However, an inserted paragraph reads as foreign text, "
       "which the judge reports as `style_shifts`, and an inserted object that never returns is reported as `abandoned_plot_elements`. "
       "These two channels measure the injection method, not contradiction detection.", "",
       "| Criterion | Controls | Positives |", "|---|---|---|",
       f"| Reported in any category | {rate(sum(r['in_any_category'] for r in ctl), len(ctl))} "
       f"| {rate(sum(r['in_any_category'] for r in pos), len(pos))} |",
       f"| **Reported as the target contradiction type** | {rate(sum(r['in_target_category'] for r in ctl), len(ctl))} "
       f"| {rate(sum(r['in_target_category'] for r in pos), len(pos))} |",
       f"| Excluding the style-shift and abandoned-plot channels | {rate(sum(1 for r in ctl if non_artifact(r)), len(ctl))} "
       f"| {rate(sum(1 for r in pos if non_artifact(r)), len(pos))} |", "",
       f"Most frequent subtypes among controls: " + ", ".join(
           f"`{c}` {n}" for c, n in Counter(col for r in ctl for col in r["hit_columns"]).most_common(3)) + ".",
       f"Positives are reported under {sum(len(r['hit_columns']) for r in pos) / max(1, len(pos)):.2f} subtypes on average, "
       f"controls under {sum(len(r['hit_columns']) for r in ctl) / max(1, len(ctl)):.2f}: "
       "positives are reported repeatedly across subtypes (taxonomy drift).", "",
       "## 4. Trend tests", ""]
if pos:
    order = {l: i for i, l in enumerate(LENGTHS)}
    x = [order[r["length"]] for r in pos]
    for name, key in [("Target-category recall", "in_target_category"), ("Detected in any category", "in_any_category")]:
        y = [int(r[key]) for r in pos]
        if len(set(y)) > 1:
            tau = stats.kendalltau(x, y)
            lg = [math.log10(max(r["gap_words"], 1)) for r in pos]
            tau_gap = stats.kendalltau(lg, y)
            md.append(f"- **{name}**: vs. length Kendall τ={tau.statistic:+.3f} (p={tau.pvalue:.3g}); "
                      f"vs. log10(gap words) τ={tau_gap.statistic:+.3f} (p={tau_gap.pvalue:.3g}).")
        else:
            md.append(f"- **{name}**: all positives have the same outcome ({'all detected' if y[0] else 'all missed'}); no trend test needed.")
    short = [r for r in pos if r["length"] in ("10k", "20k")]
    long_ = [r for r in pos if r["length"] in ("50k", "100k")]
    if short and long_:
        a = sum(r["in_target_category"] for r in short)
        b = sum(r["in_target_category"] for r in long_)
        odds, p = stats.fisher_exact([[a, len(short) - a], [b, len(long_) - b]])
        md.append(f"- **10K+20K vs. 50K+100K** (target-category recall): {a}/{len(short)} vs. {b}/{len(long_)}, "
                  f"Fisher p={p:.3g}.")

md += ["", "## 5. Scope and limitations", "",
       "- The injected statements are explicit and self-contained, so they are easier to detect than naturally occurring contradictions; this recall is an **upper bound**.",
       "- The unique marker may act as a retrieval anchor and make distant injections easier to detect.",
       "- The later statement always lies inside the terminal window, so a miss can only be due to the judge failing to recall the earlier statement from the prefix, not to the protocol excluding the error.",
       "- The two control statements are mutually consistent and estimate the false-trigger rate; \"reported as the target contradiction type\" must be distinguished from \"reported as a style shift because of insertion traces\".", ""]
(REPORT / REPORT_MD).write_text("\n".join(md), encoding="utf-8")
with open(REPORT / DETECTIONS, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\n".join(md))
