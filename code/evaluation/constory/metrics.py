#!/usr/bin/env python
# coding: utf-8
"""
ConStory-Bench Metrics: CED and GRR

Two core metrics for evaluating long-form story consistency:

1. CED (Consistency Error Density)
   - Errors per 10,000 evaluated words
   - Lower is better
   - ``CED_original`` is the unadjusted terminal-window density:
     error_count / (checked_words / 10000)
   - ``CED_new`` estimates full-story density under the explicit model that
     the two endpoints of an inconsistency are independently and uniformly
     distributed.  If a terminal window contains W words in a nominally L
     word story, its inclusion probability is
     r = 1 - (1 - min(W/L, 1)) ** 2, and
     CED_new = error_count / (r * L) * 10000.
   - Full-narrative evaluation does not need exposure correction, so its new
     and original CED are identical.

2. GRR (Group Relative Rank)
   - Average rank across all stories (group-relative comparison)
   - Lower is better
   - For each story, models are ranked by efficiency = word_count / (1 + error_count)
   - GRR = mean(rank_i) across all stories

Usage:
    python -m constory.metrics \
        --eval-dir output/ \
        --mode both
"""

import os
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tabulate import tabulate


# =============================================================================
# Evaluation Criteria (5 categories, 19 subtypes)
# =============================================================================

EVALUATION_CRITERIA = {
    "characterization": {
        "name": "Character Consistency",
        "columns": [
            "characterization_memory_contradictions",
            "characterization_knowledge_contradictions",
            "characterization_skill_power_fluctuations",
            "characterization_forgotten_abilities",
        ],
    },
    "factual_detail": {
        "name": "Factual & Detail Consistency",
        "columns": [
            "factual_detail_appearance_mismatches",
            "factual_detail_nomenclature_confusions",
            "factual_detail_quantitative_mismatches",
        ],
    },
    "narrative_style": {
        "name": "Narrative & Style",
        "columns": [
            "narrative_style_perspective_confusions",
            "narrative_style_tone_inconsistencies",
            "narrative_style_style_shifts",
        ],
    },
    "timeline_plot": {
        "name": "Timeline & Plot Logic",
        "columns": [
            "timeline_plot_absolute_time_contradictions",
            "timeline_plot_duration_timeline_contradictions",
            "timeline_plot_simultaneity_contradictions",
            "timeline_plot_causeless_effects",
            "timeline_plot_causal_logic_violations",
            "timeline_plot_abandoned_plot_elements",
        ],
    },
    "world_building": {
        "name": "World-building & Setting",
        "columns": [
            "world_building_core_rules_violations",
            "world_building_social_norms_violations",
            "world_building_geographical_contradictions",
        ],
    },
}

ALL_ERROR_COLUMNS = []
for cfg in EVALUATION_CRITERIA.values():
    ALL_ERROR_COLUMNS.extend(cfg["columns"])

TASK_TYPES = ["generation", "continuation", "expansion", "completion"]


# =============================================================================
# Helper Functions
# =============================================================================

import re

# CJK Unicode ranges (Chinese/Japanese/Korean ideographs + kana).
_CJK_PATTERN = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]"
)


def count_words(text: str) -> int:
    """Language-aware word count.

    English (and other space-delimited scripts) are counted by whitespace
    tokens; CJK characters are counted individually. `str.split()` alone
    undercounts CJK by ~6x because CJK text has no spaces, which inflates
    CED for Chinese stories. This mixes both so a Chinese and an English
    story of comparable content yield comparable word counts.
    """
    if not isinstance(text, str) or not text:
        return 0
    cjk = len(_CJK_PATTERN.findall(text))
    # Non-CJK words: strip CJK chars first, then split on whitespace.
    non_cjk = len(_CJK_PATTERN.sub(" ", text).split())
    return cjk + non_cjk


def resolve_evaluated_word_count(
    row: pd.Series,
    story_column: str,
) -> Tuple[float, str]:
    """Return the CED denominator matching the scope the errors come from.

    The long-form evaluation stores ``checked_words`` in the CSV: the full
    story length at 10k, and the length of the checked final chapters at
    20k/50k/100k. The CED numerator counts errors in that checked window, so
    the window's length takes precedence. Only when the field is absent does
    this fall back to recounting the full ``generated_story``.
    """
    checked_words = row.get("checked_words")
    if not pd.isna(checked_words):
        try:
            value = float(checked_words)
        except (TypeError, ValueError):
            value = 0.0
        if np.isfinite(value) and value > 0:
            return value, "checked_words"

    text = row.get(story_column)
    if pd.isna(text) or not isinstance(text, str):
        return 0.0, "unavailable"
    return float(count_words(text)), story_column


def check_error_exists(cell_value) -> bool:
    """Check whether a cell contains at least one error (has exact_quote)."""
    if pd.isna(cell_value):
        return False
    s = str(cell_value).strip()
    if not s or s.lower() == "none":
        return False
    return "exact_quote" in s.lower()


# Must match judge.PARSE_FAILED. A cell holding this means the judge
# response could not be parsed (malformed/refusal/empty) — NOT a zero-error
# verdict. Such cells are excluded from CED rather than counted as zero.
PARSE_FAILED = "PARSE_FAILED"


def cell_failed(cell_value) -> bool:
    """True if this cell is a parse-failure sentinel (verdict unavailable)."""
    if pd.isna(cell_value):
        return True
    value = str(cell_value).strip()
    return value == PARSE_FAILED or value.startswith("ERROR:")


def count_errors_in_cell(cell_value) -> int:
    """Count the number of error instances in a cell (JSON array length)."""
    if pd.isna(cell_value):
        return 0
    s = str(cell_value).strip()
    if not s or s.lower() == "none" or "exact_quote" not in s.lower():
        return 0
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return len(arr)
    except json.JSONDecodeError:
        pass
    return 1 if "exact_quote" in s.lower() else 0


def count_error_subtype(cell_value) -> int:
    """Official released-code semantics: at most one error per subtype."""
    return 1 if check_error_exists(cell_value) else 0


# =============================================================================
# CED: Consistency Error Density
# =============================================================================

def compute_ced_single(
    eval_path: str,
    story_column: str,
    model_name: str,
    nominal_story_words: Optional[float] = None,
    inconsistency_pair_divisor: float = 1.0,
) -> Optional[Dict]:
    """
    Compute CED (Consistency Error Density) for a single model.

    Returns dict with:
        Both released-code CED (subtype presence) and instance-count CED.
    """
    if inconsistency_pair_divisor != 1.0:
        raise ValueError(
            "inconsistency_pair_divisor must be 1; report unadjusted "
            "CED_original and use nominal_story_words for the independent-"
            "uniform-endpoints CED_new"
        )
    if nominal_story_words is not None and nominal_story_words <= 0:
        raise ValueError("nominal_story_words must be positive")

    if not os.path.exists(eval_path):
        print(f"  [WARN] {model_name}: file not found - {eval_path}")
        return None

    try:
        df = pd.read_csv(eval_path)
    except Exception as e:
        print(f"  [ERROR] {model_name}: {e}")
        return None

    if story_column not in df.columns:
        print(f"  [WARN] {model_name}: column '{story_column}' not found")
        return None

    story_densities_original_official = []
    story_densities_original_instance = []
    story_densities_new_official = []
    story_densities_new_instance = []
    total_errors_official = 0
    total_errors_instance = 0
    total_words = 0
    cat_errors_official = {cat: 0 for cat in EVALUATION_CRITERIA}
    cat_errors_instance = {cat: 0 for cat in EVALUATION_CRITERIA}
    cat_words = {cat: 0 for cat in EVALUATION_CRITERIA}
    cat_exposure_words = {cat: 0 for cat in EVALUATION_CRITERIA}
    denominator_sources = defaultdict(int)
    exposure_probabilities = []
    # A criteria fails as a unit (all its columns become PARSE_FAILED
    # together). Track how many stories we had to exclude.
    stories_excluded = 0          # dropped from overall CED (any category failed)
    cat_failed = {cat: 0 for cat in EVALUATION_CRITERIA}

    for idx in range(len(df)):
        row = df.iloc[idx]
        wc, denominator_source = resolve_evaluated_word_count(
            row, story_column
        )
        if wc == 0:
            continue

        scope_mode = str(row.get("scope_mode", "")).strip()
        is_terminal_scope = scope_mode == "target_ending_chapters"
        if is_terminal_scope and nominal_story_words is not None:
            checked_fraction = min(wc / nominal_story_words, 1.0)
            exposure_probability = 1.0 - (1.0 - checked_fraction) ** 2
            exposure_words = exposure_probability * nominal_story_words
        else:
            # The full narrative was checked, so every inconsistency endpoint
            # was exposed.  No sampling correction is needed.
            exposure_probability = 1.0
            exposure_words = wc
        # Which categories have an unusable (parse-failed) verdict?
        cat_ok = {}
        for cat, cfg in EVALUATION_CRITERIA.items():
            failed = any(
                col not in df.columns or cell_failed(row[col])
                for col in cfg["columns"]
            )
            cat_ok[cat] = not failed
            if failed:
                cat_failed[cat] += 1

        # Per-category CED: only accumulate categories with a valid verdict.
        for cat, cfg in EVALUATION_CRITERIA.items():
            if not cat_ok[cat]:
                continue
            for col in cfg["columns"]:
                if col in df.columns:
                    cat_errors_official[cat] += count_error_subtype(row[col])
                    cat_errors_instance[cat] += count_errors_in_cell(row[col])
            cat_words[cat] += wc
            cat_exposure_words[cat] += exposure_words

        # Overall CED needs all 5 categories intact, otherwise the per-story
        # total is missing an unknown number of errors. Exclude such stories
        # rather than undercounting them as clean.
        if not all(cat_ok.values()):
            stories_excluded += 1
            continue
        exposure_probabilities.append(exposure_probability)

        ec_official = sum(
            count_error_subtype(row[col])
            for col in ALL_ERROR_COLUMNS
            if col in df.columns
        )
        ec_instance = sum(
            count_errors_in_cell(row[col])
            for col in ALL_ERROR_COLUMNS
            if col in df.columns
        )

        original_scale = wc / 10000
        new_scale = exposure_words / 10000
        story_densities_original_official.append(ec_official / original_scale)
        story_densities_original_instance.append(ec_instance / original_scale)
        story_densities_new_official.append(ec_official / new_scale)
        story_densities_new_instance.append(ec_instance / new_scale)
        total_errors_official += ec_official
        total_errors_instance += ec_instance
        total_words += wc
        denominator_sources[denominator_source] += 1

    if not story_densities_original_official:
        print(
            f"  [WARN] {model_name}: no fully-parsed stories "
            f"({stories_excluded} excluded due to parse failures)"
        )
        return None

    if stories_excluded:
        print(
            f"  [WARN] {model_name}: excluded {stories_excluded} stories from "
            f"overall CED due to judge parse failures"
        )

    cat_densities_official = {}
    cat_densities_instance = {}
    for cat in EVALUATION_CRITERIA:
        if cat_words[cat] > 0:
            scale = cat_words[cat] / 10000
            cat_densities_official[cat] = cat_errors_official[cat] / scale
            cat_densities_instance[cat] = cat_errors_instance[cat] / scale
        else:
            cat_densities_official[cat] = 0.0
            cat_densities_instance[cat] = 0.0

    cat_densities_new_official = {}
    cat_densities_new_instance = {}
    for cat in EVALUATION_CRITERIA:
        if cat_exposure_words[cat] > 0:
            scale = cat_exposure_words[cat] / 10000
            cat_densities_new_official[cat] = (
                cat_errors_official[cat] / scale
            )
            cat_densities_new_instance[cat] = (
                cat_errors_instance[cat] / scale
            )
        else:
            cat_densities_new_official[cat] = 0.0
            cat_densities_new_instance[cat] = 0.0

    result = {
        "model_name": model_name,
        # Unqualified density fields report CED_new with subtype-presence
        # counting.  The explicit original/new fields below remove ambiguity.
        "avg_density": np.mean(story_densities_new_official),
        "median_density": np.median(story_densities_new_official),
        "std_density": np.std(story_densities_new_official),
        "avg_errors": total_errors_official / len(
            story_densities_new_official
        ),
        "category_densities": cat_densities_new_official,
        "avg_density_official": np.mean(story_densities_new_official),
        "median_density_official": np.median(story_densities_new_official),
        "std_density_official": np.std(story_densities_new_official),
        "avg_errors_official": (
            total_errors_official / len(story_densities_new_official)
        ),
        "category_densities_official": cat_densities_new_official,
        "avg_density_instance": np.mean(story_densities_new_instance),
        "median_density_instance": np.median(story_densities_new_instance),
        "std_density_instance": np.std(story_densities_new_instance),
        "avg_errors_instance": (
            total_errors_instance / len(story_densities_new_instance)
        ),
        "category_densities_instance": cat_densities_new_instance,
        "avg_density_original_official": np.mean(
            story_densities_original_official
        ),
        "median_density_original_official": np.median(
            story_densities_original_official
        ),
        "std_density_original_official": np.std(
            story_densities_original_official
        ),
        "avg_density_original_instance": np.mean(
            story_densities_original_instance
        ),
        "median_density_original_instance": np.median(
            story_densities_original_instance
        ),
        "std_density_original_instance": np.std(
            story_densities_original_instance
        ),
        "category_densities_original_official": cat_densities_official,
        "category_densities_original_instance": cat_densities_instance,
        "avg_density_new_official": np.mean(story_densities_new_official),
        "median_density_new_official": np.median(story_densities_new_official),
        "std_density_new_official": np.std(story_densities_new_official),
        "avg_density_new_instance": np.mean(story_densities_new_instance),
        "median_density_new_instance": np.median(story_densities_new_instance),
        "std_density_new_instance": np.std(story_densities_new_instance),
        "category_densities_new_official": cat_densities_new_official,
        "category_densities_new_instance": cat_densities_new_instance,
        "avg_words": total_words / len(story_densities_new_official),
        "nominal_story_words": nominal_story_words,
        "mean_endpoint_coverage_probability": np.mean(
            exposure_probabilities
        ),
        "ced_original_formula": "error_count / checked_words * 10000",
        "ced_new_formula": (
            "error_count / ((1-(1-min(checked_words/L,1))^2)*L) "
            "* 10000 for terminal scope; CED_original for full narrative"
        ),
        "word_denominator": (
            "checked_words"
            if denominator_sources.get("checked_words")
            == len(story_densities_new_official)
            else "mixed_or_story_column"
        ),
        "total_stories": len(story_densities_new_official),
        "stories_excluded": stories_excluded,
        "category_failed": cat_failed,
    }
    return result


def compute_ced(
    model_configs: Dict[str, Tuple[str, str]],
    eval_dir: str,
    max_workers: int = 8,
    nominal_story_words: Optional[float] = None,
    inconsistency_pair_divisor: float = 1.0,
) -> List[Dict]:
    """
    Compute CED for multiple models in parallel.

    Args:
        model_configs: {model_name: (eval_filename, story_column)}
        eval_dir: directory containing evaluation CSV files
        max_workers: thread pool size

    Returns:
        List of result dicts, sorted by avg_density (ascending).
    """
    print("Computing CED (Consistency Error Density)...")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                compute_ced_single,
                os.path.join(eval_dir, fname),
                scol,
                mname,
                nominal_story_words=nominal_story_words,
                inconsistency_pair_divisor=inconsistency_pair_divisor,
            ): mname
            for mname, (fname, scol) in model_configs.items()
        }
        for fut in as_completed(futures):
            mname = futures[fut]
            try:
                r = fut.result()
                if r:
                    results.append(r)
                    print(
                        f"  [OK] {mname}: "
                        f"CED-official={r['avg_density_official']:.3f}, "
                        f"CED-instance={r['avg_density_instance']:.3f} "
                        f"({r['total_stories']} stories)"
                    )
            except Exception as e:
                print(f"  [ERROR] {mname}: {e}")

    results.sort(key=lambda x: x["avg_density_official"])
    return results


# =============================================================================
# GRR: Group Relative Rank
# =============================================================================

def load_model_data_for_grr(
    eval_path: str,
    story_column: str,
) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Load per-story (word_count, error_count) for GRR computation.

    Returns:
        {story_id: (word_count, error_count)}
    """
    if not os.path.exists(eval_path):
        return None

    try:
        df = pd.read_csv(eval_path)
    except Exception:
        return None

    if story_column not in df.columns or "id" not in df.columns:
        return None

    data = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        sid = row["id"]
        text = row[story_column]
        if pd.isna(text) or not isinstance(text, str):
            continue
        wc = count_words(text)
        if wc == 0:
            continue
        # Skip stories with any parse-failed category: the error total would
        # be incomplete and would unfairly inflate this model's efficiency.
        if any(
            col not in df.columns or cell_failed(row[col])
            for col in ALL_ERROR_COLUMNS
        ):
            continue
        ec = sum(
            count_error_subtype(row[col])
            for col in ALL_ERROR_COLUMNS
            if col in df.columns
        )
        data[sid] = (wc, ec)

    return data if data else None


def compute_grr_from_data(
    all_model_data: Dict[str, Dict[int, Tuple[int, int]]],
) -> Dict[str, float]:
    """
    Compute GRR (Group Relative Rank) across all stories.

    For each story:
        efficiency = word_count / (1 + error_count)
        rank models by efficiency (descending), lower rank = better

    GRR = mean(rank) across all stories. Lower is better.
    """
    all_ids = set()
    for sd in all_model_data.values():
        if sd is not None:
            all_ids.update(sd.keys())

    model_ranks = defaultdict(list)

    for sid in sorted(all_ids):
        group = {}
        for mn, sd in all_model_data.items():
            if sd is not None and sid in sd:
                wc, ec = sd[sid]
                group[mn] = wc / (1 + ec)

        if len(group) < 2:
            continue

        eff = pd.Series(group)
        ranks = eff.rank(ascending=False, method="min")
        for mn, r in ranks.items():
            model_ranks[mn].append(r)

    grr = {}
    for mn in all_model_data:
        if mn in model_ranks and model_ranks[mn]:
            grr[mn] = np.mean(model_ranks[mn])
        else:
            grr[mn] = np.nan
    return grr


def compute_grr(
    model_configs: Dict[str, Tuple[str, str]],
    eval_dir: str,
    max_workers: int = 8,
) -> Dict[str, float]:
    """
    Compute GRR for multiple models.

    Args:
        model_configs: {model_name: (eval_filename, story_column)}
        eval_dir: directory containing evaluation CSV files

    Returns:
        {model_name: grr_value}, sorted ascending.
    """
    print("Computing GRR (Group Relative Rank)...")

    all_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                load_model_data_for_grr,
                os.path.join(eval_dir, fname),
                scol,
            ): mname
            for mname, (fname, scol) in model_configs.items()
        }
        for fut in as_completed(futures):
            mname = futures[fut]
            try:
                d = fut.result()
                if d:
                    all_data[mname] = d
                    print(f"  [OK] {mname}: {len(d)} stories loaded")
                else:
                    print(f"  [WARN] {mname}: no data")
            except Exception as e:
                print(f"  [ERROR] {mname}: {e}")

    grr = compute_grr_from_data(all_data)
    return dict(sorted(grr.items(), key=lambda x: x[1]))


# =============================================================================
# Display Helpers
# =============================================================================

def print_ced_table(results: List[Dict]):
    """Print unadjusted original CED and endpoint-exposure CED_new."""
    headers = [
        "Rank", "Model",
        "Original\nSubtype", "Original\nInstance",
        "New\nSubtype", "New\nInstance",
        "Mean r", "Avg checked\nwords", "Scored", "Excl.",
    ]
    rows = []
    for rank, r in enumerate(results, 1):
        rows.append([
            rank, r["model_name"],
            f"{r['avg_density_original_official']:.3f}",
            f"{r['avg_density_original_instance']:.3f}",
            f"{r['avg_density_new_official']:.3f}",
            f"{r['avg_density_new_instance']:.3f}",
            f"{r['mean_endpoint_coverage_probability']:.4f}",
            f"{r['avg_words']:,.0f}",
            r["total_stories"],
            r.get("stories_excluded", 0),
        ])
    print("\n" + "=" * 100)
    print(
        "CED Leaderboard (Original=unadjusted terminal density; "
        "New=independent-uniform-endpoint exposure correction)"
    )
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid",
                   stralign="center", numalign="center"))


def print_grr_table(grr: Dict[str, float], model_info: Optional[Dict] = None):
    """Print a formatted GRR leaderboard table."""
    headers = ["Rank", "Model", "GRR Overall"]
    rows = []
    for rank, (mn, val) in enumerate(grr.items(), 1):
        rows.append([rank, mn, f"{val:.2f}" if not np.isnan(val) else "N/A"])
    print("\n" + "=" * 70)
    print("GRR Leaderboard (Group Relative Rank, lower is better)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid",
                   stralign="center", numalign="center"))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ConStory-Bench Metrics: Compute CED and/or GRR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute both CED and GRR
  python -m constory.metrics --eval-dir output/ --config configs/models.yaml

  # Compute CED only for a single model
  python -m constory.metrics --eval-dir output/ --mode ced \\
      --eval-file judge_gpt4o.csv --story-column generated_story --model-name gpt4o
        """,
    )
    parser.add_argument("--eval-dir", required=True, help="Evaluation CSV directory")
    parser.add_argument(
        "--mode",
        choices=["ced", "grr", "both"],
        default="both",
        help="Which metric(s) to compute",
    )
    parser.add_argument("--config", help="YAML config with model definitions")
    parser.add_argument("--eval-file", help="Single eval CSV filename")
    parser.add_argument("--story-column", help="Story column name (single model)")
    parser.add_argument("--model-name", help="Model name (single model)")
    parser.add_argument("--output", help="Output CSV path for results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--nominal-story-words",
        type=float,
        help=(
            "Nominal full-story length L for terminal-scope exposure "
            "correction (for example 20000, 50000, or 100000)."
        ),
    )
    parser.add_argument(
        "--inconsistency-pair-divisor",
        type=float,
        default=1.0,
        help=(
            "Only 1 is accepted. Use --nominal-story-words for CED_new."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build model configs
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_configs = {
            m["name"]: (m["eval_file"], m["story_column"])
            for m in cfg["models"]
        }
    elif args.eval_file and args.story_column and args.model_name:
        model_configs = {
            args.model_name: (args.eval_file, args.story_column)
        }
    else:
        raise ValueError(
            "Provide either --config or (--eval-file, --story-column, --model-name)"
        )

    print("=" * 70)
    print("ConStory-Bench Metrics")
    print("=" * 70)
    print(f"  Eval dir:  {args.eval_dir}")
    print(f"  Mode:      {args.mode}")
    print(f"  Models:    {len(model_configs)}")
    print("=" * 70)

    if args.mode in ("ced", "both"):
        ced_results = compute_ced(
            model_configs,
            args.eval_dir,
            max_workers=args.workers,
            nominal_story_words=args.nominal_story_words,
            inconsistency_pair_divisor=args.inconsistency_pair_divisor,
        )
        print_ced_table(ced_results)

        if args.output:
            rows = []
            for r in ced_results:
                row = {
                    "model_name": r["model_name"],
                    # Unqualified ced_* columns use subtype-presence counting.
                    "ced_overall": r["avg_density_official"],
                    "ced_median": r["median_density_official"],
                    "ced_std": r["std_density_official"],
                    "ced_official_overall": r["avg_density_official"],
                    "ced_official_median": r["median_density_official"],
                    "ced_official_std": r["std_density_official"],
                    "ced_instance_overall": r["avg_density_instance"],
                    "ced_instance_median": r["median_density_instance"],
                    "ced_instance_std": r["std_density_instance"],
                    "ced_original_subtype_overall": r[
                        "avg_density_original_official"
                    ],
                    "ced_original_subtype_median": r[
                        "median_density_original_official"
                    ],
                    "ced_original_subtype_std": r[
                        "std_density_original_official"
                    ],
                    "ced_original_instance_overall": r[
                        "avg_density_original_instance"
                    ],
                    "ced_original_instance_median": r[
                        "median_density_original_instance"
                    ],
                    "ced_original_instance_std": r[
                        "std_density_original_instance"
                    ],
                    "ced_new_subtype_overall": r[
                        "avg_density_new_official"
                    ],
                    "ced_new_subtype_median": r[
                        "median_density_new_official"
                    ],
                    "ced_new_subtype_std": r[
                        "std_density_new_official"
                    ],
                    "ced_new_instance_overall": r[
                        "avg_density_new_instance"
                    ],
                    "ced_new_instance_median": r[
                        "median_density_new_instance"
                    ],
                    "ced_new_instance_std": r[
                        "std_density_new_instance"
                    ],
                    "avg_errors_official": r["avg_errors_official"],
                    "avg_errors_instance": r["avg_errors_instance"],
                    "avg_words": r["avg_words"],
                    "nominal_story_words": r["nominal_story_words"],
                    "mean_endpoint_coverage_probability": r[
                        "mean_endpoint_coverage_probability"
                    ],
                    "ced_original_formula": r["ced_original_formula"],
                    "ced_new_formula": r["ced_new_formula"],
                    "word_denominator": r["word_denominator"],
                    "total_stories": r["total_stories"],
                    "stories_excluded": r["stories_excluded"],
                }
                for cat, val in r["category_densities_official"].items():
                    row[f"ced_{cat}"] = val
                    row[f"ced_official_{cat}"] = val
                for cat, val in r["category_densities_instance"].items():
                    row[f"ced_instance_{cat}"] = val
                for cat, val in r[
                    "category_densities_original_official"
                ].items():
                    row[f"ced_original_subtype_{cat}"] = val
                for cat, val in r[
                    "category_densities_original_instance"
                ].items():
                    row[f"ced_original_instance_{cat}"] = val
                for cat, val in r[
                    "category_densities_new_official"
                ].items():
                    row[f"ced_new_subtype_{cat}"] = val
                for cat, val in r[
                    "category_densities_new_instance"
                ].items():
                    row[f"ced_new_instance_{cat}"] = val
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                args.output.replace(".csv", "_ced.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    if args.mode in ("grr", "both"):
        grr = compute_grr(
            model_configs, args.eval_dir, max_workers=args.workers
        )
        print_grr_table(grr)

        if args.output:
            rows = [
                {"model_name": mn, "grr": val}
                for mn, val in grr.items()
            ]
            pd.DataFrame(rows).to_csv(
                args.output.replace(".csv", "_grr.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
