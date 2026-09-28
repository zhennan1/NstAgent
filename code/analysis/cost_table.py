#!/usr/bin/env python3
"""Recompute the generation-cost table of the paper (Table 4) from the per-call usage logs.

Usage: python code/analysis/cost_table.py [package_root]

Each story in results/usage/deepseek/<arm>/<id>.jsonl was generated with usage_logger.py, which writes
one line per chat.completions call with the provider's usage object. A story's cost is the sum over all
of its calls, including length retries, rejected drafts, tool turns, summaries, memory calls, and
automatic resumptions. The table reports the mean per completed story; the mean over every finished
story, incomplete ones included, is given in parentheses where the two differ.

Prices are DeepSeek-V4-Flash off-peak rates in USD per million tokens: uncached input 0.22, cached
input 0.007, output (reasoning included) 0.66. Generation only; judging is not included.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
USAGE = ROOT / "results/usage/deepseek"
OUT = ROOT / "results/tables/cost.md"
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.22, 0.007, 0.66
ORDER = ["direct_10k", "storywriter_10k", "dome_10k", "rollsum_10k", "rollsum_20k", "rollsum_50k",
         "rollsum_100k", "nstagent_10k", "nstagent_20k", "nstagent_50k", "nstagent_100k"]
LABEL = {"direct": "Direct", "storywriter": "StoryWriter", "dome": "DOME", "rollsum": "RollSum",
         "nstagent": "NstAgent"}


def story_usage(path):
    totals = dict(calls=0, errors=0, prompt=0, cache_hit=0, cache_miss=0, completion=0, reasoning=0,
                  thinking_disabled_calls=0)
    for line in open(path, encoding="utf-8"):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        totals["calls"] += 1
        totals["thinking_disabled_calls"] += bool(entry.get("thinking_disabled"))
        if entry.get("error"):
            totals["errors"] += 1
            continue
        usage = entry.get("usage")
        if not usage:
            continue
        prompt = usage.get("prompt_tokens") or 0
        hit = usage.get("prompt_cache_hit_tokens")
        if hit is None:
            hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        totals["prompt"] += prompt
        totals["cache_hit"] += hit
        totals["cache_miss"] += usage.get("prompt_cache_miss_tokens", prompt - hit)
        totals["completion"] += usage.get("completion_tokens") or 0
        totals["reasoning"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    totals["cost_usd"] = (totals["cache_miss"] * PRICE_IN + totals["cache_hit"] * PRICE_CACHED
                          + totals["completion"] * PRICE_OUT) / 1e6
    return totals


status = {(r["arm"], r["id"]): r for r in map(json.loads, open(USAGE / "status.jsonl", encoding="utf-8"))}


def cell(key, complete, finished, scale=1.0, fmt="{:,.0f}"):
    main = fmt.format(st.fmean(s[key] for s in complete) / scale)
    if len(finished) != len(complete):
        main += f" ({fmt.format(st.fmean(s[key] for s in finished) / scale)})"
    return main


md = ["# Generation cost per story on DeepSeek-V4-Flash", "",
      "Recomputed by `code/analysis/cost_table.py` from the per-call usage logs in `results/usage/deepseek/`.",
      "Sample: the first five prompts of `data/prompts_en.jsonl` (ids 1994, 1996, 1993, 1712, 1814), generated",
      "with the settings of the main runs and one story per job. Each cell is the mean per completed story; where",
      "some stories did not complete, the mean over all finished stories follows in parentheses. Output tokens",
      "include reasoning tokens. Cost uses off-peak prices in USD per million tokens: "
      f"{PRICE_IN} for uncached input, {PRICE_CACHED} for cached input, {PRICE_OUT} for output.", "",
      "| Method | Length | Complete/finished | Calls | Input tokens | Cached input | Output tokens | "
      "Reasoning tokens | Minutes | Cost (USD) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for arm in ORDER:
    stories = []
    for path in sorted((USAGE / arm).glob("*.jsonl")):
        info = status[(arm, int(path.stem))]
        stories.append(dict(**story_usage(path), complete=info["complete"], minutes=info["elapsed_s"]))
    complete = [s for s in stories if s["complete"]]
    method, length = arm.rsplit("_", 1)
    md.append(f"| {LABEL[method]} | {length.upper()} | {len(complete)}/{len(stories)} | "
              f"{cell('calls', complete, stories)} | {cell('prompt', complete, stories)} | "
              f"{cell('cache_hit', complete, stories)} | {cell('completion', complete, stories)} | "
              f"{cell('reasoning', complete, stories)} | {cell('minutes', complete, stories, 60, '{:,.1f}')} | "
              f"{cell('cost_usd', complete, stories, fmt='{:,.2f}')} |")
md += ["", "Notes:", "",
       "- DOME keeps its released configuration: format conversion, the five-act and hierarchical outlines, and the",
       "  knowledge-graph memory calls run with reasoning disabled, and chapter writing with reasoning enabled.",
       "  Every other method uses the provider's default, reasoning enabled.",
       "- Five stories per arm, so these are rough estimates. The price per 10K words written by NstAgent is the",
       "  cost divided by the target length in units of 10K words.", ""]
OUT.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
