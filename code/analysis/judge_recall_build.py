#!/usr/bin/env python3
"""Build the judge-recall calibration set: known contradictions at controlled prefix depths.

Tests whether a lower CED at 100K could reflect the judge losing recall over a ~90K-word reference
prefix rather than better stories. For every length we inject a pair of explicit, self-contained
statements into a real NstAgent story: the later half always lands inside the terminal window (so the
evaluation protocol is obliged to report it), and the earlier half is placed at one of three prefix
depths. Detection is then measured as a function of how far back the antecedent sits.

The injected sentence pairs come verbatim from ``pair_text`` in `recall_injection.py`, shipped next
to this script.

Usage: python judge_recall_build.py --out-dir <dir> [--per-cell 2] [--seed 20260922]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path

# Paths are relative to the repository root; the stories are the shipped NstAgent arms.
ROOT = Path(__file__).resolve().parents[2]
OLD = Path(__file__).resolve().parent / "recall_injection.py"
# (label, source story file, ending-window chapters; 0 = whole-narrative marker as used at 10K)
LENGTHS = [
    ("10k", "results/stories/deepseek/nstagent_10k.jsonl", 0),
    ("20k", "results/stories/deepseek/nstagent_20k.jsonl", 7),
    ("50k", "results/stories/deepseek/nstagent_50k.jsonl", 5),
    ("100k", "results/stories/deepseek/nstagent_100k.jsonl", 4),
]
POOL = json.loads((ROOT / "results/tables/judge_recall_pool.json").read_text(encoding="utf-8"))["pool"]
CATEGORIES = ["characterization", "factual_detail", "narrative_style", "timeline_plot", "world_building"]
DEPTHS = ["far", "mid", "near"]  # antecedent position relative to the start of the evaluation window

spec = importlib.util.spec_from_file_location("old_injection", OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
pair_text = old.pair_text


def subject_of(record):
    """Best-effort protagonist name, so the injected lines name a real character."""
    state = record.get("final_state") or {}
    chars = state.get("character_states") or []
    for ch in chars:
        name = (ch or {}).get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return "the protagonist"


def window_start(n_chapters, ending):
    """Index of the first in-scope chapter. At 10K the whole story is in scope."""
    return 0 if ending == 0 else max(0, n_chapters - ending)


def placement(n_chapters, ending, depth):
    """Return (antecedent chapter, manifestation chapter).

    The manifestation always sits inside the evaluation window; at 10K, where everything is in scope,
    it sits in the final chapter so the antecedent-to-manifestation distance stays comparable.
    """
    start = window_start(n_chapters, ending)
    manifestation = n_chapters - 1 if ending == 0 else min(n_chapters - 1, start + max(1, ending // 2))
    prefix_end = (manifestation - 1) if ending == 0 else (start - 1)
    prefix_end = max(0, prefix_end)
    if depth == "near":
        antecedent = prefix_end
    elif depth == "mid":
        antecedent = max(0, prefix_end // 2)
    else:
        antecedent = 0
    if antecedent >= manifestation:
        antecedent = max(0, manifestation - 1)
    return antecedent, manifestation


def words(text):
    return len(str(text).split())


def inject(record, token, category, positive, depth, ending):
    out = copy.deepcopy(record)
    chapters = out["story"]
    n = len(chapters)
    a_idx, b_idx = placement(n, ending, depth)
    quote_a, quote_b = pair_text(category, token, positive, out.get("language", "en"), subject_of(out))
    chapters[a_idx]["content"] = f"{quote_a}\n\n{chapters[a_idx]['content']}"
    chapters[b_idx]["content"] = f"{chapters[b_idx]['content']}\n\n{quote_b}"
    start = window_start(n, ending)
    meta = dict(
        story_id=out["id"], token=token, category=category, positive=positive, depth=depth,
        chapters=n, antecedent_chapter=a_idx, manifestation_chapter=b_idx,
        window_start_chapter=start, ending_chapters=ending,
        manifestation_in_window=bool(b_idx >= start),
        gap_chapters=b_idx - a_idx,
        prefix_words_before_antecedent=sum(words(ch["content"]) for ch in chapters[:a_idx]),
        gap_words=sum(words(ch["content"]) for ch in chapters[a_idx + 1:b_idx]),
        total_words=sum(words(ch["content"]) for ch in chapters),
    )
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-cell", type=int, default=2, help="positive stories per (category, depth) cell")
    ap.add_argument("--controls", type=int, default=6, help="consistent control stories per length")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)

    manifest = []
    for label, rel, ending in LENGTHS:
        stories = [json.loads(line) for line in open(ROOT / rel, encoding="utf-8") if line.strip()]
        by_id = {s["id"]: s for s in stories if s.get("complete") and isinstance(s.get("story"), list)
                 and len(s["story"]) >= 4}
        # The package ships a subset of the 50K and 100K stories. The shuffle permutes the candidate list
        # of the full arm, recorded in judge_recall_pool.json, so the selection is the one of the study.
        pool = list(POOL[label])
        rng = random.Random(f"{args.seed}|{label}")
        rng.shuffle(pool)
        stories = [by_id[i] for i in pool if i in by_id]
        cells = [(c, d) for c in CATEGORIES for d in DEPTHS for _ in range(args.per_cell)]
        need = len(cells) + args.controls
        if len(stories) < need:
            raise SystemExit(f"{label}: need {need} stories, have {len(stories)}")
        rows = []
        for (category, depth), story in zip(cells, stories):
            token = old.stable_token(f"recall_{label}", story["id"], category, f"{depth}_pos")
            rec, meta = inject(story, token, category, True, depth, ending)
            rows.append(rec)
            manifest.append(dict(length=label, **meta))
        for i, story in enumerate(stories[len(cells):len(cells) + args.controls]):
            category = CATEGORIES[i % len(CATEGORIES)]
            depth = DEPTHS[i % len(DEPTHS)]
            token = old.stable_token(f"recall_{label}", story["id"], category, f"{depth}_ctl")
            rec, meta = inject(story, token, category, False, depth, ending)
            rows.append(rec)
            manifest.append(dict(length=label, **meta))
        path = out_dir / "inputs" / f"recall_{label}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{label}: {len(rows)} stories -> {path}")

    # Every token must appear exactly twice in its own story and nowhere else.
    problems = []
    for label, _, _ in LENGTHS:
        rows = [json.loads(l) for l in open(out_dir / "inputs" / f"recall_{label}.jsonl", encoding="utf-8")]
        by_id = {r["id"]: "\n".join(ch["content"] for ch in r["story"]) for r in rows}
        for m in [m for m in manifest if m["length"] == label]:
            n = by_id[m["story_id"]].count(m["token"])
            if n != 2:
                problems.append(f"{label}/{m['story_id']}/{m['token']}: {n} occurrences")
            if m["ending_chapters"] and not m["manifestation_in_window"]:
                problems.append(f"{label}/{m['story_id']}: manifestation outside window")
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"manifest: {len(manifest)} injections")
    print("checks:", "OK" if not problems else "\n  " + "\n  ".join(problems))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
