"""Shared 10k-generation and completeness helpers for the additional baselines.

This module only standardizes the experimental protocol; it implements no
narrative memory or planning algorithm. Each method's own pipeline still
decides how to plan, segment, compress and revise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1] / "agent"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation_common import count_words, word_bounds  # noqa: E402


def strict_length(text: str, target_words: int) -> Tuple[bool, int, int, int]:
    """Check the full text against the ±20% bound used in the main experiments."""
    actual = count_words(text or "")
    minimum, maximum = word_bounds(target_words)
    return minimum <= actual <= maximum, actual, minimum, maximum


def load_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_completed_ids(path: str, target_words: int) -> set:
    """Treat a record as completed only if it is complete and still passes a
    fresh word count."""
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            ok, _, _, _ = strict_length(item.get("story") or "", target_words)
            if item.get("complete") is True and ok:
                completed.add(item.get("id"))
    return completed


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def ordered_records(
        prompts: Iterable[Dict[str, Any]],
        records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate records in prompt order, keeping the last record per ID."""
    latest = {item["id"]: item for item in records}
    return [latest[prompt["id"]] for prompt in prompts if prompt["id"] in latest]
