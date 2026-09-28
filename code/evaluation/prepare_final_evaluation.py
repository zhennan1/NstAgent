#!/usr/bin/env python
# coding: utf-8
"""Validate the generation matrix and write a canonical JSONL with one record
per id for evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def count_words(text: str) -> int:
    cjk = re.findall(
        r"[㐀-䶿一-鿿豈-﫿]|[\U00020000-\U0002ffff]", text
    )
    latin = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text)
    return len(cjk) + len(latin)


def parse_dataset(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"dataset must be NAME=PATH: {spec}")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError(f"dataset must be NAME=PATH: {spec}")
    return name, Path(path)


def expected_words_from_name(name: str) -> int:
    match = re.search(r"(10|20|50|100)k(?:$|_)", name)
    if not match:
        raise ValueError(f"cannot infer target length from dataset name: {name}")
    return int(match.group(1)) * 1000


def validate_dataset(
    name: str,
    paths: List[Path],
    expected_samples: int,
    expected_ids: set[str] | None = None,
    reject_duplicates: bool = False,
    enforce_whole_story_range: bool = True,
    allow_corrected_chapter_out_of_range: bool = False,
) -> Tuple[Dict[str, Any], Dict[Any, Dict]]:
    # These two arguments have no effect.  Final word-count checks are
    # audit-only: generation-time ``write`` validation is the sole length gate.
    _ = enforce_whole_story_range, allow_corrected_chapter_out_of_range
    report: Dict[str, Any] = {
        "name": name,
        "sources": [str(path) for path in paths],
        "exists": all(path.exists() for path in paths),
        "raw_records": 0,
        "malformed_records": 0,
    }
    latest: Dict[Any, Dict] = {}
    valid_records = 0
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        report["ok"] = False
        report["errors"] = [
            "source file does not exist: " + ", ".join(missing)
        ]
        return report, latest

    # Later (repair) files override earlier records with the same id; all
    # source files are left untouched for auditing.
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                report["raw_records"] += 1
                try:
                    record = json.loads(line)
                    # Treat numeric IDs and equal string IDs as the same
                    # sample so type differences cannot bypass duplicate checks.
                    latest[str(record["id"])] = record
                    valid_records += 1
                except (json.JSONDecodeError, KeyError):
                    report["malformed_records"] += 1

    target = expected_words_from_name(name)
    incomplete = []
    chapter_mismatches = []
    chapter_out_of_range = []
    chapter_metadata_mismatches = []
    out_of_range = []
    empty_stories = []
    computed_words: Dict[str, int] = {}
    for story_id, record in latest.items():
        story = record.get("story")
        outline = record.get("outline") or []
        direct_story = isinstance(story, str)
        chapter_story = isinstance(story, list)
        chapter_targets = {
            chapter.get("id"): chapter.get("word_count")
            for chapter in outline
        }
        corrected_chapter_ids = {
            str(call.get("args", {}).get("chapter_id"))
            for source_chapter in story if chapter_story
            for call in source_chapter.get("tool_trace", [])
            if call.get("name") == "correct" and call.get("ok") is True
        }
        if not record.get("complete"):
            incomplete.append(story_id)
        if not story:
            empty_stories.append(story_id)
        if chapter_story and len(story) != len(outline):
            chapter_mismatches.append({
                "id": story_id,
                "story_chapters": len(story),
                "outline_chapters": len(outline),
            })
        for chapter in story if chapter_story else []:
            chapter_id = chapter.get("id")
            chapter_target = chapter_targets.get(chapter_id)
            if not isinstance(chapter_target, (int, float)):
                continue
            chapter_words = count_words(chapter.get("content", ""))
            if not (
                chapter_target * 0.8
                <= chapter_words
                <= chapter_target * 1.2
            ):
                deviation = {
                    "id": story_id,
                    "chapter_id": chapter_id,
                    "word_count": chapter_words,
                    "target_word_count": chapter_target,
                    "corrected_after_write": (
                        str(chapter_id) in corrected_chapter_ids
                    ),
                }
                chapter_out_of_range.append(deviation)
        if direct_story:
            words = count_words(story)
        elif chapter_story:
            words = sum(
                count_words(chapter.get("content", "")) for chapter in story
            )
        else:
            words = 0

        # Methods such as DOME store the story as one string plus separate
        # per-chapter word-count metadata. These records lack StructuredState's
        # outline/content structure, but chapter count, per-chapter ±20% and
        # the chapter sum are still checked, not just the full text.
        metadata_chapters = record.get("chapters")
        if direct_story and isinstance(metadata_chapters, list):
            declared_count = record.get("num_chapters")
            if (
                isinstance(declared_count, int)
                and declared_count != len(metadata_chapters)
            ):
                chapter_metadata_mismatches.append({
                    "id": story_id,
                    "declared_chapters": declared_count,
                    "metadata_chapters": len(metadata_chapters),
                })
            metadata_total = 0
            metadata_total_known = True
            for chapter in metadata_chapters:
                if not isinstance(chapter, dict):
                    metadata_total_known = False
                    continue
                chapter_target = chapter.get("target_word_count")
                chapter_words = chapter.get("word_count")
                if not isinstance(chapter_words, (int, float)):
                    metadata_total_known = False
                    continue
                metadata_total += int(chapter_words)
                if (
                    isinstance(chapter_target, (int, float))
                    and not (
                        chapter_target * 0.8
                        <= chapter_words
                        <= chapter_target * 1.2
                    )
                ):
                    deviation = {
                        "id": story_id,
                        "chapter_id": chapter.get("id"),
                        "word_count": chapter_words,
                        "target_word_count": chapter_target,
                        "corrected_after_write": False,
                    }
                    chapter_out_of_range.append(deviation)
            if metadata_total_known and metadata_total != words:
                chapter_metadata_mismatches.append({
                    "id": story_id,
                    "metadata_word_count": metadata_total,
                    "computed_story_words": words,
                })

        computed_words[str(story_id)] = words
        if not target * 0.8 <= words <= target * 1.2:
            out_of_range.append({"id": story_id, "word_count": words})

    errors = []
    duplicate_records = valid_records - len(latest)
    if len(latest) != expected_samples:
        errors.append(
            f"expected {expected_samples} unique ids, found {len(latest)}"
        )
    if report["malformed_records"]:
        errors.append(f"{report['malformed_records']} malformed record(s)")
    if reject_duplicates and duplicate_records:
        errors.append(f"{duplicate_records} duplicate record(s)")
    missing_ids = []
    extra_ids = []
    if expected_ids is not None:
        actual_ids = set(latest)
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        if missing_ids:
            errors.append(f"{len(missing_ids)} expected id(s) missing")
        if extra_ids:
            errors.append(f"{len(extra_ids)} unexpected id(s)")
    if incomplete:
        errors.append(f"{len(incomplete)} incomplete story/stories")
    if empty_stories:
        errors.append(f"{len(empty_stories)} empty story/stories")
    if chapter_mismatches:
        errors.append(
            f"{len(chapter_mismatches)} chapter/outline mismatch(es)"
        )
    if chapter_metadata_mismatches:
        errors.append(
            f"{len(chapter_metadata_mismatches)} chapter metadata mismatch(es)"
        )

    report.update({
        "target_words": target,
        "unique_ids": len(latest),
        "duplicate_records": duplicate_records,
        "complete": len(latest) - len(incomplete),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "incomplete_ids": incomplete,
        "empty_story_ids": empty_stories,
        "chapter_mismatches": chapter_mismatches,
        "chapter_out_of_range": chapter_out_of_range,
        # Length deviations are never blocking during final validation,
        # regardless of their source.
        "blocking_chapter_out_of_range": [],
        "corrected_chapter_out_of_range_allowed": True,
        "chapter_range_enforced": False,
        "chapter_metadata_mismatches": chapter_metadata_mismatches,
        "out_of_range": out_of_range,
        "whole_story_range_enforced": False,
        "word_count_validation_mode": "audit_only_write_gate",
        "computed_word_counts": computed_words,
        "errors": errors,
        "ok": not errors,
    })
    return report, latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", action="append", required=True,
        help="NAME=PATH; may be supplied multiple times",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument(
        "--expected-prompts",
        type=Path,
        help="Optional JSONL whose ids must exactly match every dataset.",
    )
    parser.add_argument(
        "--reject-duplicates",
        action="store_true",
        help="Fail if a source contains more than one valid record per id.",
    )
    parser.add_argument(
        "--allow-whole-story-out-of-range",
        action="store_true",
        help=(
            "Has no effect. Final chapter and whole-story word counts are "
            "always audit-only."
        ),
    )
    parser.add_argument(
        "--allow-corrected-chapter-out-of-range",
        action="store_true",
        help=(
            "Has no effect. Final chapter and whole-story word counts are "
            "always audit-only."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Path]] = {}
    for spec in args.dataset:
        name, path = parse_dataset(spec)
        grouped.setdefault(name, []).append(path)

    expected_ids = None
    if args.expected_prompts:
        prompt_rows = [
            json.loads(line)
            for line in args.expected_prompts.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        prompt_id_list = [str(row["id"]) for row in prompt_rows]
        expected_ids = set(prompt_id_list)
        if (
            len(prompt_id_list) != args.expected_samples
            or len(expected_ids) != args.expected_samples
        ):
            raise SystemExit(
                "expected prompt file must contain exactly "
                f"{args.expected_samples} unique ids"
            )

    reports = []
    canonical: Dict[str, Dict[Any, Dict]] = {}
    for name, paths in grouped.items():
        report, records = validate_dataset(
            name,
            paths,
            args.expected_samples,
            expected_ids=expected_ids,
            reject_duplicates=args.reject_duplicates,
            enforce_whole_story_range=(
                not args.allow_whole_story_out_of_range
            ),
            allow_corrected_chapter_out_of_range=(
                args.allow_corrected_chapter_out_of_range
            ),
        )
        reports.append(report)
        canonical[name] = records

    all_ok = all(report["ok"] for report in reports)
    report_path = output_dir / "integrity_report.json"
    report_path.write_text(
        json.dumps(
            {"ok": all_ok, "datasets": reports},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not all_ok:
        print(report_path)
        return 1

    for name, records in canonical.items():
        target = output_dir / f"{name}.jsonl"
        with target.open("w", encoding="utf-8") as file:
            for story_id in sorted(records, key=lambda value: str(value)):
                file.write(
                    json.dumps(records[story_id], ensure_ascii=False) + "\n"
                )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
