#!/usr/bin/env python
# coding: utf-8
"""
ConStory-Bench evaluation adapter.
Reads story JSONL from the generation pipeline, runs consistency evaluation
using the original ConStory-Checker judge with 5 error categories.

Input: stories JSONL with {id, prompt, story (str or list of chapters)}
Output: CSV with error annotations per category/subtype
"""

import json
import asyncio
import argparse
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
from tqdm import tqdm

# Reuse judge logic from the original ConStory-Bench
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, SCRIPT_DIR)
from judge import (
    JudgeLLMClient, ConStoryChecker, load_prompt_templates,
    EVALUATION_CRITERIA, FatalAPIError,
    setup_logger, DEFAULT_CONCURRENT, DEFAULT_MAX_TOKENS,
)


TARGET_ENDING_MARKER = """\
>>>>>>>>>>>> TARGET ENDING CHAPTERS START >>>>>>>>>>>>
IMPORTANT: The following chapters are the only target chapters for error
attribution. Continue using the entire preceding narrative as reference
evidence, but report an error only when its later contradictory passage
appears in one of the following chapters.
For every reported item except the Timeline check's global
abandoned_plot_elements, exact_quote must be copied verbatim from text after
this marker. If only the contradiction_pair is after this marker, swap the
two evidence fields; if neither is after this marker, omit the item.
Target chapter IDs: {target_chapter_ids}
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""


def count_words(text: str) -> int:
    """Estimate checked-window length by counting English words and single CJK
    characters."""
    return len(
        re.findall(
            r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[\u3400-\u9fff]",
            text,
        )
    )


def normalize_story(
    story: Any,
    target_ending_chapters: int,
) -> Tuple[str, Dict[str, Any]]:
    """Flatten a chapter list into full text, optionally inserting one scope
    marker before the last N chapters."""
    if isinstance(story, str):
        if target_ending_chapters:
            raise ValueError(
                "--target-ending-chapters requires a chapter-list story"
            )
        return story, {
            "scope_mode": "full_narrative",
            "target_chapter_ids": "",
            "target_chapter_names": "",
            "target_chapter_count": 0,
            "checked_words": count_words(story),
        }

    if not isinstance(story, list):
        return str(story), {
            "scope_mode": "full_narrative",
            "target_chapter_ids": "",
            "target_chapter_names": "",
            "target_chapter_count": 0,
            "checked_words": count_words(str(story)),
        }

    chapter_rows = []
    for index, chapter in enumerate(story):
        if isinstance(chapter, dict):
            chapter_id = chapter.get("id", index + 1)
            chapter_name = str(chapter.get("name", "")).strip()
            content = str(chapter.get("content", ""))
        else:
            chapter_id = index + 1
            chapter_name = ""
            content = str(chapter)
        chapter_rows.append(
            {
                "id": chapter_id,
                "name": chapter_name,
                "content": content,
            }
        )

    target_count = min(target_ending_chapters, len(chapter_rows))
    target_start = len(chapter_rows) - target_count
    target_rows = chapter_rows[target_start:] if target_count else chapter_rows
    target_ids = [row["id"] for row in target_rows]
    target_names = [row["name"] for row in target_rows]
    target_ids_text = ", ".join(str(value) for value in target_ids)

    parts = []
    for index, row in enumerate(chapter_rows):
        if target_count and index == target_start:
            parts.append(
                TARGET_ENDING_MARKER.format(
                    target_chapter_ids=target_ids_text
                )
            )
        if target_count:
            heading = f"## Chapter ID {row['id']}"
            if row["name"]:
                heading += f": {row['name']}"
        else:
            heading = f"## {row['name']}"
        parts.append(f"{heading}\n\n{row['content']}")

    checked_text = "\n\n".join(row["content"] for row in target_rows)
    return "\n\n".join(parts), {
        "scope_mode": (
            "target_ending_chapters" if target_count else "full_narrative"
        ),
        "target_chapter_ids": target_ids_text if target_count else "",
        "target_chapter_names": (
            " | ".join(target_names) if target_count else ""
        ),
        "target_chapter_count": target_count,
        "checked_words": count_words(checked_text),
    }


def load_stories_jsonl(
    path: str,
    include_incomplete: bool = False,
    target_ending_chapters: int = 0,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Load the latest record per id and normalize chapters to flat text."""
    rows_by_id = {}
    stats = {
        "records": 0,
        "duplicates": 0,
        "incomplete_skipped": 0,
        "empty_skipped": 0,
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["records"] += 1
            record = json.loads(line.strip())
            if record.get("complete") is False and not include_incomplete:
                stats["incomplete_skipped"] += 1
                continue
            # Normalize story: could be string (baseline) or list of chapters (agent)
            story = record.get("story", "")
            story, scope_metadata = normalize_story(
                story,
                target_ending_chapters=target_ending_chapters,
            )
            if not isinstance(story, str) or not story.strip():
                stats["empty_skipped"] += 1
                continue
            sid = record["id"]
            key = str(sid)
            if key in rows_by_id:
                stats["duplicates"] += 1
            rows_by_id[key] = {
                "id": record["id"],
                "prompt": record.get("prompt", ""),
                "generated_story": story,
                "language": record.get("language", ""),
                **scope_metadata,
            }
    return list(rows_by_id.values()), stats


def main():
    parser = argparse.ArgumentParser(description="ConStory-Bench: evaluate story consistency")
    parser.add_argument("--input", required=True, help="Input stories JSONL")
    parser.add_argument("--output", default=None, help="Output CSV (default: auto)")
    parser.add_argument("--judge-model", default="DeepSeek-V4-Flash")
    parser.add_argument("--api-base", default="https://www.autodl.art/api/v1")
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", "")
    )
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum judge output tokens for each consistency category",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1200.0,
        help="Per judge request timeout in seconds (default: 1200)",
    )
    parser.add_argument("--prompts-dir", default=os.path.join(SCRIPT_DIR, "prompts_terminal_scope"))
    parser.add_argument(
        "--target-ending-chapters",
        type=int,
        default=0,
        help=(
            "Insert a terminal-scope marker before the final N chapters; "
            "0 keeps the official full-narrative behavior"
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Evaluate records with complete=false (excluded by default)",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or OPENAI_API_KEY is required")
    if args.concurrent <= 0:
        parser.error("--concurrent must be > 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.target_ending_chapters < 0:
        parser.error("--target-ending-chapters must be >= 0")

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger("constory_eval", f"logs/constory_eval_{ts}.log")

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(SCRIPT_DIR, f"results_{base}_{ts}.csv")

    stories, load_stats = load_stories_jsonl(
        args.input,
        include_incomplete=args.include_incomplete,
        target_ending_chapters=args.target_ending_chapters,
    )
    logger.info(f"Loaded {len(stories)} stories from {args.input}")
    logger.info(f"Input normalization: {load_stats}")

    templates = load_prompt_templates(args.prompts_dir)
    logger.info(f"Loaded {len(templates)} prompt templates")

    client = JudgeLLMClient(
        api_base=args.api_base, api_key=args.api_key,
        model=args.judge_model, max_concurrent=args.concurrent,
        logger=logger, max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
    )
    checker = ConStoryChecker(
        client=client, prompt_templates=templates,
        story_column="generated_story", logger=logger,
    )

    # Resume
    import pandas as pd
    completed_ids = set()
    existing_by_id = {}
    if not args.no_resume and os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output)
            for record in existing.to_dict("records"):
                key = str(record["id"])
                existing_by_id[key] = record
                if (
                    record.get("evaluation_status") == "completed"
                    and record.get("criteria_completed") in (5, 5.0, "5")
                ):
                    completed_ids.add(key)
            logger.info(
                f"Resuming: {len(completed_ids)} stories complete; "
                f"{len(existing_by_id) - len(completed_ids)} stories need "
                "missing-criteria recovery"
            )
        except Exception:
            pass

    results_by_id = dict(existing_by_id)
    to_eval = [s for s in stories if str(s["id"]) not in completed_ids]
    logger.info(f"Evaluating {len(to_eval)} stories")

    async def run():
        connector = aiohttp.TCPConnector(limit=max(args.concurrent * 2, 50))
        async with aiohttp.ClientSession(connector=connector) as session:
            pbar = tqdm(total=len(to_eval), desc="ConStory-Bench", unit="story")
            # Story-level concurrency. Each story still fans out to 5 criteria,
            # but the JudgeLLMClient's internal semaphore caps total in-flight
            # HTTP requests at args.concurrent.
            story_sem = asyncio.Semaphore(args.concurrent)
            save_lock = asyncio.Lock()
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            fatal_seen = asyncio.Event()

            async def process(story_data):
                if fatal_seen.is_set():
                    return
                async with story_sem:
                    if fatal_seen.is_set():
                        return
                    try:
                        sid = str(story_data["id"])
                        result = await checker.evaluate_single(
                            session,
                            story_data,
                            existing_result=existing_by_id.get(sid),
                        )
                    except FatalAPIError as e:
                        logger.error(f"Fatal: {e}")
                        fatal_seen.set()
                        return
                    except Exception as e:
                        logger.error(f"Error id={story_data['id']}: {e}")
                        return
                    async with save_lock:
                        results_by_id[str(story_data["id"])] = result
                        pd.DataFrame(list(results_by_id.values())).to_csv(
                            args.output, index=False, encoding="utf-8-sig"
                        )
                        pbar.update(1)

            await asyncio.gather(*(process(s) for s in to_eval))
            pbar.close()

    asyncio.run(run())
    print(f"Done! Results saved to {args.output}")


if __name__ == "__main__":
    main()
