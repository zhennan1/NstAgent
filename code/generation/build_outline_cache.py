#!/usr/bin/env python
# coding: utf-8
"""Populate and validate the shared LongStory outline-artifact cache.

The cache stores the full planning chain (premise, optional detail/acts, and
the normalized chapter outline).  Generation experiments should populate it
once with this command and then run both conditions with
``--outline-cache-policy require``.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from agent_core import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTLINE_CACHE_LOCK_TIMEOUT,
    DEFAULT_TEMPERATURE,
    OUTLINE_CACHE_POLICIES,
    OutlineAgent,
    OutlineCache,
    detect_language,
    load_prompt_items,
    stage_count_for_words,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate model/prompt outline artifacts without writing stories"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True,
                        help="model name/path sent to the endpoint")
    parser.add_argument("--cache-model-id", required=True,
                        help="canonical model identity stored in cache keys")
    parser.add_argument("--api-base", required=True)
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY")
    )
    parser.add_argument("--word-count", type=int, default=10000)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrent", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--client-max-retries", type=int, default=2)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--policy",
        choices=tuple(x for x in OUTLINE_CACHE_POLICIES if x != "off"),
        default="read-write",
    )
    parser.add_argument(
        "--lock-timeout", type=float,
        default=DEFAULT_OUTLINE_CACHE_LOCK_TIMEOUT,
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.word_count <= 0:
        parser.error("--word-count must be > 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.concurrent <= 0:
        parser.error("--concurrent must be > 0")
    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.end is not None and args.end <= args.start:
        parser.error("--end must be greater than --start")
    if args.client_max_retries < 0:
        parser.error("--client-max-retries must be >= 0")
    if args.lock_timeout <= 0:
        parser.error("--lock-timeout must be > 0")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("outline_cache_builder")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(handler)
    return logger


def atomic_write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(
        f"{target.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temp.open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, target)
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass


async def run(args: argparse.Namespace) -> int:
    logger = setup_logger()
    items = load_prompt_items(args.input)
    end = args.end if args.end is not None else len(items)
    selected = items[args.start:end]
    logger.info(
        "Populating %d outlines: model=%s cache_model_id=%s words=%d "
        "policy=%s concurrency=%d",
        len(selected), args.model, args.cache_model_id, args.word_count,
        args.policy, args.concurrent,
    )

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        max_retries=args.client_max_retries,
    )
    cache = OutlineCache(
        cache_dir=args.cache_dir,
        policy=args.policy,
        model_id=args.cache_model_id,
        model_request_name=args.model,
        word_count=args.word_count,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        disable_thinking=args.disable_thinking,
        lock_timeout=args.lock_timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrent)
    rows: List[Dict[str, Any]] = []
    rows_lock = asyncio.Lock()
    pbar = atqdm(total=len(selected), desc="Outlines", unit="outline")

    async def process_one(item: Dict[str, Any]) -> None:
        started = time.monotonic()
        story_id = item["id"]
        prompt = item["prompt"]
        lang = detect_language(prompt)
        outliner = OutlineAgent(
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            word_count=args.word_count,
            logger=logger,
            lang=lang,
            disable_thinking=args.disable_thinking,
        )

        def validator(artifacts: Dict[str, Any]) -> None:
            expected_stages = stage_count_for_words(args.word_count)
            if artifacts.get("stages") != expected_stages:
                raise ValueError(
                    f"expected stages={expected_stages}, "
                    f"got {artifacts.get('stages')!r}"
                )
            outliner._validate_chapter_outline(
                artifacts.get("outline"), prompt
            )

        row: Dict[str, Any] = {
            "id": story_id,
            "cache_model_id": args.cache_model_id,
            "model_request_name": args.model,
            "word_count": args.word_count,
            "language": lang,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        async with semaphore:
            try:
                artifacts, metadata = await cache.get_or_create(
                    story_id=story_id,
                    prompt=prompt,
                    lang=lang,
                    generator=lambda: outliner.generate(prompt),
                    validator=validator,
                )
                row.update({
                    "status": metadata["status"],
                    "cache_key": metadata["cache_key"],
                    "artifacts_sha256": metadata["artifacts_sha256"],
                    "cache_path": metadata["path"],
                    "outline_chapters": len(artifacts["outline"]),
                    "outline_stages": artifacts["stages"],
                })
            except Exception as error:
                logger.exception("Outline id=%s failed", story_id)
                row.update({
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            finally:
                row["elapsed_seconds"] = round(
                    time.monotonic() - started, 3
                )
                async with rows_lock:
                    rows.append(row)
                pbar.update(1)

    try:
        await asyncio.gather(*(process_one(item) for item in selected))
    finally:
        pbar.close()
        await client.close()

    rows.sort(key=lambda row: str(row["id"]))
    atomic_write_jsonl(args.manifest, rows)
    errors = sum(row["status"] == "error" for row in rows)
    generated = sum(row["status"] == "generated" for row in rows)
    hits = len(rows) - errors - generated
    logger.info(
        "Finished: total=%d generated=%d cache_hits=%d errors=%d manifest=%s",
        len(rows), generated, hits, errors, args.manifest,
    )
    return 1 if errors else 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
