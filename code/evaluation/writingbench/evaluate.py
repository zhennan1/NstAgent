#!/usr/bin/env python
# coding: utf-8
"""
WritingBench evaluation (native pipeline, adapted for free-form story prompts).

Faithful to the original WritingBench query-dependent evaluation framework
(https://arxiv.org/abs/2503.05244):

  Phase 1 — Dynamic Criteria Generation:
      For each query the LLM generates 5 instance-specific criteria. Each
      criterion has a `name`, a `criteria_description`, AND full per-band
      scoring rubrics ("1-2", "3-4", "5-6", "7-8", "9-10"). This is the exact
      prompt from the WritingBench paper appendix.

  Phase 2 — Rubric-based Scoring:
      Each criterion is scored independently on a 1-10 scale using the native
      WritingBench scoring prompt (see prompt.py), which restates the full
      criterion (with rubrics) both before and after the response.

Our stories come from ConStory-Bench prompts (not WritingBench's 1,000 queries),
so criteria are generated per prompt. To keep comparisons across methods fair,
generated criteria are CACHED per prompt: stories from different methods
written from the same prompt are always scored against identical criteria
(mirroring WritingBench's pre-fixed checklists).

Output is compatible with the native calculate_scores.py:
    {"index": <id>, "scores": {<criterion_name>: [{"score", "reason"}]}}
"""

import json
import re
import hashlib
import asyncio
import argparse
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import httpx
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# Native WritingBench scoring prompt (criteria restated before & after response).
from prompt import evaluate_system, evaluate_prompt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Phase 1: Criteria Generation (verbatim from WritingBench paper appendix) ---
CRITERIA_GEN_SYSTEM = (
    "You are an expert evaluator with extensive experience in evaluating the "
    "response of a given query."
)

CRITERIA_GEN_PROMPT = """
Please generate five strict evaluation criteria for assessing the response given the following query. Each criterion should include the following fields: name, criteria_description, 1-2, 3-4, 5-6, 7-8, 9-10.
The criteria should be designed to emphasize detailed assessment and distinguish subtle differences in quality. Ensure that the criteria can discern issues such as relevance, coherence, depth, specificity, and adherence to the query context.
Do not include any additional text. Only output the criteria in the specified JSON format.

** Query **
{query}

** Output format **
[
    {{
        "name": "first_criteria_name",
        "criteria_description": "Description for the first criteria, emphasizing detailed and critical assessment.",
        "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
        "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
        "5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
        "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
        "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."
    }},
    ...
]
""".strip()

CRITERION_FIELDS = ["name", "criteria_description", "1-2", "3-4", "5-6", "7-8", "9-10"]
CRITERIA_SCHEMA_VERSION = "writingbench-five-rubrics-v1"
CONTENT_FILTER_RETRY_INSTRUCTION = (
    "\nFor a safety-compliant verdict, do not quote, summarize, name, or "
    "otherwise reproduce any plot event or sensitive detail from the response. "
    "Apply the rubric internally, then return exactly one JSON object with the "
    "integer score and the generic reason: \"Rubric-based evaluation completed "
    "without reproducing story content.\" Do not output any other text."
)


async def streaming_chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    request_timeout: float,
    logger: logging.Logger,
    label: str,
) -> Tuple[str, str | None]:
    """Collect only the final answer while keeping long reasoning alive.

    DeepSeek-V4-Pro may emit minutes of ``reasoning_content`` before a short
    final JSON verdict. Streaming prevents a live request from looking idle;
    the explicit asyncio timeout remains the hard per-attempt ceiling.
    """
    started_at = time.monotonic()
    first_event_at = None
    content_parts: List[str] = []
    reasoning_chars = 0
    chunks = 0
    finish_reason = None

    async def collect() -> None:
        nonlocal first_event_at, reasoning_chars, chunks, finish_reason
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        async for event in stream:
            if not event.choices:
                continue
            if first_event_at is None:
                first_event_at = time.monotonic()
                logger.info(
                    f"  {label}: stream started after "
                    f"{first_event_at - started_at:.1f}s"
                )
            choice = event.choices[0]
            delta = choice.delta
            content = getattr(delta, "content", None) or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_chars += len(reasoning)
            chunks += 1
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    async with asyncio.timeout(request_timeout):
        await collect()
    content = "".join(content_parts)
    logger.info(
        f"  {label}: stream finished in {time.monotonic() - started_at:.1f}s; "
        f"chunks={chunks}, reasoning_chars={reasoning_chars}, "
        f"content_chars={len(content)}, finish_reason={finish_reason!r}"
    )
    return content, finish_reason


def setup_logger(name: str, log_file: str) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def extract_json(text: str):
    text = text.strip()
    candidates = [text]
    for pat in [r'```json\s*\n?(.*?)\n?\s*```', r'```\s*\n?(.*?)\n?\s*```']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            candidates.append(m.group(1).strip())
    # Try strict parse first, then again with CJK "smart" quotes normalized to
    # plain ones (judge models often wrap phrases in “...”/「...」inside reason,
    # which breaks JSON — the original WritingBench used eval() to tolerate this).
    for cand in candidates:
        for variant in (cand, cand.translate(str.maketrans("“”‘’「」『』", '\'\'\'\'\'\'\'\''))):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue
    # Last-resort regex: pull score/reason out of the raw text (safer than eval).
    m = re.search(r'"score"\s*:\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        # Grab everything after `"reason": "` up to the last closing brace, then
        # strip trailing quote chars (incl. CJK smart quotes the model may leave).
        rm = re.search(r'"reason"\s*:\s*"(.*)', text, re.DOTALL)
        reason = ""
        if rm:
            reason = rm.group(1).rsplit("}", 1)[0].strip().rstrip(",").strip()
            reason = reason.strip('"“”\'‘’」』')
        return {"score": score, "reason": reason}
    raise ValueError(f"Cannot extract JSON from: {text[:300]}")


def prompt_key(prompt: str) -> str:
    material = f"{CRITERIA_SCHEMA_VERSION}\n{prompt.strip()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_criteria_cache(path: str) -> Dict[str, List[Dict]]:
    cache = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    criteria = rec["criteria"]
                    if valid_criteria(criteria):
                        cache[rec["key"]] = criteria
                except (json.JSONDecodeError, KeyError):
                    pass
    return cache


def valid_criteria(criteria) -> bool:
    """Require exactly five unique, fully populated WritingBench rubrics."""
    if not isinstance(criteria, list) or len(criteria) != 5:
        return False
    names = set()
    for c in criteria:
        if (
            not isinstance(c, dict)
            or not set(CRITERION_FIELDS).issubset(c)
        ):
            return False
        if any(
            not isinstance(c[field], str) or not c[field].strip()
            for field in CRITERION_FIELDS
        ):
            return False
        normalized_name = c["name"].strip().casefold()
        if normalized_name in names:
            return False
        names.add(normalized_name)
    return True


async def generate_criteria(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    logger: logging.Logger,
    max_tokens: int,
    request_timeout: float,
) -> List[Dict]:
    """Phase 1: generate 5 instance-specific criteria WITH full per-band rubrics."""
    msg = CRITERIA_GEN_PROMPT.format(query=prompt)
    last_error = "unknown criteria-generation failure"
    for retry in range(3):
        try:
            content, finish_reason = await streaming_chat_completion(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": CRITERIA_GEN_SYSTEM},
                    {"role": "user", "content": msg},
                ],
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                logger=logger,
                label="criteria generation",
            )
            if finish_reason != "stop":
                raise ValueError(
                    f"incomplete finish_reason={finish_reason!r}"
                )
            criteria = extract_json(content)
            if valid_criteria(criteria):
                return criteria
            last_error = (
                "response must contain exactly five unique, non-empty rubrics "
                f"with fields {CRITERION_FIELDS}"
            )
            logger.warning(
                f"  criteria gen retry {retry+1}: {last_error}"
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(f"  criteria gen retry {retry+1}: {e}")
        if retry < 2:
            await asyncio.sleep(3 * (retry + 1))
    raise RuntimeError(
        f"criteria generation failed after retries: {last_error}"
    )


def format_criterion(c: Dict) -> str:
    """Render a single criterion (name + description + all rubric bands) as the
    `criteria` block for the native scoring prompt."""
    lines = [f"{c['name']}: {c['criteria_description']}"]
    for band in ["1-2", "3-4", "5-6", "7-8", "9-10"]:
        lines.append(f"{band}: {c.get(band, '')}")
    return "\n".join(lines)


async def score_criterion(client: AsyncOpenAI, model: str, criterion: Dict,
                          query: str, response: str,
                          logger: logging.Logger,
                          max_tokens: int,
                          request_timeout: float) -> Dict:
    """Phase 2: score one criterion with the native WritingBench scoring prompt."""
    prompt = evaluate_prompt.format(
        criteria=format_criterion(criterion), query=query, response=response,
    )
    last_error = "unknown judge failure"
    content_filter_retry = False
    for retry in range(3):
        try:
            system_content = evaluate_system
            user_content = prompt
            if content_filter_retry:
                system_content += CONTENT_FILTER_RETRY_INSTRUCTION
                user_content += CONTENT_FILTER_RETRY_INSTRUCTION
            content, finish_reason = await streaming_chat_completion(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                logger=logger,
                label=f"criterion {criterion['name']!r}",
            )
            if finish_reason != "stop":
                if finish_reason == "content_filter":
                    content_filter_retry = True
                raise ValueError(
                    f"incomplete finish_reason={finish_reason!r}"
                )
            parsed = extract_json(content)
            if isinstance(parsed, dict) and "score" in parsed:
                score = parsed["score"]
                reason = parsed.get("reason")
                if (
                    type(score) is int
                    and 1 <= score <= 10
                    and isinstance(reason, str)
                    and reason.strip()
                ):
                    return {
                        "status": "completed",
                        "score": score,
                        "reason": reason,
                    }
            raise ValueError(
                "judge response requires integer score 1-10 and non-empty reason"
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(f"  criterion '{criterion['name']}' retry {retry+1}: {e}")
        if retry < 2:
            await asyncio.sleep(3 * (retry + 1))
    return {
        "status": "failed",
        "score": None,
        "reason": "",
        "error": last_error,
    }


def load_stories(
    path: str, include_incomplete: bool = False
) -> Tuple[List[Dict], Dict[str, int]]:
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
            story = record.get("story", "")
            if isinstance(story, list):
                parts = []
                for ch in story:
                    if isinstance(ch, dict):
                        parts.append(f"## {ch.get('name', '')}\n\n{ch.get('content', '')}")
                    else:
                        parts.append(str(ch))
                story = "\n\n".join(parts)
            if not isinstance(story, str) or not story.strip():
                stats["empty_skipped"] += 1
                continue
            key = str(record["id"])
            if key in rows_by_id:
                stats["duplicates"] += 1
            rows_by_id[key] = {
                "id": record["id"],
                "prompt": record.get("prompt", ""),
                "story": story,
            }
    return list(rows_by_id.values()), stats


def _completed_existing_score(
    existing_record: Dict | None, criterion_name: str
) -> Dict | None:
    """Return one trustworthy cached criterion score, if present."""
    if not isinstance(existing_record, dict):
        return None
    evaluations = existing_record.get("scores", {}).get(criterion_name)
    if not isinstance(evaluations, list) or not evaluations:
        return None
    evaluation = evaluations[0]
    if not isinstance(evaluation, dict):
        return None
    score = evaluation.get("score")
    reason = evaluation.get("reason")
    if (
        evaluation.get("status", "completed") == "completed"
        and type(score) is int
        and 1 <= score <= 10
        and isinstance(reason, str)
        and reason.strip()
    ):
        return evaluation
    return None


async def evaluate_one_story(
    client: AsyncOpenAI,
    model: str,
    story: Dict,
    criteria: List[Dict],
    logger: logging.Logger,
    max_tokens: int,
    request_timeout: float = 1200.0,
    existing_record: Dict | None = None,
) -> Dict:
    """Score only missing criteria and preserve prior successful verdicts."""
    scores: Dict[str, List[Dict]] = {}
    missing = []
    for criterion in criteria:
        name = criterion["name"]
        cached = _completed_existing_score(existing_record, name)
        if cached is not None:
            scores[name] = [cached]
        else:
            missing.append(criterion)

    if missing:
        results = await asyncio.gather(*[
            score_criterion(
                client,
                model,
                criterion,
                story["prompt"],
                story["story"],
                logger,
                max_tokens,
                request_timeout,
            )
            for criterion in missing
        ])
        for criterion, result in zip(missing, results):
            scores[criterion["name"]] = [result]

    # Preserve the canonical cached-criteria order in every appended record.
    scores = {
        criterion["name"]: scores[criterion["name"]]
        for criterion in criteria
    }
    failed = [
        criterion["name"]
        for criterion in criteria
        if scores[criterion["name"]][0].get("status") != "completed"
    ]
    completed = len(criteria) - len(failed)
    return {
        "index": story["id"],
        "scores": scores,
        "evaluation_status": (
            "completed" if completed == 5 else f"partial_{completed}_5"
        ),
        "criteria_completed": completed,
        "criteria_failed": len(failed),
        "failed_criteria": failed,
        "story_characters": len(story["story"]),
    }


async def main_async(args):
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger("writingbench_eval", f"logs/writingbench_eval_{ts}.log")

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(SCRIPT_DIR, f"results_{base}_{ts}.jsonl")
    if args.criteria_cache is None:
        args.criteria_cache = os.path.join(SCRIPT_DIR, "criteria_cache.jsonl")

    stories, load_stats = load_stories(
        args.input, include_incomplete=args.include_incomplete
    )
    logger.info(f"Loaded {len(stories)} stories")
    logger.info(f"Input normalization: {load_stats}")

    # Resume at criterion granularity. Completed stories are skipped; partial
    # stories preserve successful scores and request only their missing cells.
    existing_by_id: Dict[str, Dict] = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        existing_by_id[str(record["index"])] = record
                    except (json.JSONDecodeError, KeyError):
                        pass
        completed_ids = {
            story_id for story_id, record in existing_by_id.items()
            if record.get("evaluation_status") in (None, "completed")
            and int(record.get("criteria_completed", 5)) == 5
        }
        partial_ids = set(existing_by_id) - completed_ids
        logger.info(
            f"Resuming: {len(completed_ids)} stories complete; "
            f"{len(partial_ids)} stories need missing-criterion recovery"
        )
    else:
        completed_ids = set()

    to_eval = [s for s in stories if str(s["id"]) not in completed_ids]
    if not to_eval:
        logger.info("Nothing to evaluate.")
        write_summary(args.output)
        return
    logger.info(f"Evaluating {len(to_eval)} stories")

    # Cross-run criteria cache: identical prompt -> identical criteria (fair comparison)
    criteria_cache = load_criteria_cache(args.criteria_cache)
    logger.info(f"Loaded {len(criteria_cache)} cached criteria sets")

    http_limit = max(args.concurrent * 5 + 16, 64)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=http_limit, max_keepalive_connections=http_limit),
        # Streaming uses an explicit total asyncio timeout per call.  The
        # transport timeout here detects a genuinely idle socket between SSE
        # chunks, rather than limiting total model reasoning time.
        timeout=httpx.Timeout(connect=30, read=min(300, args.request_timeout),
                              write=min(300, args.request_timeout), pool=30),
    )
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        http_client=http_client,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrent)
    out_lock = asyncio.Lock()
    cache_lock = asyncio.Lock()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    pbar = atqdm(total=len(to_eval), desc="WritingBench", unit="story")

    async def get_criteria(prompt: str) -> List[Dict]:
        key = prompt_key(prompt)
        # Fast path: already cached
        async with cache_lock:
            if key in criteria_cache:
                return criteria_cache[key]
        # Generate outside the lock, then store once
        criteria = await generate_criteria(
            client, args.judge_model, prompt, logger, args.max_tokens,
            args.request_timeout,
        )
        async with cache_lock:
            if key not in criteria_cache:
                criteria_cache[key] = criteria
                with open(args.criteria_cache, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "criteria": criteria}, ensure_ascii=False) + "\n")
            return criteria_cache[key]

    async def process_one(story):
        async with semaphore:
            try:
                criteria = await get_criteria(story["prompt"])
                record = await evaluate_one_story(
                    client,
                    args.judge_model,
                    story,
                    criteria,
                    logger,
                    args.max_tokens,
                    args.request_timeout,
                    existing_record=existing_by_id.get(str(story["id"])),
                )
            except Exception as e:
                logger.error(f"  id={story['id']}: evaluation failed: {e}")
                record = {
                    "index": story["id"],
                    "scores": {},
                    "evaluation_status": "failed",
                    "criteria_completed": 0,
                    "criteria_failed": 5,
                    "failed_criteria": [],
                    "story_characters": len(story["story"]),
                    "error": str(e),
                }
            async with out_lock:
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            valid_scores = [
                evaluation["score"]
                for evaluations in record["scores"].values()
                for evaluation in evaluations
                if evaluation.get("status", "completed") == "completed"
                and type(evaluation.get("score")) is int
            ]
            avg_text = (
                f"{sum(valid_scores) / len(valid_scores):.2f}"
                if valid_scores else "N/A"
            )
            logger.info(
                f"  id={story['id']}: status={record['evaluation_status']}, "
                f"avg={avg_text}"
            )
            pbar.update(1)

    task_results = await asyncio.gather(
        *[process_one(s) for s in to_eval], return_exceptions=True
    )
    for task_error in task_results:
        if isinstance(task_error, Exception):
            logger.error(f"Unhandled story task error: {task_error}")
    pbar.close()
    await http_client.aclose()
    write_summary(args.output)
    print(f"\nDone! Results saved to {args.output}")


def write_summary(output_path: str):
    if not os.path.exists(output_path):
        return
    records_by_id = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    records_by_id[str(record["index"])] = record
                except json.JSONDecodeError:
                    pass
    records = list(records_by_id.values())
    if not records:
        return

    complete_story_avg = []
    all_valid_scores = []
    criterion_failures = 0
    status_counts = {"completed": 0, "partial": 0, "failed": 0}
    for r in records:
        status = r.get("evaluation_status", "completed")
        bucket = (
            "completed" if status == "completed"
            else "partial" if str(status).startswith("partial_")
            else "failed"
        )
        status_counts[bucket] += 1
        vals = [
            e.get("score")
            for evals in r.get("scores", {}).values()
            for e in evals
            if e.get("status", "completed") == "completed"
            and type(e.get("score")) is int
            and 1 <= e["score"] <= 10
        ]
        all_valid_scores.extend(vals)
        criterion_failures += int(r.get("criteria_failed", max(0, 5 - len(vals))))
        if status == "completed" and len(vals) == 5:
            complete_story_avg.append(sum(vals) / 5)
    overall = (
        sum(complete_story_avg) / len(complete_story_avg)
        if complete_story_avg else None
    )
    valid_score_avg = (
        sum(all_valid_scores) / len(all_valid_scores)
        if all_valid_scores else None
    )

    lines = [
        f"Total stories: {len(records)}",
        f"Completed stories: {status_counts['completed']}",
        f"Partial stories: {status_counts['partial']}",
        f"Failed stories: {status_counts['failed']}",
        f"Valid criterion scores: {len(all_valid_scores)}",
        f"Failed criterion scores: {criterion_failures}",
        (
            f"Overall average (completed stories only): "
            f"{overall:.2f}/10  (scaled x10 = {overall*10:.1f})"
            if overall is not None else
            "Overall average (completed stories only): N/A"
        ),
        (
            f"Diagnostic average (all valid criterion scores): "
            f"{valid_score_avg:.2f}/10"
            if valid_score_avg is not None else
            "Diagnostic average (all valid criterion scores): N/A"
        ),
    ]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    summary_path = os.path.splitext(output_path)[0] + ".txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="WritingBench (native): evaluate story quality")
    parser.add_argument("--input", required=True, help="Input stories JSONL")
    parser.add_argument("--output", default=None, help="Output scores JSONL")
    parser.add_argument("--criteria-cache", default=None,
                        help="JSONL cache of per-prompt criteria (shared across methods for fair comparison)")
    parser.add_argument("--judge-model", default="DeepSeek-V4-Flash")
    parser.add_argument("--api-base", default="https://www.autodl.art/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--concurrent", type=int, default=20,
                        help="Max concurrent stories (criteria within a story always run in parallel)")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1200.0,
        help="Per judge request timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum judge output tokens per criteria-generation/scoring call",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Evaluate records with complete=false (excluded by default)",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or an explicitly configured endpoint key is required")
    if args.concurrent <= 0:
        parser.error("--concurrent must be > 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
