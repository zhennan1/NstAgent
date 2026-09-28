#!/usr/bin/env python
# coding: utf-8
"""
ConStory-Checker: Automated Consistency Evaluation Pipeline

Evaluates generated stories for consistency errors across 5 categories
(19 fine-grained subtypes) using any OpenAI-compatible LLM as judge.

The pipeline:
1. Loads generated stories (parquet)
2. For each story, sends 5 parallel evaluation requests (one per category)
3. Parses structured JSON responses to extract error instances
4. Saves results incrementally to CSV with resume support

Usage:
    python -m constory.judge \
        --input data/stories/gpt4o.parquet \
        --story-column generated_story \
        --model-name gpt4o \
        --judge-model gpt-4o \
        --api-base https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --concurrent 20
"""

import os
import re
import json
import asyncio
import argparse
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import aiohttp
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MAX_TOKENS = 16384
TEMPERATURE = 0.5
DEFAULT_CONCURRENT = 20
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5
REQUEST_TIMEOUT = 1200
CONNECT_TIMEOUT = 30
BATCH_DELAY_SECONDS = 1
CONTENT_FILTER_RETRY_INSTRUCTION = (
    "\n\nSafety-compliant output requirement: return only the requested JSON. "
    "Where exact_quote is required, copy only the shortest span necessary to "
    "identify the evidence. Do not repeat surrounding graphic, sexual, violent, "
    "self-harm, or otherwise sensitive details in context or explanations."
)

FATAL_ERROR_CODES = {
    "Arrearage", "InvalidApiKey", "Unauthorized",
    "AccountDisabled", "InsufficientBalance",
}

# Prompt template file mapping
PROMPT_FILE_MAPPING = {
    "characterization": "characterization.md",
    "factual_detail": "factual_detail.md",
    "narrative_style": "narrative_style.md",
    "timeline_plot": "timeline_plot.md",
    "world_building": "world_building.md",
}

# Evaluation criteria taxonomy (5 categories, 19 subtypes)
EVALUATION_CRITERIA = {
    "characterization": {
        "name": "Character Consistency",
        "sub_criteria": [
            "memory_contradictions",
            "knowledge_contradictions",
            "skill_power_fluctuations",
            "forgotten_abilities",
        ],
    },
    "factual_detail": {
        "name": "Factual & Detail Consistency",
        "sub_criteria": [
            "appearance_mismatches",
            "nomenclature_confusions",
            "quantitative_mismatches",
        ],
    },
    "narrative_style": {
        "name": "Narrative & Style",
        "sub_criteria": [
            "perspective_confusions",
            "tone_inconsistencies",
            "style_shifts",
        ],
    },
    "timeline_plot": {
        "name": "Timeline & Plot Logic",
        "sub_criteria": [
            "absolute_time_contradictions",
            "duration_timeline_contradictions",
            "simultaneity_contradictions",
            "causeless_effects",
            "causal_logic_violations",
            "abandoned_plot_elements",
        ],
    },
    "world_building": {
        "name": "World-building & Setting",
        "sub_criteria": [
            "core_rules_violations",
            "social_norms_violations",
            "geographical_contradictions",
        ],
    },
}


# Sentinel written to an error cell when the judge response could not be
# parsed (malformed output, refusal, empty). Distinct from "[]" (a trusted
# zero-error verdict) so metrics can EXCLUDE it instead of counting it as
# zero errors, which would silently deflate CED.
PARSE_FAILED = "PARSE_FAILED"


def _is_trustworthy_criteria_cell(value: Any) -> bool:
    """Whether a persisted CSV cell is a successful judge verdict."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text) and text != PARSE_FAILED


class FatalAPIError(Exception):
    """Raised when an unrecoverable API error is detected."""
    pass


# =============================================================================
# Logging
# =============================================================================

def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# =============================================================================
# Prompt Loader
# =============================================================================

def load_prompt_templates(prompts_dir: str) -> Dict[str, str]:
    """Load evaluation prompt templates from the prompts directory."""
    templates = {}
    for criteria, filename in PROMPT_FILE_MAPPING.items():
        filepath = os.path.join(prompts_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompt template not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            templates[criteria] = f.read()
    return templates


# =============================================================================
# OpenAI-Compatible Judge Client
# =============================================================================

class JudgeLLMClient:
    """Async LLM client for consistency evaluation via OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_concurrent: int,
        logger: logging.Logger,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        request_timeout: float = REQUEST_TIMEOUT,
        stream: bool = True,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logger
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.stream = stream

    async def _read_streaming_response(
        self,
        resp: aiohttp.ClientResponse,
        criteria_name: str,
        started_at: float,
    ) -> Dict[str, Any]:
        """Collect an OpenAI-compatible SSE response without discarding it.

        V4-Pro can spend several minutes producing reasoning tokens before its
        short final JSON.  Streaming makes that activity visible to the HTTP
        client and distinguishes a live long-running request from a dead one.
        Only ``content`` is returned to the evaluator; hidden reasoning is
        counted for diagnostics and never parsed as a verdict.
        """
        content_parts: List[str] = []
        reasoning_chars = 0
        chunks = 0
        finish_reason = None
        first_event_at = None

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                self.logger.warning(
                    f"[{criteria_name}] ignored malformed SSE event"
                )
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            if first_event_at is None:
                first_event_at = time.monotonic()
                self.logger.info(
                    f"[{criteria_name}] stream started after "
                    f"{first_event_at - started_at:.1f}s"
                )
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if isinstance(content, str) and content:
                content_parts.append(content)
            if isinstance(reasoning, str) and reasoning:
                reasoning_chars += len(reasoning)
            chunks += 1
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

        elapsed = time.monotonic() - started_at
        content = "".join(content_parts)
        self.logger.info(
            f"[{criteria_name}] stream finished in {elapsed:.1f}s; "
            f"chunks={chunks}, reasoning_chars={reasoning_chars}, "
            f"content_chars={len(content)}, finish_reason={finish_reason!r}"
        )
        return {
            "content": content,
            "finish_reason": finish_reason,
        }

    async def evaluate_criteria(
        self,
        session: aiohttp.ClientSession,
        prompt_template: str,
        story_content: str,
        criteria_name: str,
        target_chapter_ids: str = "",
    ) -> Dict[str, Any]:
        """Evaluate a story for one criteria category with retry logic."""
        async with self.semaphore:
            prompt = prompt_template.replace("{{ Content }}", story_content)
            prompt = prompt.replace(
                "{{ Query }}",
                f"{EVALUATION_CRITERIA[criteria_name]['name']} Analysis",
            )
            prompt = prompt.replace(
                "{{ Target Chapter IDs }}",
                target_chapter_ids,
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                # This is the judge OUTPUT budget, not the input-context size.
                # A 100k output budget needlessly consumes the model's context
                # allowance and can make an otherwise valid long-story request
                # fail before inference starts.
                "max_tokens": self.max_tokens,
                "temperature": TEMPERATURE,
                "stream": self.stream,
            }
            timeout = aiohttp.ClientTimeout(
                total=self.request_timeout,
                connect=CONNECT_TIMEOUT,
                # With streaming, continuous reasoning chunks keep the read
                # alive.  A five-minute gap is treated as a genuinely stalled
                # connection while the separate total timeout remains a hard
                # cost/time ceiling.
                sock_read=min(300.0, self.request_timeout),
            )
            content_filter_retry = False

            for retry in range(MAX_RETRIES):
                try:
                    request_prompt = prompt
                    if content_filter_retry:
                        request_prompt += CONTENT_FILTER_RETRY_INSTRUCTION
                    payload["messages"] = [{
                        "role": "user", "content": request_prompt
                    }]
                    url = f"{self.api_base}/chat/completions"
                    started_at = time.monotonic()
                    async with session.post(
                        url, json=payload, headers=headers, timeout=timeout
                    ) as resp:
                        if resp.status == 200:
                            if self.stream:
                                streamed = await self._read_streaming_response(
                                    resp, criteria_name, started_at
                                )
                                finish_reason = streamed["finish_reason"]
                                content = streamed["content"]
                            else:
                                result = await resp.json()
                                choice = result["choices"][0]
                                finish_reason = choice.get("finish_reason")
                                msg = choice["message"]
                                content = msg.get("content", "")
                            if finish_reason != "stop":
                                if finish_reason == "content_filter":
                                    content_filter_retry = True
                                raise ValueError(
                                    "judge returned incomplete finish_reason="
                                    f"{finish_reason!r}"
                                )
                            if not isinstance(content, str) or not content.strip():
                                raise ValueError("judge returned an empty response")
                            return {
                                "success": True,
                                "content": content,
                                "finish_reason": finish_reason,
                            }
                        else:
                            error_text = await resp.text()
                            self.logger.warning(
                                f"[{criteria_name}] API error "
                                f"(retry {retry+1}/{MAX_RETRIES}): "
                                f"HTTP {resp.status}"
                            )
                            try:
                                ej = json.loads(error_text)
                                code = ej.get("error", {}).get("code", "")
                                if code in FATAL_ERROR_CODES:
                                    raise FatalAPIError(
                                        f"Fatal ({code}): "
                                        f"{ej['error'].get('message', '')}"
                                    )
                            except (json.JSONDecodeError, FatalAPIError) as e:
                                if isinstance(e, FatalAPIError):
                                    raise

                            if retry < MAX_RETRIES - 1:
                                await asyncio.sleep(
                                    RETRY_DELAY_BASE * (retry + 1)
                                )
                            else:
                                return {
                                    "success": False,
                                    "error": f"HTTP {resp.status}",
                                }

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"[{criteria_name}] Timeout "
                        f"(retry {retry+1}/{MAX_RETRIES})"
                    )
                    if retry < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_BASE * (retry + 1))
                    else:
                        return {"success": False, "error": "Timeout"}

                except FatalAPIError:
                    raise

                except Exception as e:
                    self.logger.error(
                        f"[{criteria_name}] Error "
                        f"(retry {retry+1}/{MAX_RETRIES}): {e}"
                    )
                    if retry < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_BASE * (retry + 1))
                    else:
                        return {"success": False, "error": str(e)}

            return {"success": False, "error": "Max retries exceeded"}


# =============================================================================
# Response Parser
# =============================================================================

def parse_criteria_response(
    response_content: str,
    sub_criteria_list: List[str],
    criteria_name: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Parse LLM judge response and extract per-subtype error JSON arrays."""
    def _warn(msg: str):
        # A silent fallback to "[]" is indistinguishable from a genuinely
        # clean story and biases CED downward (common when a long-story verdict
        # is truncated). Surface it so it isn't mistaken for zero errors.
        preview = response_content.strip().replace("\n", " ")[:200]
        full = (
            f"[{criteria_name}] parse fallback: {msg} "
            f"(len={len(response_content)}, preview={preview!r})"
        )
        if logger:
            logger.warning(full)
        else:
            print(f"[WARN] {full}")

    try:
        # Strip a leading ```json / ``` code fence if present.
        text = response_content.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        # Extract the first balanced {...} object. This handles the common
        # case where the judge emits valid JSON followed by trailing prose
        # (which makes a plain json.loads raise "Extra data" and would
        # otherwise drop a valid verdict).
        obj = _extract_first_json_object(text)
        if obj is not None:
            try:
                parsed = json.loads(obj)
                return _extract_subcriteria(parsed, sub_criteria_list)
            except (json.JSONDecodeError, ValueError) as e:
                _warn(f"invalid criteria schema ({e}) -> PARSE_FAILED")
                return {sc: PARSE_FAILED for sc in sub_criteria_list}

        # No parseable JSON object. This is a genuine failure: malformed
        # output, a truncated response, a refusal (e.g. "I can't provide that"),
        # or an empty body. Mark every subtype as PARSE_FAILED so metrics
        # can exclude it rather than treat it as a zero-error verdict.
        if not text:
            _warn("empty response body -> PARSE_FAILED")
        else:
            _warn("no parseable JSON object -> PARSE_FAILED")
        return {sc: PARSE_FAILED for sc in sub_criteria_list}

    except Exception as e:
        _warn(f"unrecoverable parse error ({e}) -> PARSE_FAILED")
        return {sc: PARSE_FAILED for sc in sub_criteria_list}


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring, or None.

    Tracks string literals and escapes so braces inside quoted values don't
    throw off the depth count.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None  # unbalanced / truncated


def _extract_subcriteria(
    parsed: dict, sub_criteria_list: List[str]
) -> Dict[str, str]:
    """Validate and serialize every expected subtype.

    A missing key or malformed value is an unavailable verdict, not evidence
    that the story has zero errors for that subtype. The category is therefore
    rejected as a unit by ``parse_criteria_response``.
    """
    if not isinstance(parsed, dict):
        raise ValueError("top-level response must be a JSON object")

    # Also accept ``duration_contradictions`` as an alternative key for the
    # taxonomy/CSV column ``duration_timeline_contradictions``, and always
    # normalize it to the canonical taxonomy key.
    legacy_aliases = {
        "duration_timeline_contradictions": ("duration_contradictions",),
    }

    results = {}
    for sc in sub_criteria_list:
        source_key = sc
        if source_key not in parsed:
            source_key = next(
                (alias for alias in legacy_aliases.get(sc, ()) if alias in parsed),
                source_key,
            )
        if source_key not in parsed:
            raise ValueError(f"missing subtype {sc!r}")
        val = parsed[source_key]
        if not isinstance(val, list):
            raise ValueError(f"subtype {sc!r} must be a JSON array")
        for item_index, item in enumerate(val):
            if not isinstance(item, dict):
                raise ValueError(
                    f"subtype {sc!r} item {item_index} must be an object"
                )
            quote = item.get("exact_quote")
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError(
                    f"subtype {sc!r} item {item_index} lacks exact_quote"
                )
        results[sc] = json.dumps(val, ensure_ascii=False)
    return results


# =============================================================================
# ConStory-Checker Evaluator
# =============================================================================

class ConStoryChecker:
    """Main evaluation pipeline: judge each story across all 5 criteria."""

    def __init__(
        self,
        client: JudgeLLMClient,
        prompt_templates: Dict[str, str],
        story_column: str,
        logger: logging.Logger,
    ):
        self.client = client
        self.templates = prompt_templates
        self.story_column = story_column
        self.logger = logger

    async def evaluate_single(
        self,
        session: aiohttp.ClientSession,
        story_data: Dict[str, Any],
        existing_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate missing categories while preserving successful prior cells."""
        sid = story_data.get("id", story_data.get("original_id", "unknown"))
        story_text = story_data.get(self.story_column, "")
        target_chapter_ids = str(
            story_data.get("target_chapter_ids", "")
        )

        self.logger.info(f"Evaluating story id={sid}")

        result = dict(story_data)
        result["evaluation_timestamp"] = datetime.now().isoformat()
        result["evaluation_status"] = "in_progress"
        result["judge_model"] = self.client.model

        # An unevaluated cell is unknown, never a trusted zero-error verdict.
        for cat, cfg in EVALUATION_CRITERIA.items():
            for sc in cfg["sub_criteria"]:
                result[f"{cat}_{sc}"] = PARSE_FAILED

        completed_cats = set()
        if isinstance(existing_result, dict):
            for cat, cfg in EVALUATION_CRITERIA.items():
                fields = [f"{cat}_{sc}" for sc in cfg["sub_criteria"]]
                values = [existing_result.get(field) for field in fields]
                if all(_is_trustworthy_criteria_cell(value) for value in values):
                    completed_cats.add(cat)
                    for field, value in zip(fields, values):
                        result[field] = value
            if completed_cats:
                self.logger.info(
                    f"  id={sid}: preserving {len(completed_cats)}/5 "
                    "completed criteria from prior output"
                )

        try:
            # Evaluate all 5 categories concurrently (real parallelism, not
            # sequential await over raw coroutines).
            cats = [
                cat for cat in EVALUATION_CRITERIA
                if cat not in completed_cats
            ]
            coros = [
                self.client.evaluate_criteria(
                    session,
                    self.templates[cat],
                    story_text,
                    cat,
                    target_chapter_ids=target_chapter_ids,
                )
                for cat in cats
            ]
            responses = await asyncio.gather(*coros, return_exceptions=True)

            ok = len(completed_cats)
            errors = {}
            for cat, resp in zip(cats, responses):
                if isinstance(resp, FatalAPIError):
                    # Re-raise to let outer layer mark this story as fatal.
                    raise resp
                if isinstance(resp, Exception):
                    self.logger.error(f"[{cat}] task error: {resp}")
                    resp = {"success": False, "error": str(resp)}

                cfg = EVALUATION_CRITERIA[cat]
                if resp["success"]:
                    parsed = parse_criteria_response(
                        resp["content"], cfg["sub_criteria"], cat,
                        logger=self.logger,
                    )
                    for sc in cfg["sub_criteria"]:
                        result[f"{cat}_{sc}"] = parsed.get(sc, PARSE_FAILED)
                    if any(
                        value == PARSE_FAILED for value in parsed.values()
                    ):
                        errors[cat] = "judge response parse/schema failure"
                    else:
                        ok += 1
                else:
                    err = resp.get("error", "Unknown")
                    errors[cat] = err
                    for sc in cfg["sub_criteria"]:
                        result[f"{cat}_{sc}"] = PARSE_FAILED

            result["evaluation_status"] = (
                "completed" if ok == 5 else f"partial_{ok}_5"
            )
            result["criteria_completed"] = ok
            result["criteria_failed"] = 5 - ok
            result["evaluation_errors"] = json.dumps(
                errors, ensure_ascii=False
            )
            self.logger.info(f"  id={sid}: {ok}/5 criteria completed")

        except FatalAPIError:
            raise
        except Exception as e:
            self.logger.error(f"Critical error for id={sid}: {e}")
            result["evaluation_status"] = f"error: {e}"
            result["criteria_completed"] = 0
            result["criteria_failed"] = 5
            result["evaluation_errors"] = json.dumps(
                {"pipeline": str(e)}, ensure_ascii=False
            )

        return result

    async def run(
        self,
        input_path: str,
        output_path: str,
        model_name: str,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        resume: bool = True,
    ) -> str:
        """Run the full evaluation pipeline."""
        # Load stories
        df = pd.read_parquet(input_path)
        self.logger.info(f"Loaded {len(df)} stories from {input_path}")

        if self.story_column not in df.columns:
            raise ValueError(
                f"Column '{self.story_column}' not found. "
                f"Available: {list(df.columns)}"
            )

        if end_idx is None:
            end_idx = len(df)
        df_slice = df.iloc[start_idx:end_idx]

        # Resume support
        results = []
        processed_ids = set()
        if resume and os.path.exists(output_path):
            try:
                existing = pd.read_csv(output_path)
                if "evaluation_status" in existing.columns:
                    completed = existing[
                        existing["evaluation_status"] == "completed"
                    ]
                    results = completed.to_dict("records")
                    id_col = "id" if "id" in completed.columns else "original_id"
                    processed_ids = set(completed[id_col].astype(str))
                else:
                    results = existing.to_dict("records")
                self.logger.info(f"Resuming: {len(results)} completed")
            except Exception as e:
                self.logger.warning(f"Could not load resume file: {e}")

        # Filter remaining
        to_eval = []
        id_col = "id" if "id" in df_slice.columns else "original_id"
        for _, row in df_slice.iterrows():
            if str(row.get(id_col, "")) not in processed_ids:
                to_eval.append(row.to_dict())

        self.logger.info(f"Evaluating {len(to_eval)} stories")

        # Async evaluation
        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=self.client.request_timeout + 60)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            sem = asyncio.Semaphore(self.client.semaphore._value)

            async def _with_sem(data):
                async with sem:
                    return await self.evaluate_single(session, data)

            pbar = tqdm(
                total=len(to_eval),
                desc=f"Judging ({model_name})",
                unit="story",
            )

            batch_size = self.client.semaphore._value * 2
            for i in range(0, len(to_eval), batch_size):
                batch = to_eval[i : i + batch_size]
                batch_tasks = [_with_sem(s) for s in batch]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                for res in batch_results:
                    if isinstance(res, FatalAPIError):
                        self._save(results, output_path)
                        raise res
                    if isinstance(res, Exception):
                        self.logger.error(f"Batch error: {res}")
                        continue
                    results.append(res)
                    pbar.update(1)

                self._save(results, output_path)
                await asyncio.sleep(BATCH_DELAY_SECONDS)

            pbar.close()

        self._save(results, output_path)
        self.logger.info(
            f"Evaluation complete: {len(results)} stories -> {output_path}"
        )
        return output_path

    def _save(self, results: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ConStory-Checker: Evaluate story consistency using LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Judge with OpenAI o4-mini (default)
  python -m constory.judge \\
      --input data/stories/my_model.parquet \\
      --story-column generated_story \\
      --model-name my_model \\
      --concurrent 20

  # Judge with a different model or self-hosted endpoint
  python -m constory.judge \\
      --input data/stories/llama3.parquet \\
      --story-column generated_story \\
      --model-name llama3 \\
      --judge-model qwen3-235b \\
      --api-base http://localhost:8000/v1 \\
      --api-key token-abc123
        """,
    )
    parser.add_argument("--input", required=True, help="Input stories parquet")
    parser.add_argument(
        "--output-dir", default="output", help="Output directory for CSV results"
    )
    parser.add_argument("--story-column", required=True, help="Story text column name")
    parser.add_argument("--model-name", required=True, help="Name for output files")
    parser.add_argument(
        "--judge-model", default="o4-mini", help="Judge model name (default: o4-mini)"
    )
    parser.add_argument(
        "--api-base",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--prompts-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts"),
        help="Directory containing prompt templates",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum judge output tokens per category "
            f"(default: {DEFAULT_MAX_TOKENS})"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1200.0,
        help="Per judge request timeout in seconds (default: 1200)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "API key required. Set --api-key or $OPENAI_API_KEY env variable."
        )
    if args.concurrent <= 0:
        raise ValueError("--concurrent must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger("judge", f"logs/judge_{ts}.log", args.log_level)

    # Load prompt templates
    templates = load_prompt_templates(args.prompts_dir)
    logger.info(f"Loaded {len(templates)} prompt templates from {args.prompts_dir}")

    output_file = os.path.join(
        args.output_dir,
        f"judge_{args.model_name}_{args.start}_{args.end or 'end'}_{ts}.csv",
    )

    print("=" * 70)
    print("ConStory-Checker: Consistency Evaluation Pipeline")
    print("=" * 70)
    print(f"  Judge model:   {args.judge_model}")
    print(f"  API base:      {args.api_base}")
    print(f"  Input:         {args.input}")
    print(f"  Story column:  {args.story_column}")
    print(f"  Model name:    {args.model_name}")
    print(f"  Output:        {output_file}")
    print(f"  Range:         {args.start} to {args.end or 'end'}")
    print(f"  Concurrent:    {args.concurrent}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Resume:        {'off' if args.no_resume else 'on'}")
    print("=" * 70)

    client = JudgeLLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.judge_model,
        max_concurrent=args.concurrent,
        logger=logger,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
    )

    checker = ConStoryChecker(
        client=client,
        prompt_templates=templates,
        story_column=args.story_column,
        logger=logger,
    )

    asyncio.run(
        checker.run(
            input_path=args.input,
            output_path=output_file,
            model_name=args.model_name,
            start_idx=args.start,
            end_idx=args.end,
            resume=not args.no_resume,
        )
    )

    print(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    main()
