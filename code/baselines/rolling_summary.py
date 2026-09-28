#!/usr/bin/env python
# coding: utf-8
"""RollingSummary: chapter-by-chapter writing that carries context forward
through a model-generated rolling summary.

Each chapter sees only the full outline and the story-so-far summary up to the
previous chapter. It does not read earlier chapter text and does not maintain
StructuredState's structured state.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from generation_common import (  # noqa: E402
    DEFAULT_DYNAMIC_TOKEN_MINIMUM,
    DEFAULT_DYNAMIC_TOKEN_OVERHEAD,
    DEFAULT_DYNAMIC_TOKEN_RATIO_EN,
    DEFAULT_DYNAMIC_TOKEN_RATIO_ZH,
    LENGTH_CONTROL_MODES,
    LONG_FORM_MIN_WORDS,
    LengthValidation,
    TOKEN_CONTROL_MODES,
    chat_completion,
    count_words,
    detect_language,
    dynamic_max_tokens,
    extract_json,
    load_completed_ids,
    retry_messages,
    setup_logger,
    timestamp,
    update_failure_state,
    validate_generation,
    validation_dict,
    word_bounds,
)
from agent_core import OutlineCache  # noqa: E402


METHOD_ID = "rolling_summary"
METHOD_NAME = "RollingSummary"
CHECKPOINT_VERSION = 11
SUBMISSION_MODES = ("plain",)

DEFAULT_MAX_TOKENS = 16384
DEFAULT_OUTLINE_MAX_TOKENS = 16384
DEFAULT_SUMMARY_MAX_TOKENS = 16384
DEFAULT_ROLLING_DYNAMIC_TOKEN_OVERHEAD = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENT = 8
DEFAULT_WORD_COUNT = 10000
DEFAULT_MAX_LENGTH_ATTEMPTS = 50
DEFAULT_AUTO_RESUMES = 3
MAX_OUTLINE_ATTEMPTS = 5
MAX_SUMMARY_ATTEMPTS = 5
DYNAMIC_TOKEN_RETRY_MIN_STEP = 1024
DYNAMIC_TOKEN_RETRY_GROWTH = 1.5

CHAPTER_ANCHORS = (
    (0, 0),
    (10000, 10),
    (20000, 15),
    (50000, 25),
    (100000, 40),
    (1000000, 100),
)

SYSTEM_PROMPTS = {
    "outline_zh": "你是一位经验丰富的小说家，善于规划完整的长篇叙事。",
    "outline_en": (
        "You are an experienced novelist who carefully plans complete "
        "long-form narratives."
    ),
    "chapter_zh": (
        "你是一位经验丰富的长篇小说家。请根据完整大纲和故事至今摘要，"
        "写出当前章的完整正文。"
    ),
    "chapter_en": (
        "You are an experienced long-form novelist. Write the complete current "
        "chapter from the full outline and the story-so-far summary."
    ),
    "summary_zh": (
        "你负责维护长篇小说的滚动摘要。摘要应简洁，但必须保留后续写作真正"
        "需要的人物处境、关键事件、关系变化、时间线和未解决情节。压缩已经"
        "稳定的早期信息，不要复述场景、对话或描写。"
    ),
    "summary_en": (
        "Maintain a compact rolling summary of a long novel. Preserve the "
        "character situations, key events, relationship changes, timeline, "
        "and unresolved threads needed for later chapters. Compress settled "
        "earlier information instead of retelling scenes, dialogue, or prose."
    ),
}


def recommend_chapter_count(word_count: int) -> int:
    """Suggest an adjustable chapter count using StructuredState's anchors."""
    if word_count <= 0:
        return 0
    if word_count >= CHAPTER_ANCHORS[-1][0]:
        return CHAPTER_ANCHORS[-1][1]
    for index in range(len(CHAPTER_ANCHORS) - 1):
        words0, chapters0 = CHAPTER_ANCHORS[index]
        words1, chapters1 = CHAPTER_ANCHORS[index + 1]
        if words0 <= word_count <= words1:
            ratio = (word_count - words0) / (words1 - words0)
            return max(1, math.ceil(
                chapters0 + ratio * (chapters1 - chapters0)
            ))
    return CHAPTER_ANCHORS[-1][1]


def next_dynamic_token_budget(current: int, ceiling: int) -> int:
    """Grow the budget quickly after truncation, without exceeding
    StructuredState's global hard ceiling.

    Qwen's thinking-token usage varies far more than its prose-token usage. A
    fixed 128-token increment can take dozens of failures to gain enough room,
    which shows up as a "stalled" job. The budget therefore grows by at least
    1024 tokens and by a factor of 1.5, reaching the ceiling within a few
    retries.
    """
    grown = math.ceil(current * DYNAMIC_TOKEN_RETRY_GROWTH)
    stepped = current + DYNAMIC_TOKEN_RETRY_MIN_STEP
    return min(ceiling, max(grown, stepped))


def normalize_chapter_targets(
        outline: List[Dict[str, Any]],
        total_target: int,
) -> List[Dict[str, Any]]:
    """Rescale the outline's relative word allocations so that chapter
    targets sum exactly to the total target."""
    planned = [int(chapter["word_count"]) for chapter in outline]
    planned_total = sum(planned)
    raw = [value * total_target / planned_total for value in planned]
    normalized = [max(1, math.floor(value)) for value in raw]
    difference = total_target - sum(normalized)
    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index] - math.floor(raw[index]),
        reverse=difference > 0,
    )
    step = 1 if difference > 0 else -1
    cursor = 0
    while difference:
        index = order[cursor % len(order)]
        if step > 0 or normalized[index] > 1:
            normalized[index] += step
            difference -= step
        cursor += 1

    result = []
    for index, (chapter, target) in enumerate(zip(outline, normalized)):
        item = dict(chapter)
        item["id"] = index
        item["planned_word_count"] = int(chapter["word_count"])
        item["word_count"] = target
        result.append(item)
    return result


def validate_outline(raw: Any, total_target: int) -> List[Dict[str, Any]]:
    """Check outline structure, single-chapter long-form outlines and total
    length, then normalize chapter IDs and targets."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("outline must be a non-empty JSON array")
    if total_target >= LONG_FORM_MIN_WORDS and len(raw) == 1:
        raise ValueError("a long-form outline cannot contain only one chapter")

    cleaned = []
    for index, chapter in enumerate(raw):
        if not isinstance(chapter, dict):
            raise ValueError(f"chapter {index} must be an object")
        target = chapter.get("word_count")
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise ValueError(
                f"chapter {index} word_count must be a positive integer"
            )
        cleaned.append({
            "id": index,
            "name": str(chapter.get("name") or f"Chapter {index + 1}"),
            "description": str(chapter.get("description") or ""),
            "word_count": target,
        })

    planned_total = sum(chapter["word_count"] for chapter in cleaned)
    minimum, maximum = word_bounds(total_target)
    if not minimum <= planned_total <= maximum:
        raise ValueError(
            f"outline plans {planned_total} words; accepted total is "
            f"{minimum}-{maximum}"
        )
    return normalize_chapter_targets(cleaned, total_target)


def validate_frozen_outline(
        raw: Any, total_target: int) -> List[Dict[str, Any]]:
    """Validate a cached outline without changing IDs or word targets."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("cached outline must be a non-empty JSON array")
    if total_target >= LONG_FORM_MIN_WORDS and len(raw) == 1:
        raise ValueError("a long-form cached outline cannot contain one chapter")
    for index, chapter in enumerate(raw):
        if not isinstance(chapter, dict):
            raise ValueError(f"cached chapter {index} must be an object")
        if chapter.get("id") != index:
            raise ValueError(
                f"cached chapter id must remain {index}; got "
                f"{chapter.get('id')!r}"
            )
        if not isinstance(chapter.get("name"), str):
            raise ValueError(f"cached chapter {index} name must be a string")
        if not isinstance(chapter.get("description"), str):
            raise ValueError(
                f"cached chapter {index} description must be a string"
            )
        target = chapter.get("word_count")
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise ValueError(
                f"cached chapter {index} word_count must be a positive integer"
            )
    planned_total = sum(chapter["word_count"] for chapter in raw)
    minimum, maximum = word_bounds(total_target)
    if not minimum <= planned_total <= maximum:
        raise ValueError(
            f"cached outline plans {planned_total} words; accepted total is "
            f"{minimum}-{maximum}"
        )
    return copy.deepcopy(raw)


def outline_prompt(prompt: str, total_target: int, language: str) -> str:
    """Build the single-stage chapter-outline prompt; descriptions have no
    length limit."""
    recommended = recommend_chapter_count(total_target)
    per_chapter = round(total_target / recommended)
    minimum, maximum = word_bounds(total_target)
    if language == "zh":
        return (
            f"故事要求：\n{prompt}\n\n"
            f"请规划一部目标约 {total_target} 字的完整小说。建议约 "
            f"{recommended} 章、平均每章约 {per_chapter} 字；剧情需要时可"
            "适当调整章节数。\n"
            "只输出 JSON 数组。每章必须包含：id、name、description、"
            "word_count。id 从 0 开始连续递增；word_count 为正整数；"
            f"所有章节 word_count 之和必须在 {minimum}–{maximum} 字。"
        )
    return (
        f"Story prompt:\n{prompt}\n\n"
        f"Plan a complete novel targeting about {total_target} words. "
        f"Approximately {recommended} chapters averaging {per_chapter} words "
        "is recommended; adjust the chapter count when the plot requires it.\n"
        "Output only a JSON array. Every chapter must contain id, name, "
        "description, and word_count. IDs start at 0 and increase "
        "consecutively; word_count is a positive integer; all chapter "
        f"word_count values must sum to {minimum}-{maximum} words."
    )


def chapter_prompt(
        prompt: str,
        outline: Sequence[Dict[str, Any]],
        chapter: Dict[str, Any],
        previous_summary: Optional[str],
        language: str,
        target_ratio_zh: float = 1.0,
        target_ratio_en: float = 1.0,
) -> str:
    """Build the chapter prompt; it requests only the current chapter's prose
    and never asks for the summary in the same output."""
    minimum, maximum = word_bounds(chapter["word_count"])
    outline_text = json.dumps(outline, ensure_ascii=False, indent=2)
    summary = previous_summary or ("无；这是第一章。" if language == "zh"
                                   else "None; this is the first chapter.")
    if language == "zh":
        calibrated_instruction = ""
        target_instruction = (
            "以给定的目标字数本身为写作目标，不要瞄准允许范围的上界。"
            "大约在目标字数的 85%–90% 时开始解决当前章节的冲突，"
            "并在目标值附近自然收束；不要为了接近上界而填充或续写。"
        )
        if target_ratio_zh < 1.0:
            working_target = max(1, round(
                chapter["word_count"] * target_ratio_zh
            ))
            hard_end = max(
                working_target,
                round(chapter["word_count"] * (target_ratio_zh + 0.125)),
            )
            calibrated_instruction = (
                f"为校正当前模型稳定的中文过度生成，本次以 {working_target} "
                f"字为内部写作目标，并务必在 {hard_end} 字以内完整结束；"
                f"正式验收仍严格使用原始范围 {minimum}–{maximum} 字。"
            )
            target_instruction = (
                f"原始 {chapter['word_count']} 字仅是冻结大纲的计划字段和"
                f"验收依据，不是本轮应追逐的生成长度。本轮只以 "
                f"{working_target} 字为写作目标，大约在该目标的 85%–90% "
                f"时开始解决冲突，并在 {hard_end} 字以内完整结束；不要再"
                "向原始目标或允许上界扩写。"
            )
        return (
            f"# 原始故事提示\n{prompt}\n\n"
            f"# 完整冻结大纲\n{outline_text}\n\n"
            f"# 故事至今摘要\n{summary}\n\n"
            "# 当前任务\n"
            f"章节 ID：{chapter['id']}\n"
            f"章节名：{chapter['name']}\n"
            f"章节大纲：{chapter['description']}\n"
            f"正文目标为 {chapter['word_count']} 字，允许范围为 "
            f"{minimum}–{maximum} 字。"
            "请严格控制在该目标字数的 ±20% 以内。\n\n"
            f"{calibrated_instruction}"
            "直接输出当前章的完整小说正文，不要输出摘要、字数说明、写作计划"
            "或其他元文本。"
            "即使你能看到完整大纲，也只能写当前章节；不要开始、合并或"
            "提前写任何后续章节。当前章节完成后立即结束回答。"
            f"{target_instruction}"
            "章节大纲只是计划，请扩展为具有场景、对话、感官细节和内心"
            "活动的正文。"
        )
    calibrated_instruction = ""
    target_instruction = (
        "Aim for the stated target itself, not the top of the accepted "
        "range. Begin resolving the current chapter by roughly 85-90% of "
        "the target and finish it naturally near the target; do not pad "
        "toward the upper bound. "
    )
    if target_ratio_en < 1.0:
        working_target = max(1, round(
            chapter["word_count"] * target_ratio_en
        ))
        hard_end = max(
            working_target,
            round(chapter["word_count"] * (target_ratio_en + 0.125)),
        )
        calibrated_instruction = (
            "To correct this model's stable English over-generation, use "
            f"{working_target} words as the internal writing target for "
            f"this attempt and finish the chapter within {hard_end} words. "
            f"Formal acceptance still uses the original {minimum}-{maximum} "
            "word range. "
        )
        target_instruction = (
            f"The original {chapter['word_count']}-word value is only the "
            "frozen outline field and formal acceptance reference; do not "
            f"pursue it as this attempt's writing length. Aim only for "
            f"{working_target} words, begin resolving the chapter by roughly "
            f"85-90% of that target, and finish completely within {hard_end} "
            "words. Do not expand toward the original target or accepted "
            "upper bound. "
        )
    return (
        f"# Original Story Prompt\n{prompt}\n\n"
        f"# Full Frozen Outline\n{outline_text}\n\n"
        f"# Story-so-far summary\n{summary}\n\n"
        "# Current task\n"
        f"Chapter ID: {chapter['id']}\n"
        f"Chapter title: {chapter['name']}\n"
        f"Chapter outline: {chapter['description']}\n"
        f"The chapter targets {chapter['word_count']} words; the accepted "
        f"range is {minimum}-{maximum} words. Keep the chapter strictly "
        "within ±20% of this target word count.\n\n"
        f"{calibrated_instruction}"
        "Output only the complete prose of the current chapter directly, "
        "without a summary, word-count commentary, writing plan, or other "
        "meta-text. "
        "Even though the full outline is visible, write only the current "
        "chapter. Do not begin, merge, or pre-write any later chapter; end "
        "the response as soon as the current chapter is complete. "
        f"{target_instruction}"
        "The chapter outline is only a plan; expand it into scenes, "
        "dialogue, sensory detail, and interiority."
    )


def summary_prompt(
        prompt: str,
        previous_summary: Optional[str],
        chapter: Dict[str, Any],
        content: str,
        language: str,
        summary_word_limit: int = 0,
) -> str:
    """Build the prompt that updates the rolling summary with the new chapter.

    The summary is internal method state and does not count toward story
    length.
    """
    previous = previous_summary or ("无" if language == "zh" else "None")
    if language == "zh":
        instruction = (
            f"原始故事要求：\n{prompt}\n\n"
            f"上一版故事至今摘要：\n{previous}\n\n"
            f"刚完成的第 {chapter['id']} 章《{chapter['name']}》：\n"
            f"{content}\n\n"
            "请直接输出更新后的故事至今摘要。不要评价本章，不要加入写作建议"
            "或其他元文本。"
        )
        if summary_word_limit == 0:
            instruction += (
                "请自行决定摘要所需长度：完整保留后续章节可能需要的信息，"
                "同时保持简洁，且不要续写小说正文。"
            )
        return instruction
    instruction = (
        f"Original story prompt:\n{prompt}\n\n"
        f"Previous story-so-far summary:\n{previous}\n\n"
        f"Newly completed Chapter {chapter['id']}, {chapter['name']}:\n"
        f"{content}\n\n"
        "Output only the updated story-so-far summary directly. Do not "
        "critique the chapter, offer writing advice, or add other meta-text. "
    )
    if summary_word_limit > 0:
        return (
            instruction
            + "Keep the complete updated summary at no more than "
            + f"{summary_word_limit} words; compress older details rather "
            + "than continuing to write prose."
        )
    return (
        instruction
        + "Choose the summary length needed to preserve all information "
        + "that may matter to later chapters, while remaining concise and "
        + "never continuing the story prose."
    )


def checkpoint_fingerprint(prompt: str, target_words: int) -> str:
    payload = (
        f"{CHECKPOINT_VERSION}\0{METHOD_ID}\0{target_words}\0{prompt}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RollingSummaryWriter:
    """Generate chapters in outline order, passing earlier context only
    through the rolling summary."""

    def __init__(
            self,
            client: AsyncOpenAI,
            model: str,
            max_tokens: int,
            outline_max_tokens: int,
            summary_max_tokens: int,
            temperature: float,
            word_count: int,
            logger: logging.Logger,
            token_control: str,
            length_control_mode: str,
            max_length_attempts: int,
            checkpoint_dir: Optional[str],
            dynamic_token_ratio_en: float,
            dynamic_token_ratio_zh: float,
            dynamic_token_overhead: int,
            dynamic_token_minimum: int,
            transport_retries: int,
            chapter_target_ratio_zh: float = 1.0,
            chapter_target_ratio_en: float = 1.0,
            disable_thinking: bool = False,
            outline_cache_dir: Optional[str] = None,
            outline_cache_policy: str = "off",
            outline_cache_model_id: Optional[str] = None,
            outline_cache_lock_timeout: float = 7200.0,
            summary_word_limit: int = 0):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.outline_max_tokens = outline_max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.summary_word_limit = summary_word_limit
        self.temperature = temperature
        self.word_count = word_count
        self.logger = logger
        self.token_control = token_control
        self.length_control_mode = length_control_mode
        self.max_length_attempts = max_length_attempts
        self.checkpoint_dir = checkpoint_dir
        self.dynamic_token_ratio_en = dynamic_token_ratio_en
        self.dynamic_token_ratio_zh = dynamic_token_ratio_zh
        self.dynamic_token_overhead = dynamic_token_overhead
        self.dynamic_token_minimum = dynamic_token_minimum
        self.chapter_target_ratio_zh = chapter_target_ratio_zh
        self.chapter_target_ratio_en = chapter_target_ratio_en
        # Mirror the OpenAI client's retry budget. The argument counts extra
        # retries, so the total number of calls is at least 1; passing 0 adds
        # no application-level retries.
        self.transport_retries = max(1, transport_retries)
        self.disable_thinking = disable_thinking
        if outline_cache_policy not in ("off", "require"):
            raise ValueError(
                "RollingSummary supports only off or require cache policy; "
                "it must never populate the shared planning cache"
            )
        self.outline_cache = (
            OutlineCache(
                cache_dir=outline_cache_dir,
                policy="require",
                model_id=outline_cache_model_id or model,
                model_request_name=model,
                word_count=word_count,
                max_tokens=outline_max_tokens,
                temperature=temperature,
                disable_thinking=disable_thinking,
                lock_timeout=outline_cache_lock_timeout,
            )
            if outline_cache_policy == "require" else None
        )

    def chapter_budget(self, target_words: int, language: str) -> int:
        if self.token_control == "fixed":
            return self.max_tokens
        return dynamic_max_tokens(
            target_words,
            language,
            self.max_tokens,
            ratio_en=self.dynamic_token_ratio_en,
            ratio_zh=self.dynamic_token_ratio_zh,
            overhead=self.dynamic_token_overhead,
            minimum=self.dynamic_token_minimum,
        )

    async def generate_outline(
            self, prompt: str, language: str) -> List[Dict[str, Any]]:
        base = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[f"outline_{language}"],
            },
            {
                "role": "user",
                "content": outline_prompt(prompt, self.word_count, language),
            },
        ]
        messages = base
        for attempt in range(1, MAX_OUTLINE_ATTEMPTS + 1):
            try:
                response = await chat_completion(
                    self.client,
                    self.model,
                    messages,
                    self.outline_max_tokens,
                    self.temperature,
                    self.logger,
                    retries=self.transport_retries,
                    disable_thinking=self.disable_thinking,
                )
                if response.finish_reason != "stop" or response.token_limit_reached:
                    raise ValueError(
                        f"incomplete outline: finish_reason="
                        f"{response.finish_reason!r}, token_limit="
                        f"{response.token_limit_reached}"
                    )
                outline = validate_outline(
                    extract_json(response.content), self.word_count
                )
                self.logger.info(
                    "Outline accepted: %d chapters, normalized targets total %d words",
                    len(outline),
                    sum(chapter["word_count"] for chapter in outline),
                )
                return outline
            except Exception as error:
                self.logger.warning(
                    "Outline generation attempt %d/%d failed: %s",
                    attempt, MAX_OUTLINE_ATTEMPTS, error,
                )
                if attempt == MAX_OUTLINE_ATTEMPTS:
                    raise
                feedback = (
                    f"上一次大纲不合格：{error}。请重新输出完整 JSON 数组。"
                    if language == "zh" else
                    f"The previous outline failed validation: {error}. "
                    "Regenerate the complete JSON array."
                )
                messages = base + [{"role": "user", "content": feedback}]
        raise RuntimeError("outline attempts exhausted")

    async def generate_chapter(
            self,
            prompt: str,
            outline: List[Dict[str, Any]],
            chapter: Dict[str, Any],
            previous_summary: Optional[str],
            language: str,
            accepted_word_bounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        base = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[f"chapter_{language}"],
            },
            {
                "role": "user",
                "content": chapter_prompt(
                    prompt,
                    outline,
                    chapter,
                    previous_summary,
                    language,
                    self.chapter_target_ratio_zh,
                    self.chapter_target_ratio_en,
                ),
            },
        ]
        if accepted_word_bounds is not None:
            accepted_min, accepted_max = accepted_word_bounds
            if language == "zh":
                bound_instruction = (
                    "这是最后一章。为使全文通过总字数校验，本章必须严格为 "
                    f"{accepted_min}-{accepted_max} 字；其他章节目标不变。"
                )
            else:
                bound_instruction = (
                    "This is the final chapter. To keep the complete story "
                    "within its global word-count bound, this chapter must be "
                    f"strictly between {accepted_min} and {accepted_max} words. "
                    "All other chapter targets remain unchanged."
                )
            base.append({"role": "user", "content": bound_instruction})
        messages = base
        failure = None
        attempts = []
        budget = self.chapter_budget(chapter["word_count"], language)
        for attempt in range(1, self.max_length_attempts + 1):
            response = await chat_completion(
                self.client,
                self.model,
                messages,
                budget,
                self.temperature,
                self.logger,
                retries=self.transport_retries,
                disable_thinking=self.disable_thinking,
            )
            content = response.content
            validation = validate_generation(
                content,
                response,
                chapter["word_count"],
                accepted_finish_reasons=("stop",),
            )
            if validation.ok and accepted_word_bounds is not None:
                accepted_min, accepted_max = accepted_word_bounds
                if not accepted_min <= validation.actual_words <= accepted_max:
                    direction = (
                        f"too short: {validation.actual_words} < {accepted_min}"
                        if validation.actual_words < accepted_min else
                        f"too long: {validation.actual_words} > {accepted_max}"
                    )
                    validation = LengthValidation(
                        ok=False,
                        actual_words=validation.actual_words,
                        target_words=chapter["word_count"],
                        min_words=accepted_min,
                        max_words=accepted_max,
                        finish_reason=validation.finish_reason,
                        token_limit_reached=validation.token_limit_reached,
                        reason=direction,
                    )
            attempts.append({
                "attempt": attempt,
                "validation": validation_dict(validation),
                "requested_max_tokens": response.requested_max_tokens,
                "completion_tokens": response.completion_tokens,
                "submission_mode": "plain",
                "tools_passed": False,
            })
            self.logger.info(
                "Chapter %s attempt=%d: %d words, finish=%r, tokens=%s/%s, "
                "token_limit=%s, accepted=%s",
                chapter["id"],
                attempt,
                validation.actual_words,
                validation.finish_reason,
                response.completion_tokens,
                response.requested_max_tokens,
                response.token_limit_reached,
                validation.ok,
            )
            if validation.ok:
                return content, {
                    "token_control": self.token_control,
                    "length_control_mode": self.length_control_mode,
                    "submission_mode": "plain",
                    "tools_passed": False,
                    "disable_thinking": self.disable_thinking,
                    "chapter_target_ratio_zh": self.chapter_target_ratio_zh,
                    "chapter_target_ratio_en": self.chapter_target_ratio_en,
                    "initial_max_tokens": attempts[0]["requested_max_tokens"],
                    "final_max_tokens": budget,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                }
            failure = update_failure_state(
                self.length_control_mode,
                failure,
                content,
                validation,
            )
            messages = retry_messages(base, failure, language)
            if language == "zh" and self.chapter_target_ratio_zh < 1.0:
                working_target = max(1, round(
                    chapter["word_count"] * self.chapter_target_ratio_zh
                ))
                hard_end = max(
                    working_target,
                    round(chapter["word_count"] * (
                        self.chapter_target_ratio_zh + 0.125
                    )),
                )
                messages[-1]["content"] += (
                    f" 上文中的原始 {chapter['word_count']} 字只是冻结大纲"
                    f"字段和验收依据，不是本轮生成目标；不要以它为写作"
                    f"长度。本轮只以 {working_target} 字为内部写作目标，"
                    f"并务必在 {hard_end} 字以内完整结束。正式验收范围"
                    "不变。"
                )
            if language == "en" and self.chapter_target_ratio_en < 1.0:
                working_target = max(1, round(
                    chapter["word_count"] * self.chapter_target_ratio_en
                ))
                hard_end = max(
                    working_target,
                    round(chapter["word_count"] * (
                        self.chapter_target_ratio_en + 0.125
                    )),
                )
                messages[-1]["content"] += (
                    f" The original {chapter['word_count']}-word value is "
                    "only the frozen outline field and formal acceptance "
                    "reference, not this attempt's writing target. Aim only "
                    f"for {working_target} words and finish completely "
                    f"within {hard_end} words. Formal acceptance remains "
                    "unchanged."
                )
            # The dynamic budget estimate fluctuates with reasoning volume and
            # the prose token/word ratio. Raise the next draft's budget only
            # when the response hit the token limit and the text is not yet
            # above the allowed maximum; keep the budget for over-long drafts
            # so they are not amplified.
            if (
                self.token_control == "dynamic"
                and response.token_limit_reached
                and validation.actual_words <= validation.max_words
            ):
                next_budget = next_dynamic_token_budget(
                    budget,
                    self.max_tokens,
                )
                if next_budget > budget:
                    self.logger.info(
                        "Chapter %s next-draft dynamic budget: %d -> %d tokens",
                        chapter["id"],
                        budget,
                        next_budget,
                    )
                    budget = next_budget
        raise RuntimeError(
            f"chapter {chapter['id']} did not satisfy length after "
            f"{self.max_length_attempts} attempts"
        )

    async def update_summary(
            self,
            prompt: str,
            previous_summary: Optional[str],
            chapter: Dict[str, Any],
            content: str,
            language: str) -> str:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[f"summary_{language}"],
            },
            {
                "role": "user",
                "content": summary_prompt(
                    prompt,
                    previous_summary,
                    chapter,
                    content,
                    language,
                    self.summary_word_limit,
                ),
            },
        ]
        for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
            response = await chat_completion(
                self.client,
                self.model,
                messages,
                self.summary_max_tokens,
                self.temperature,
                self.logger,
                retries=self.transport_retries,
                disable_thinking=self.disable_thinking,
            )
            if (
                response.content
                and response.finish_reason == "stop"
                and not response.token_limit_reached
            ):
                return response.content
            self.logger.warning(
                "Chapter %s summary attempt %d/%d incomplete: finish=%r, token_limit=%s, "
                "submitted_words=%s",
                chapter["id"],
                attempt,
                MAX_SUMMARY_ATTEMPTS,
                response.finish_reason,
                response.token_limit_reached,
                None,
            )
        raise RuntimeError(f"chapter {chapter['id']} summary retries exhausted")

    def checkpoint_path(self, story_id: Any) -> Optional[str]:
        if not self.checkpoint_dir:
            return None
        return os.path.join(self.checkpoint_dir, f"{story_id}.json")

    def load_checkpoint(
            self, story_id: Any, prompt: str) -> Optional[Dict[str, Any]]:
        path = self.checkpoint_path(story_id)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            expected = checkpoint_fingerprint(prompt, self.word_count)
            if payload.get("fingerprint") != expected:
                self.logger.warning("Ignoring checkpoint with mismatched fingerprint: %s", path)
                return None
            return payload
        except Exception as error:
            self.logger.warning("Failed to read checkpoint %s: %s", path, error)
            return None

    def save_checkpoint(self, story_id: Any, prompt: str,
                        payload: Dict[str, Any]) -> None:
        path = self.checkpoint_path(story_id)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = dict(payload)
        data["fingerprint"] = checkpoint_fingerprint(prompt, self.word_count)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
        os.replace(temporary, path)

    def remove_checkpoint(self, story_id: Any) -> None:
        path = self.checkpoint_path(story_id)
        if path and os.path.exists(path):
            os.remove(path)

    async def generate_story(self, item: Dict[str, Any]) -> Dict[str, Any]:
        story_id = item["id"]
        prompt = item["prompt"]
        language = item.get("language")
        if language not in ("zh", "en"):
            language = detect_language(prompt)
        checkpoint = self.load_checkpoint(story_id, prompt)
        if checkpoint:
            outline = checkpoint["outline"]
            chapters = checkpoint.get("chapters", [])
            summaries = checkpoint.get("summaries", [])
            outline_artifacts = checkpoint.get("outline_artifacts", {})
            outline_cache_metadata = checkpoint.get("outline_cache")
            self.logger.info(
                "RollingSummary id=%s resuming from checkpoint: %d/%d chapters",
                story_id, len(chapters), len(outline),
            )
        else:
            if self.outline_cache is None:
                # Standalone mode without a shared outline cache. The paired
                # experiments use --outline-cache-policy require and never
                # execute this branch.
                outline = await self.generate_outline(prompt, language)
                outline_artifacts = {}
                outline_cache_metadata = {
                    "policy": "off",
                    "status": "legacy_generated",
                }
            else:
                async def forbidden_generator() -> Dict[str, Any]:
                    raise RuntimeError(
                        "RollingSummary is read-only and cannot generate a "
                        "missing shared outline"
                    )

                def validate_cached_artifacts(value: Dict[str, Any]) -> None:
                    if not isinstance(value, dict):
                        raise ValueError(
                            "cached outline artifacts must be a JSON object"
                        )
                    validate_frozen_outline(
                        value.get("outline"), self.word_count
                    )

                artifacts, outline_cache_metadata = (
                    await self.outline_cache.get_or_create(
                        story_id=story_id,
                        prompt=prompt,
                        lang=language,
                        generator=forbidden_generator,
                        validator=validate_cached_artifacts,
                    )
                )
                outline = validate_frozen_outline(
                    artifacts["outline"], self.word_count
                )
                outline_artifacts = {
                    key: copy.deepcopy(value)
                    for key, value in artifacts.items()
                    if key != "outline"
                }
                self.logger.info(
                    "RollingSummary id=%s outline cache hit key=%s",
                    story_id,
                    (outline_cache_metadata.get("cache_key") or "-")[:16],
                )
            chapters = []
            summaries = []
            self.save_checkpoint(story_id, prompt, {
                "outline": outline,
                "outline_artifacts": outline_artifacts,
                "outline_cache": outline_cache_metadata,
                "chapters": chapters,
                "summaries": summaries,
            })

        global_minimum, global_maximum = word_bounds(self.word_count)
        # The aggregate whole-story range is diagnostic only.  Every chapter
        # remains subject to its own cached target ±20%, but independently
        # valid chapter deviations may accumulate beyond the nominal global
        # range and must not reopen an otherwise complete story.
        if len(chapters) == len(outline):
            summaries = summaries[:max(0, len(outline) - 1)]

        previous_summary = summaries[-1] if summaries else None
        for chapter in outline[len(chapters):]:
            is_final_chapter = len(chapters) + 1 == len(outline)
            content, stats = await self.generate_chapter(
                prompt,
                outline,
                chapter,
                previous_summary,
                language,
                accepted_word_bounds=None,
            )
            # A summary after the final chapter has no downstream consumer.
            # Skipping it avoids one expensive thinking call and prevents an
            # otherwise complete story from failing on unused summary text.
            current_summary = None
            if not is_final_chapter:
                current_summary = await self.update_summary(
                    prompt, previous_summary, chapter, content, language
                )
            chapters.append({
                "id": chapter["id"],
                "name": chapter["name"],
                "content": content,
                "word_count": count_words(content),
                "target_word_count": chapter["word_count"],
                "generation_stats": stats,
            })
            if current_summary is not None:
                summaries.append(current_summary)
                previous_summary = current_summary
            self.save_checkpoint(story_id, prompt, {
                "outline": outline,
                "outline_artifacts": outline_artifacts,
                "outline_cache": outline_cache_metadata,
                "chapters": chapters,
                "summaries": summaries,
            })

        total_words = sum(chapter["word_count"] for chapter in chapters)
        whole_story_length_ok = (
            global_minimum <= total_words <= global_maximum
        )
        if not whole_story_length_ok:
            self.logger.warning(
                "RollingSummary id=%s finished all chapters but whole length "
                "%d is outside diagnostic range [%d, %d]; the whole-story "
                "limit is not enforced",
                story_id,
                total_words,
                global_minimum,
                global_maximum,
            )
        record = {
            "id": story_id,
            "prompt": prompt,
            "language": language,
            "method": METHOD_NAME,
            "method_id": METHOD_ID,
            "outline_artifacts": outline_artifacts,
            "outline_cache": outline_cache_metadata,
            "outline": outline,
            "story": chapters,
            "summary": summaries,
            "target_word_count": self.word_count,
            "word_count": total_words,
            "whole_story_length_ok": whole_story_length_ok,
            "accepted_word_count_range": [global_minimum, global_maximum],
            "whole_story_length_enforced": False,
            "complete": True,
            "generation_stats": {
                "token_control": self.token_control,
                "length_control_mode": self.length_control_mode,
                "submission_mode": "plain",
                "tools_passed": False,
                "disable_thinking": self.disable_thinking,
                "chapter_target_ratio_zh": self.chapter_target_ratio_zh,
                "chapter_target_ratio_en": self.chapter_target_ratio_en,
                "max_tokens": self.max_tokens,
                "summary_max_tokens": self.summary_max_tokens,
                "summary_word_limit": (
                    self.summary_word_limit
                    if self.summary_word_limit > 0 else None
                ),
                "summary_length_policy": (
                    "prompt_max_words"
                    if self.summary_word_limit > 0 else "model_decides"
                ),
                "chapter_count": len(chapters),
                "chapter_attempts": sum(
                    chapter["generation_stats"]["attempt_count"]
                    for chapter in chapters
                ),
            },
            "timestamp": timestamp(),
        }
        self.remove_checkpoint(story_id)
        return record


async def main_async(args: argparse.Namespace) -> None:
    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(METHOD_ID, f"logs/{METHOD_ID}_{stamp}.log")
    with open(args.input, "r", encoding="utf-8") as file:
        prompts = [json.loads(line) for line in file if line.strip()]
    end = args.end if args.end is not None else len(prompts)
    prompts = prompts[args.start:end]

    completed = set() if args.no_resume else load_completed_ids(args.output)
    pending = [item for item in prompts if item["id"] not in completed]
    logger.info(
        "Loaded %d items, %d already completed, generating %d",
        len(prompts), len(completed), len(pending),
    )
    if not pending:
        print("Nothing to generate.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    failures_output = args.failures_output or args.output + ".failures.jsonl"
    checkpoint_dir = args.checkpoint_dir or args.output + ".checkpoints"
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        max_retries=args.client_max_retries,
        timeout=args.request_timeout,
    )
    writer = RollingSummaryWriter(
        client=client,
        model=args.model,
        max_tokens=args.max_tokens,
        outline_max_tokens=args.outline_max_tokens,
        summary_max_tokens=args.summary_max_tokens,
        summary_word_limit=args.summary_word_limit,
        temperature=args.temperature,
        word_count=args.word_count,
        logger=logger,
        token_control=args.token_control,
        length_control_mode=args.length_control_mode,
        max_length_attempts=args.max_length_attempts,
        checkpoint_dir=checkpoint_dir,
        dynamic_token_ratio_en=args.dynamic_token_ratio_en,
        dynamic_token_ratio_zh=args.dynamic_token_ratio_zh,
        dynamic_token_overhead=args.dynamic_token_overhead,
        dynamic_token_minimum=args.dynamic_token_minimum,
        transport_retries=args.client_max_retries + 1,
        chapter_target_ratio_zh=args.chapter_target_ratio_zh,
        chapter_target_ratio_en=args.chapter_target_ratio_en,
        disable_thinking=args.disable_thinking,
        outline_cache_dir=args.outline_cache_dir,
        outline_cache_policy=args.outline_cache_policy,
        outline_cache_model_id=args.outline_cache_model_id,
        outline_cache_lock_timeout=args.outline_cache_lock_timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrent)
    output_lock = asyncio.Lock()
    progress = atqdm(total=len(pending), desc=METHOD_NAME, unit="story")

    async def process_one(item: Dict[str, Any]) -> None:
        async with semaphore:
            error: Optional[Exception] = None
            for resume in range(args.auto_resumes + 1):
                try:
                    record = await writer.generate_story(item)
                    error = None
                    break
                except Exception as current_error:
                    error = current_error
                    logger.error(
                        "RollingSummary id=%s attempt %d/%d failed: %s",
                        item["id"],
                        resume + 1,
                        args.auto_resumes + 1,
                        current_error,
                    )
                    if resume < args.auto_resumes:
                        await asyncio.sleep(5 * (resume + 1))
            if error is None:
                destination = args.output
            else:
                record = {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "language": item.get("language", ""),
                    "method": METHOD_NAME,
                    "method_id": METHOD_ID,
                    "target_word_count": args.word_count,
                    "complete": False,
                    "error": str(error),
                    "timestamp": timestamp(),
                }
                destination = failures_output
            async with output_lock:
                os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
                with open(destination, "a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.update(1)

    await asyncio.gather(*(process_one(item) for item in pending))
    progress.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RollingSummary: outline + chapters + rolling summary"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures-output")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default=os.environ.get(
        "OPENAI_API_KEY", "EMPTY"
    ))
    parser.add_argument(
        "--client-max-retries",
        type=int,
        default=2,
        help="Extra OpenAI client retries after connection/API errors.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for a single OpenAI API request (default 600); raise it for very long outputs.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--outline-max-tokens",
        type=int,
        default=DEFAULT_OUTLINE_MAX_TOKENS,
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=DEFAULT_SUMMARY_MAX_TOKENS,
    )
    parser.add_argument(
        "--summary-word-limit",
        type=int,
        default=0,
        help=(
            "Optional maximum words for each complete English rolling "
            "summary. By default (0) there is no limit and the model chooses "
            "the necessary summary length; the API token ceiling still applies."
        ),
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument("--word-count", type=int, default=DEFAULT_WORD_COUNT)
    parser.add_argument(
        "--submission-mode",
        choices=SUBMISSION_MODES,
        default="plain",
        help=(
            "Fixed value; RollingSummary always reads the plain assistant "
            "text and never passes tools to the model."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Diagnostic option: disable thinking via Qwen/vLLM "
            "chat_template_kwargs; comparisons should use the same reasoning "
            "mode as the target method"
        ),
    )
    parser.add_argument(
        "--outline-cache-dir",
        help="Directory containing immutable shared planning artifacts.",
    )
    parser.add_argument(
        "--outline-cache-policy",
        choices=("off", "require"),
        default="off",
        help=(
            "require reads an existing shared outline and fails if it is "
            "missing; RollingSummary never writes to the shared cache"
        ),
    )
    parser.add_argument(
        "--outline-cache-model-id",
        help="Canonical model id used in the shared cache key.",
    )
    parser.add_argument(
        "--outline-cache-lock-timeout",
        type=float,
        default=7200.0,
    )
    parser.add_argument(
        "--token-control",
        choices=TOKEN_CONTROL_MODES,
        default="fixed",
    )
    parser.add_argument(
        "--length-control-mode",
        choices=LENGTH_CONTROL_MODES,
        default="adaptive_recent_failure",
    )
    parser.add_argument(
        "--max-length-attempts",
        type=int,
        default=DEFAULT_MAX_LENGTH_ATTEMPTS,
    )
    parser.add_argument(
        "--auto-resumes",
        type=int,
        default=DEFAULT_AUTO_RESUMES,
    )
    parser.add_argument(
        "--dynamic-token-ratio-en",
        type=float,
        default=DEFAULT_DYNAMIC_TOKEN_RATIO_EN,
    )
    parser.add_argument(
        "--dynamic-token-ratio-zh",
        type=float,
        default=DEFAULT_DYNAMIC_TOKEN_RATIO_ZH,
    )
    parser.add_argument(
        "--dynamic-token-overhead",
        type=int,
        default=DEFAULT_ROLLING_DYNAMIC_TOKEN_OVERHEAD,
    )
    parser.add_argument(
        "--dynamic-token-minimum",
        type=int,
        default=DEFAULT_DYNAMIC_TOKEN_MINIMUM,
    )
    parser.add_argument(
        "--chapter-target-ratio-zh",
        type=float,
        default=1.0,
        help=(
            "Chinese-only internal writing-target calibration. The cached "
            "target and strict acceptance bounds are never changed."
        ),
    )
    parser.add_argument(
        "--chapter-target-ratio-en",
        type=float,
        default=1.0,
        help=(
            "English-only internal writing-target calibration. The cached "
            "target and strict acceptance bounds are never changed."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    for name in (
        "max_tokens",
        "outline_max_tokens",
        "summary_max_tokens",
        "word_count",
        "max_length_attempts",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.client_max_retries < 0:
        parser.error("--client-max-retries must be >= 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.summary_word_limit < 0:
        parser.error("--summary-word-limit must be >= 0")
    if args.outline_cache_policy == "require" and not args.outline_cache_dir:
        parser.error(
            "--outline-cache-dir is required when cache policy is require"
        )
    if args.outline_cache_lock_timeout <= 0:
        parser.error("--outline-cache-lock-timeout must be > 0")
    if args.auto_resumes < 0:
        parser.error("--auto-resumes must be >= 0")
    if not 0 < args.chapter_target_ratio_zh <= 1.0:
        parser.error("--chapter-target-ratio-zh must be in (0, 1]")
    if not 0 < args.chapter_target_ratio_en <= 1.0:
        parser.error("--chapter-target-ratio-en must be in (0, 1]")
    print(
        f"{METHOD_NAME}: model={args.model}, target={args.word_count}, "
        f"token_control={args.token_control}, "
        f"length_control={args.length_control_mode}, "
        "submission_mode=plain, tools_passed=false, "
        f"disable_thinking={args.disable_thinking}, "
        f"concurrent={args.concurrent}"
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
