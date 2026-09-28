#!/usr/bin/env python
# coding: utf-8
"""Generation and length-control utilities shared by Direct and RollingSummary.

Only method-agnostic logic lives here: mixed Chinese/English word counting,
token budgets, API response checks, failed-draft retention, and JSONL
helpers. Both baselines therefore use the same length criterion while
keeping their own generation pipelines.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


WORD_COUNT_TOLERANCE = 0.20
WORD_COUNT_TOLERANCE_PERCENT = 20
LONG_FORM_MIN_WORDS = 10000
LENGTH_CONTROL_MODES = (
    "feedback_only",
    "recent_failure_context",
    "adaptive_recent_failure",
)
TOKEN_CONTROL_MODES = ("fixed", "dynamic")
ADAPTIVE_MIN_IMPROVEMENT = 0.10

DEFAULT_DYNAMIC_TOKEN_RATIO_EN = 1.35
DEFAULT_DYNAMIC_TOKEN_RATIO_ZH = 0.85
DEFAULT_DYNAMIC_TOKEN_OVERHEAD = 384
DEFAULT_DYNAMIC_TOKEN_MINIMUM = 1024

_CJK_CHAR = re.compile(
    r"[㐀-䶿一-鿿豈-﫿]"
    r"|[\U00020000-\U0002ffff]"
)
_ASCII_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def count_words(text: str) -> int:
    """Count mixed-language words: each CJK character and each ASCII word counts 1."""
    if not text:
        return 0
    return len(_CJK_CHAR.findall(text)) + len(_ASCII_WORD.findall(text))


def detect_language(text: str) -> str:
    """Classify text as Chinese when CJK makes up at least 30% of letter characters."""
    if not text:
        return "en"
    cjk = len(_CJK_CHAR.findall(text))
    letters = sum(1 for char in text if char.isalpha() or _CJK_CHAR.match(char))
    if letters == 0:
        return "en"
    return "zh" if cjk / letters >= 0.30 else "en"


def word_bounds(target_words: int) -> Tuple[int, int]:
    """Return the closed interval of ±20% around the target word count."""
    return (
        math.ceil(target_words * (1 - WORD_COUNT_TOLERANCE)),
        math.floor(target_words * (1 + WORD_COUNT_TOLERANCE)),
    )


def dynamic_max_tokens(
        target_words: int,
        language: str,
        ceiling: int,
        ratio_en: float = DEFAULT_DYNAMIC_TOKEN_RATIO_EN,
        ratio_zh: float = DEFAULT_DYNAMIC_TOKEN_RATIO_ZH,
        overhead: int = DEFAULT_DYNAMIC_TOKEN_OVERHEAD,
        minimum: int = DEFAULT_DYNAMIC_TOKEN_MINIMUM,
) -> int:
    """Per-call new-token cap: target * ratio + overhead, within [minimum, ceiling]."""
    ratio = ratio_zh if language == "zh" else ratio_en
    estimated = math.ceil(target_words * ratio + overhead)
    return min(ceiling, max(minimum, estimated))


def extract_content(message: Any) -> str:
    """Return the output from content, or from a reasoning field if content is empty."""
    content = getattr(message, "content", None)
    if content:
        return content.strip()
    data = message.model_dump() if hasattr(message, "model_dump") else {}
    for field in ("reasoning_content", "reasoning"):
        reasoning = data.get(field, "") or ""
        if not reasoning:
            continue
        if "</think>" in reasoning:
            return reasoning.split("</think>", 1)[1].strip()
        return reasoning.strip()
    raise ValueError("empty response: no content or reasoning")


@dataclass
class ChatResult:
    """Content and completion metadata for one Chat Completions call."""

    content: str
    finish_reason: Optional[str]
    requested_max_tokens: int
    completion_tokens: Optional[int]
    token_limit_reached: bool
    tool_calls: Optional[List[Dict[str, Any]]] = None


async def chat_completion(
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        logger: logging.Logger,
        retries: int = 3,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        disable_thinking: bool = False,
) -> ChatResult:
    """Retry only network/API errors; the caller treats length/truncation as a failed attempt."""
    for retry in range(retries):
        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            if disable_thinking:
                # Official request-level Qwen3/Qwen3.5 vLLM switch; diagnostic only
                # (effect of thinking on the prose token budget). Off in the main comparisons.
                kwargs["extra_body"] = {
                    "top_k": 20,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                }
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            finish_reason = getattr(choice, "finish_reason", None)
            completion_tokens = (
                getattr(response.usage, "completion_tokens", None)
                if getattr(response, "usage", None) is not None else None
            )
            tool_calls = []
            for call in getattr(message, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                tool_calls.append({
                    "id": getattr(call, "id", "") or "",
                    "name": getattr(function, "name", "") or "",
                    "arguments": getattr(function, "arguments", "") or "",
                })
            try:
                content = extract_content(message)
            except ValueError:
                content = ""
            return ChatResult(
                content=content,
                finish_reason=finish_reason,
                requested_max_tokens=max_tokens,
                completion_tokens=completion_tokens,
                token_limit_reached=(
                    # Hosted reasoning APIs may include hidden reasoning in
                    # completion_tokens even when max_tokens applies only to
                    # the visible answer, so a valid `stop` response can
                    # report usage greater than the requested visible-output
                    # budget. The provider
                    # finish reason is authoritative for a completed answer;
                    # usage is only a truncation fallback when no terminal
                    # finish reason was supplied.
                    finish_reason == "length"
                    or (
                        finish_reason not in ("stop", "tool_calls")
                        and isinstance(completion_tokens, int)
                        and completion_tokens >= max_tokens
                    )
                ),
                tool_calls=tool_calls,
            )
        except Exception as error:
            if retry == retries - 1:
                raise
            logger.warning(
                "API call failed, transport retry %d/%d: %s",
                retry + 1,
                retries,
                error,
            )
            await asyncio.sleep(5 * (retry + 1))
    raise RuntimeError("API transport retries exhausted")


@dataclass
class LengthValidation:
    """Strict length and completion check result for one draft."""

    ok: bool
    actual_words: int
    target_words: int
    min_words: int
    max_words: int
    finish_reason: Optional[str]
    token_limit_reached: bool
    reason: str

    @property
    def distance(self) -> int:
        if self.actual_words < self.min_words:
            return self.min_words - self.actual_words
        if self.actual_words > self.max_words:
            return self.actual_words - self.max_words
        return 0


def validate_generation(
        text: str,
        response: ChatResult,
        target_words: int,
        accepted_finish_reasons: Tuple[str, ...] = ("stop",),
) -> LengthValidation:
    """A draft passes only if it stopped normally and its length is within ±20%."""
    minimum, maximum = word_bounds(target_words)
    actual = count_words(text)
    if response.finish_reason not in accepted_finish_reasons:
        reason = (
            f"finish_reason={response.finish_reason!r}, expected one of "
            f"{accepted_finish_reasons}"
        )
    elif response.token_limit_reached:
        reason = "completion token budget was exhausted"
    elif actual < minimum:
        reason = f"too short: {actual} < {minimum}"
    elif actual > maximum:
        reason = f"too long: {actual} > {maximum}"
    else:
        reason = "accepted"
    return LengthValidation(
        ok=reason == "accepted",
        actual_words=actual,
        target_words=target_words,
        min_words=minimum,
        max_words=maximum,
        finish_reason=response.finish_reason,
        token_limit_reached=response.token_limit_reached,
        reason=reason,
    )


@dataclass
class FailureState:
    """Latest failure info for the next rewrite; the draft text may be dropped."""

    content: Optional[str]
    validation: LengthValidation
    content_sha256: str
    best_distance: int
    retained: bool


def update_failure_state(
        mode: str,
        previous: Optional[FailureState],
        content: str,
        validation: LengthValidation,
) -> FailureState:
    """Keep the latest failed draft, or only meaningfully improved ones (adaptive)."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if mode == "recent_failure_context":
        retain = True
    elif mode == "adaptive_recent_failure":
        # A draft already within the word range but with an incomplete
        # finish_reason or tool wrapper is the best reference for the next
        # attempt, so always keep it (best_distance may already be 0).
        if validation.distance == 0:
            retain = True
        elif previous is None:
            retain = True
        else:
            improved = (
                validation.distance
                <= previous.best_distance * (1 - ADAPTIVE_MIN_IMPROVEMENT)
                if previous.best_distance > 0 else False
            )
            retain = improved and digest != previous.content_sha256
            if not retain:
                # Adaptive mode means "keep the best usable failed draft",
                # not merely "drop the latest bad body".  Returning a new
                # state with the worse validation would make subsequent
                # feedback describe that worse attempt while also removing
                # the near-valid draft from context, causing correction
                # trajectories to drift away from the target.
                return previous
    elif mode == "feedback_only":
        retain = False
    else:
        raise ValueError(f"unsupported length-control mode: {mode}")

    previous_best = previous.best_distance if previous is not None else None
    best_distance = validation.distance
    if previous_best is not None:
        best_distance = min(best_distance, previous_best)
    retain = retain and bool(content)
    return FailureState(
        content=content if retain else None,
        validation=validation,
        content_sha256=digest,
        best_distance=best_distance,
        retained=retain,
    )


def failure_feedback(
        state: FailureState,
        language: str,
        expects_tool: bool = False,
) -> str:
    """Explain why the last draft failed and how the next one should end."""
    item = state.validation
    in_range_but_truncated = (
        item.distance == 0
        and (item.finish_reason == "length" or item.token_limit_reached)
    )
    # If the draft is within range but hit the token limit, repeating the
    # original target makes the model chase the same length. Move the retry
    # target to the low end of the range with explicit ending points; the
    # acceptance range and experiment target are unchanged.
    safe_target = math.floor(
        item.min_words + (item.max_words - item.min_words) * 0.05
    )
    ending_start = math.floor(safe_target * 0.90)
    hard_deadline = min(
        item.max_words,
        math.ceil(safe_target * 1.04),
    )
    if language == "zh":
        text = (
            f"上一稿不合格：实际为 {item.actual_words} 字，目标为 "
            f"{item.target_words} 字，允许范围为 "
            f"{item.min_words}–{item.max_words} 字；"
            f"响应结束原因为 {item.finish_reason!r}。"
        )
        if in_range_but_truncated:
            if expects_tool:
                return text + (
                    "上一稿字数已经合格，但输出在工具调用结束前被截断。"
                    "请保持大致相同的正文长度，只收束被截断的结尾并重新完整"
                    "调用写入工具，不要继续扩写。"
                )
            return text + (
                "上一稿字数已经合格，但正文在自然结局前被输出上限截断。"
                f"本次不要再追求 {item.target_words} 字，改以约 "
                f"{safe_target} 字为写作目标；这仍在允许范围内。保留核心"
                f"情节，最迟在约 {ending_start} 字进入最终收尾，并在 "
                f"{hard_deadline} 字前完整结束。不要续写或只补结尾，也"
                "不要调用工具；只输出完整修订稿。"
            )
        if item.actual_words < item.min_words:
            gap = item.min_words - item.actual_words
            ratio = item.target_words / max(item.actual_words, 1)
            text += (
                f"上一稿至少还少 {gap} 字；本次应扩展到约为上一稿的 "
                f"{ratio:.2f} 倍。保留已有情节，不要压缩或概括；通过增加"
                "完整场景、对话、行动过程和人物反应来补足篇幅，并确保"
                "结局仍然完整。"
            )
        elif item.actual_words > item.max_words:
            excess = item.actual_words - item.max_words
            keep_percent = max(
                1,
                round(100 * item.target_words / max(item.actual_words, 1)),
            )
            target_ending_start = math.floor(item.target_words * 0.90)
            target_hard_deadline = min(
                item.max_words,
                math.ceil(item.target_words * 1.04),
            )
            text += (
                f"上一稿至少超出 {excess} 字。请将完整重写稿压缩到上一稿"
                f"长度的约 {keep_percent}%，以约 {item.target_words} 字为目标；"
                f"最迟在约 {target_ending_start} 字开始最终收尾，并在 "
                f"{target_hard_deadline} 字前完整结束。保留完整因果链和结局，"
                "删除重复描写、重复解释和不改变剧情的场景，"
                "不要只截断结尾，也不要在改写时继续扩写。"
            )
        return text + (
            "上一稿正文仍完整保留在对话中。请在其基础上重写完整正文，"
            "不要只输出增补片段。"
            if state.retained else
            "上一稿正文未保留在本轮工作上下文中。请重新输出完整正文，"
            "不要只输出增补片段。"
        )
    text = (
        f"The previous complete draft failed validation: it has "
        f"{item.actual_words} words; the target is {item.target_words}, "
        f"the accepted range is {item.min_words}-{item.max_words}, and "
        f"finish_reason was {item.finish_reason!r}. "
    )
    if in_range_but_truncated:
        if expects_tool:
            return text + (
                "Its length is already valid, but the response was cut off "
                "before the tool call finished. Keep approximately the same "
                "body length, only close the truncated ending, and call the "
                "write tool again completely; do not expand it."
            )
        return text + (
            "Its length is already valid, but the prose hit the output limit "
            "before reaching a natural ending. On this retry, do not aim for "
            f"{item.target_words} words; aim for about {safe_target} words, "
            "which remains inside the accepted range. Preserve the core plot, "
            f"begin the final resolution by about {ending_start} words, and "
            f"finish the complete narrative before {hard_deadline} words. Do "
            "not continue or append to the draft, and do not call a tool; "
            "output only the complete revised story."
        )
    if item.actual_words < item.min_words:
        gap = item.min_words - item.actual_words
        ratio = item.target_words / max(item.actual_words, 1)
        text += (
            f"It is still at least {gap} words below the minimum. Expand the "
            f"next complete draft to about {ratio:.2f} times this draft's "
            "length. Preserve the existing plot instead of compressing or "
            "summarizing it; add fully developed scenes, dialogue, action, "
            "and character reactions while retaining a complete ending. "
        )
    elif item.actual_words > item.max_words:
        excess = item.actual_words - item.max_words
        keep_percent = max(
            1, round(100 * item.target_words / max(item.actual_words, 1))
        )
        text += (
            f"It exceeds the maximum by at least {excess} words. Preserve the "
            f"causal chain and ending. Rewrite the complete chapter at about "
            f"{keep_percent}% of the "
            "previous length, aiming for the stated target rather than the "
            "upper bound. Preserve the causal chain and full ending, but "
            "remove repeated description, repeated explanation, and scenes "
            "that do not change the plot; do not merely truncate the ending. "
        )
    return text + (
        "The full failed draft remains in the conversation. Rewrite the "
        "complete text from it; do not return only an added fragment."
        if state.retained else
        "The failed draft is not retained in this working context. Rewrite "
        "the complete text; do not return only an added fragment."
    )


def retry_messages(
        base_messages: Sequence[Dict[str, str]],
        failure: FailureState,
        language: str,
) -> List[Dict[str, str]]:
    """Rebuild a compact retry context with at most one failed draft."""
    messages = [dict(item) for item in base_messages]
    if failure.retained and failure.content:
        messages.append({"role": "assistant", "content": failure.content})
    messages.append({
        "role": "user",
        "content": failure_feedback(failure, language, expects_tool=False),
    })
    return messages


def extract_tool_text(response: ChatResult, tool_name: str,
                      argument_name: str = "content") -> Optional[str]:
    """Read a tool call's prose argument; None if missing or the JSON is incomplete."""
    for call in response.tool_calls or []:
        if call.get("name") != tool_name:
            continue
        try:
            arguments = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None
        content = arguments.get(argument_name)
        if isinstance(content, str) and content:
            return content
        return None
    return None


def tool_retry_messages(
        base_messages: Sequence[Dict[str, Any]],
        failure: FailureState,
        language: str,
        tool_name: str,
        argument_name: str = "content",
) -> List[Dict[str, Any]]:
    """Rebuild the failure context from the real tool trace, with at most one full draft."""
    messages = [dict(item) for item in base_messages]
    if failure.retained and failure.content:
        call_id = "retained_rejected_write"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(
                        {argument_name: failure.content},
                        ensure_ascii=False,
                    ),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({
                "ok": False,
                "word_count": failure.validation.actual_words,
                "target_word_count": failure.validation.target_words,
                "min_word_count": failure.validation.min_words,
                "max_word_count": failure.validation.max_words,
                "finish_reason": failure.validation.finish_reason,
            }, ensure_ascii=False),
        })
    messages.append({
        "role": "user",
        "content": failure_feedback(failure, language, expects_tool=True),
    })
    return messages


def extract_json(text: str) -> Any:
    """Extract the first complete JSON object from JSON, a code block, or mixed text."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for pattern in (
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
    ):
        match = re.search(pattern, stripped, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    for opening, closing in (("[", "]"), ("{", "}")):
        start = stripped.find(opening)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start:index + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not extract JSON: {stripped[:300]}...")


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Create a logger that writes to both a file and the terminal."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_completed_ids(output: str) -> set:
    """Treat an id as done only if its latest record has complete=true."""
    completed = set()
    if not os.path.exists(output):
        return completed
    with open(output, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("complete") is True and "id" in row:
                completed.add(row["id"])
    return completed


def validation_dict(validation: LengthValidation) -> Dict[str, Any]:
    """Convert a validation record for JSONL output."""
    return asdict(validation)


def timestamp() -> str:
    return datetime.now().isoformat()
