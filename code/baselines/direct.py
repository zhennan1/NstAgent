#!/usr/bin/env python
# coding: utf-8
"""Direct: generate the whole story in a single call, with strict length
control over the full text.

Each attempt is a direct generation, with no chapter-by-chapter continuation
or external memory. When the length check fails, the model must regenerate the
entire story, so this remains a direct-output baseline rather than a chaptered
method.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    TOKEN_CONTROL_MODES,
    chat_completion,
    detect_language,
    dynamic_max_tokens,
    extract_tool_text,
    load_completed_ids,
    retry_messages,
    setup_logger,
    timestamp,
    tool_retry_messages,
    update_failure_state,
    validate_generation,
    validation_dict,
    word_bounds,
)


METHOD_ID = "direct"
METHOD_NAME = "Direct"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENT = 8
DEFAULT_WORD_COUNT = 10000
DEFAULT_MAX_LENGTH_ATTEMPTS = 10
SUBMISSION_MODES = ("plain", "tool")

SYSTEM_PROMPTS = {
    "zh": (
        "你是一位经验丰富的小说家。请根据要求直接写出完整故事，重视人物发展、"
        "世界设定一致性、情节逻辑与叙事完成度。"
    ),
    "en": (
        "You are an experienced novelist. Write the complete story directly, "
        "with strong character development, consistent world-building, "
        "logical plot progression, and a finished narrative arc."
    ),
}


def direct_prompt(
        prompt: str,
        target_words: int,
        language: str,
        submission_mode: str = "plain",
) -> str:
    """Place the exact target and the shared ±20% range closest to the output."""
    minimum, maximum = word_bounds(target_words)
    if language == "zh":
        output_instruction = (
            "直接输出完整故事正文。不要输出大纲、字数说明、写作计划、"
            "工具调用或其他元文本。"
            if submission_mode == "plain" else
            "直接写出完整故事，并使用 write_story 提交正文。不要输出大纲、"
            "字数说明、写作计划或其他元文本，也不要在工具调用外输出正文。"
        )
        return (
            f"故事要求：\n{prompt}\n\n"
            "# 输出要求\n"
            f"{output_instruction}\n"
            f"完整正文目标为 {target_words} 字，允许范围为 "
            f"{minimum}–{maximum} 字。请为高潮后的结局预留约最后 10% "
            "篇幅，在允许范围内完整解决核心冲突，并在最后一句后立即停止，"
            "不要持续写到 token 上限。"
        )
    output_instruction = (
        "Output only the complete story prose directly. Do not output an "
        "outline, word-count commentary, writing plan, tool call, or other "
        "meta-text."
        if submission_mode == "plain" else
        "Write the complete story directly and submit it through write_story. "
        "Do not output an outline, word-count commentary, writing plan, other "
        "meta-text, or prose outside the tool call."
    )
    return (
        f"Story prompt:\n{prompt}\n\n"
        "# Output requirements\n"
        f"{output_instruction}\n"
        f"The complete story targets {target_words} words; the accepted range "
        f"is {minimum}-{maximum} words. Reserve roughly the final 10% for the "
        "resolution after the climax, fully resolve the core conflict inside "
        "the accepted range, and stop immediately after the final sentence; "
        "do not continue writing until the token limit."
    )


def direct_write_tool(target_words: int, language: str) -> List[Dict[str, Any]]:
    """Expose a single tool for submitting the full story; no memory or
    retrieval capability is provided."""
    minimum, maximum = word_bounds(target_words)
    if language == "zh":
        description = (
            f"提交一次直接生成的完整故事正文。精确目标为 {target_words} 字，"
            f"允许范围为 {minimum}–{maximum} 字；字数不合格时返回 ok=false，"
            "应重新提交完整故事。"
        )
    else:
        description = (
            "Submit the complete directly generated story. The exact target "
            f"is {target_words} words and the accepted range is "
            f"{minimum}-{maximum}; if validation returns ok=false, resubmit "
            "the complete story."
        )
    return [{
        "type": "function",
        "function": {
            "name": "write_story",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete story prose.",
                    },
                },
                "required": ["content"],
            },
        },
    }]


class DirectWriter:
    """Generate the full story directly and, on failure, build the next
    rewrite context according to the selected strategy."""

    def __init__(
            self,
            client: AsyncOpenAI,
            model: str,
            max_tokens: int,
            temperature: float,
            word_count: int,
            logger: logging.Logger,
            token_control: str,
            length_control_mode: str,
            max_length_attempts: int,
            dynamic_token_ratio_en: float,
            dynamic_token_ratio_zh: float,
            dynamic_token_overhead: int,
            dynamic_token_minimum: int,
            submission_mode: str = "plain",
            disable_thinking: bool = False,
            transport_retries: int = 3):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.word_count = word_count
        self.logger = logger
        self.token_control = token_control
        self.length_control_mode = length_control_mode
        self.max_length_attempts = max_length_attempts
        self.dynamic_token_ratio_en = dynamic_token_ratio_en
        self.dynamic_token_ratio_zh = dynamic_token_ratio_zh
        self.dynamic_token_overhead = dynamic_token_overhead
        self.dynamic_token_minimum = dynamic_token_minimum
        self.submission_mode = submission_mode
        self.disable_thinking = disable_thinking
        self.transport_retries = transport_retries

    def output_budget(self, language: str) -> int:
        if self.token_control == "fixed":
            return self.max_tokens
        return dynamic_max_tokens(
            self.word_count,
            language,
            self.max_tokens,
            ratio_en=self.dynamic_token_ratio_en,
            ratio_zh=self.dynamic_token_ratio_zh,
            overhead=self.dynamic_token_overhead,
            minimum=self.dynamic_token_minimum,
        )

    async def generate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        language = item.get("language")
        if language not in ("zh", "en"):
            language = detect_language(item["prompt"])
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPTS.get(
                language, SYSTEM_PROMPTS["en"]
            )},
            {"role": "user", "content": direct_prompt(
                item["prompt"],
                self.word_count,
                language,
                self.submission_mode,
            )},
        ]
        messages = base_messages
        failure = None
        attempts: List[Dict[str, Any]] = []
        budget = self.output_budget(language)
        tools = (
            direct_write_tool(self.word_count, language)
            if self.submission_mode == "tool" else None
        )

        for attempt in range(1, self.max_length_attempts + 1):
            response = await chat_completion(
                self.client,
                self.model,
                messages,
                budget,
                self.temperature,
                self.logger,
                retries=self.transport_retries,
                tools=tools,
                disable_thinking=self.disable_thinking,
            )
            submitted = (
                extract_tool_text(response, "write_story")
                if self.submission_mode == "tool" else None
            )
            story = (
                submitted if submitted is not None else response.content
            )
            accepted_reasons = (
                ("tool_calls",)
                if self.submission_mode == "tool" and submitted is not None
                else ("stop",)
            )
            validation = validate_generation(
                story,
                response,
                self.word_count,
                accepted_finish_reasons=accepted_reasons,
            )
            attempts.append({
                "attempt": attempt,
                "validation": validation_dict(validation),
                "requested_max_tokens": response.requested_max_tokens,
                "completion_tokens": response.completion_tokens,
                "tool_call_received": submitted is not None,
                "submission_mode": self.submission_mode,
            })
            self.logger.info(
                "Direct id=%s attempt=%d: %d words, finish=%r, accepted=%s",
                item["id"],
                attempt,
                validation.actual_words,
                validation.finish_reason,
                validation.ok,
            )
            if validation.ok:
                return {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "language": language,
                    "method": METHOD_NAME,
                    "method_id": METHOD_ID,
                    "story": story,
                    "target_word_count": self.word_count,
                    "word_count": validation.actual_words,
                    "complete": True,
                    "generation_stats": {
                        "token_control": self.token_control,
                        "length_control_mode": self.length_control_mode,
                        "submission_mode": self.submission_mode,
                        "disable_thinking": self.disable_thinking,
                        "max_tokens": self.max_tokens,
                        "per_attempt_max_tokens": budget,
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                    },
                    "timestamp": timestamp(),
                }
            failure = update_failure_state(
                self.length_control_mode,
                failure,
                story,
                validation,
            )
            messages = (
                tool_retry_messages(
                    base_messages,
                    failure,
                    language,
                    "write_story",
                )
                if self.submission_mode == "tool" else
                retry_messages(base_messages, failure, language)
            )

        last = attempts[-1]["validation"] if attempts else None
        raise RuntimeError(
            f"Direct id={item['id']} did not satisfy length after "
            f"{self.max_length_attempts} attempts; last={last}"
        )


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
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        max_retries=args.client_max_retries,
    )
    writer = DirectWriter(
        client=client,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        word_count=args.word_count,
        logger=logger,
        token_control=args.token_control,
        length_control_mode=args.length_control_mode,
        max_length_attempts=args.max_length_attempts,
        dynamic_token_ratio_en=args.dynamic_token_ratio_en,
        dynamic_token_ratio_zh=args.dynamic_token_ratio_zh,
        dynamic_token_overhead=args.dynamic_token_overhead,
        dynamic_token_minimum=args.dynamic_token_minimum,
        submission_mode=args.submission_mode,
        disable_thinking=args.disable_thinking,
        transport_retries=args.client_max_retries + 1,
    )
    semaphore = asyncio.Semaphore(args.concurrent)
    output_lock = asyncio.Lock()
    progress = atqdm(total=len(pending), desc=METHOD_NAME, unit="story")

    async def process_one(item: Dict[str, Any]) -> None:
        async with semaphore:
            try:
                record = await writer.generate(item)
                destination = args.output
            except Exception as error:
                logger.error("Direct id=%s failed: %s", item["id"], error)
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
        description="Direct: one-call complete-story generation"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures-output")
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
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument("--word-count", type=int, default=DEFAULT_WORD_COUNT)
    parser.add_argument(
        "--submission-mode",
        choices=SUBMISSION_MODES,
        default="plain",
        help=(
            "plain returns the story text directly (the Direct protocol); "
            "tool wraps the text in a write_story tool call"
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
        default=DEFAULT_DYNAMIC_TOKEN_OVERHEAD,
    )
    parser.add_argument(
        "--dynamic-token-minimum",
        type=int,
        default=DEFAULT_DYNAMIC_TOKEN_MINIMUM,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.word_count <= 0:
        parser.error("--word-count must be > 0")
    if args.max_length_attempts <= 0:
        parser.error("--max-length-attempts must be > 0")
    if args.client_max_retries < 0:
        parser.error("--client-max-retries must be >= 0")
    print(
        f"{METHOD_NAME}: model={args.model}, target={args.word_count}, "
        f"token_control={args.token_control}, "
        f"length_control={args.length_control_mode}, "
        f"submission_mode={args.submission_mode}, "
        f"disable_thinking={args.disable_thinking}, "
        f"concurrent={args.concurrent}"
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
