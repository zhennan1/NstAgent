#!/usr/bin/env python
# coding: utf-8
"""NstAgent component ablations (English prompts only).

Runs the unchanged NstAgent core (``agent_core.py``) with exactly one
component removed:

    no_state                -State: no update tool and no narrative state in the
                            prompt; read/search/correct and the write gate stay
    no_lookback             -Lookback: no read/search/correct tools; state stays
    none                    control; reproduces NstAgent surfaces byte for byte

Every change is an exact-text edit of an NstAgent surface the model sees: system
prompt, chapter prompt, tool schemas, controller nudges and tool results.
Chapter-prompt and schema edits must match exactly once, so a core change
cannot silently turn an ablation into a no-op. Removed tools that the model
still calls get the core's own ``unknown tool`` response and are counted.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import agent_core as core
import nstagent

METHOD_NAME = "NstAgentAblation"
PROTOCOL_VERSION = "narrative_ops_prompt_v1_ablation_v1"
ABLATIONS = ("none", "no_state", "no_lookback")
LOOKBACK_TOOLS = ("read", "search", "correct")
STATE_OMITTED = "<<NSTAGENT_ABLATION_STATE_OMITTED>>"

Edits = Sequence[Tuple[str, str]]

# --- NstAgent English surfaces (exact text rendered by the frozen core) -----

STEP_READ = (
    "1. Use read to revisit a specific chapter, or search to locate prior "
    "facts, as needed.\n"
)
STEP_WRITE = "2. Use write to write the complete current chapter.\n"
STEP_CORRECT = (
    "3. If necessary, use correct to precisely fix consistency errors in "
    "the current or a prior chapter.\n"
)
STEP_UPDATE = (
    "4. Once the prose is final, make exactly one update call containing "
    "character-state upserts, newly established completed events, new future "
    "requirements, and resolved future-requirement keys.\n"
)
STEP_DONE = (
    "5. When everything is complete, make no tool call and output only "
    "DONE; any other text does not finish the chapter.\n\n"
)
UPDATE_RULES = (
    "- In upsert_character_state, provide {name, description} with each "
    "character's complete current location, goal, relationships, knowledge, "
    "possessions, and physical or emotional condition; an existing name is "
    "replaced rather than appended. In add_past_event, record a completed "
    "event only when the fact is not already explicit in the frozen outline "
    "and may affect later plot; give it a stable snake_case key. Add only "
    "concrete later obligations to add_future_requirement, also with stable "
    "keys, and pass the keys of requirements fulfilled in this chapter to "
    "resolve_future_requirement. Every field must be a native JSON array; "
    "never serialize or quote an array as a string. Include all four arrays "
    "and use [] when one has no changes.\n"
)

SYSTEM_EDITS: Dict[str, Edits] = {
    "none": (),
    "no_state": ((
        "write the current chapter, fix inconsistencies, and maintain the "
        "structured narrative state.",
        "write the current chapter, and fix inconsistencies.",
    ),),
    "no_lookback": ((
        "Use the provided tools to read written chapters, search for "
        "specified content, write the current chapter, fix inconsistencies, "
        "and maintain",
        "Use the provided tools to write the current chapter and maintain",
    ),),
}

PROMPT_EDITS: Dict[str, Edits] = {
    "none": (),
    "no_state": (
        (f"# Current Narrative State\n\"{STATE_OMITTED}\"\n\n", ""),
        (STEP_UPDATE, ""),
        (STEP_DONE, "4." + STEP_DONE[2:]),
        ("Writing and state requirements:\n", "Writing requirements:\n"),
        (UPDATE_RULES, ""),
    ),
    "no_lookback": (
        (STEP_READ, ""),
        (STEP_WRITE, "1." + STEP_WRITE[2:]),
        (STEP_CORRECT, ""),
        (STEP_UPDATE, "2." + STEP_UPDATE[2:]),
        (STEP_DONE, "3." + STEP_DONE[2:]),
    ),
}

# Controller nudges (user turns) and tool-result texts; edited when present.
NUDGE_EDITS: Dict[str, Edits] = {
    "none": (),
    "no_state": (
        (", then make exactly one update call containing all four "
         "narrative-operation arrays, and finally output DONE.",
         ", then output DONE."),
        (" If state is not updated yet, make exactly one update call "
         "containing the character upserts, past events, new future "
         "requirements, and resolved requirement keys; otherwise output only "
         "DONE now.",
         " Output only DONE now."),
    ),
    "no_lookback": (),
}

RESULT_EDITS: Dict[str, Edits] = {
    "none": (),
    "no_state": (
        ("proceed to the state update + DONE", "proceed to DONE"),
        ("then update the state + DONE", "then output DONE"),
    ),
    "no_lookback": ((
        "further whole-chapter writes are not allowed. Use correct for any "
        "changes, then update the state + DONE",
        "further whole-chapter writes are not allowed. Update the state + DONE",
    ),),
}

WRITE_DESCRIPTION_EDIT = (
    "After that, use correct for any changes; another write will be rejected.",
    "After that, another write will be rejected.",
)

EXPECTED_TOOL_NAMES = {
    "none": {"read", "search", "correct", "write", "update"},
    "no_state": {"read", "search", "correct", "write"},
    "no_lookback": {"write", "update"},
}


def apply_exact(text: str, edits: Edits, surface: str) -> str:
    """Apply edits that must each match exactly once."""
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"ablation edit for {surface} matched {count} times: {old[:80]!r}"
            )
        text = text.replace(old, new)
    return text


def apply_present(text: str, edits: Edits) -> str:
    """Apply edits to text that may or may not contain them."""
    for old, new in edits:
        text = text.replace(old, new)
    return text


def ablate_tools(tools: List[Dict[str, Any]], ablation: str) -> List[Dict[str, Any]]:
    """Return the chapter tool list with the ablated component removed."""
    if ablation == "none":
        return tools
    tools = copy.deepcopy(tools)
    if ablation == "no_state":
        tools = [t for t in tools if t["function"]["name"] != "update"]
    elif ablation == "no_lookback":
        tools = [t for t in tools if t["function"]["name"] not in LOOKBACK_TOOLS]
        write = next(t for t in tools if t["function"]["name"] == "write")
        write["function"]["description"] = apply_exact(
            write["function"]["description"], (WRITE_DESCRIPTION_EDIT,),
            "write tool description",
        )
    names = {t["function"]["name"] for t in tools}
    if names != EXPECTED_TOOL_NAMES[ablation]:
        raise ValueError(f"unexpected tools for {ablation}: {sorted(names)}")
    return tools


class _StateView:
    """Stand-in passed to the core prompt builder to control state rendering."""

    def __init__(self, payload: Any):
        self.payload = payload

    def to_dict(self) -> Any:
        return self.payload


class AblationAgent(core.LongStoryAgent):
    """NstAgent with one component removed; see the module docstring."""

    def __init__(self, *args: Any, ablation: str = "none", **kwargs: Any):
        super().__init__(*args, **kwargs)
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}")
        if (self.state_update_mode != nstagent.STATE_UPDATE_MODE
                or self.narrative_ops_guidance_placement
                != nstagent.GUIDANCE_PLACEMENT):
            raise ValueError("ablations are defined only on NstAgent")
        self.ablation = ablation
        self.ablation_chapter_stats = self._new_chapter_stats()
        v1_system = core.get_system_prompt("chapter", "en")
        self.v1_system_prompt = v1_system
        self.system_prompt = apply_exact(
            v1_system, SYSTEM_EDITS[ablation], "system prompt"
        )

    @staticmethod
    def _new_chapter_stats() -> Dict[str, Any]:
        return {"blocked_tool_calls": {}}

    def _require_english(self, lang: str) -> None:
        if self.ablation != "none" and lang != "en":
            raise ValueError("NstAgent ablations support English prompts only")

    def _build_chapter_prompt(self, prompt: str, outline: List[Dict],
                              chapter_info: Dict, state: core.StoryState,
                              chapters: List[Dict[str, Any]], lang: str) -> str:
        if self.ablation == "none":
            return super()._build_chapter_prompt(
                prompt, outline, chapter_info, state, chapters, lang
            )
        self._require_english(lang)
        if self.ablation == "no_state":
            view = _StateView(STATE_OMITTED)
        else:
            view = _StateView(state.to_dict())
        text = super()._build_chapter_prompt(
            prompt, outline, chapter_info, view, chapters, lang
        )
        text = apply_exact(text, PROMPT_EDITS[self.ablation], "chapter prompt")
        if STATE_OMITTED in text:
            raise ValueError("state placeholder leaked into chapter prompt")
        return text

    def _ablate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Edit the system prompt and controller nudges in place."""
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content")
            if content == self.v1_system_prompt:
                messages[0]["content"] = self.system_prompt
            elif content != self.system_prompt:
                raise ValueError("unexpected chapter system prompt")
        for message in messages[2:]:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                message["content"] = apply_present(
                    message["content"], NUDGE_EDITS[self.ablation]
                )

    async def _llm(self, messages: List[Dict], use_tools: bool = False,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   max_tokens: Optional[int] = None):
        if use_tools and self.ablation != "none":
            self._ablate_messages(messages)
            tools = ablate_tools(
                tools if tools is not None else core.TOOLS, self.ablation
            )
        return await super()._llm(
            messages, use_tools=use_tools, tools=tools, max_tokens=max_tokens
        )

    def _dispatch_tool(self, name: str, args: Dict[str, Any],
                       *rest: Any, **kwargs: Any) -> Dict[str, Any]:
        if self.ablation == "none":
            return super()._dispatch_tool(name, args, *rest, **kwargs)
        blocked = self.ablation_chapter_stats["blocked_tool_calls"]
        if name not in EXPECTED_TOOL_NAMES[self.ablation]:
            blocked[name] = blocked.get(name, 0) + 1
            return {"ok": False, "error": f"unknown tool {name}"}
        result = super()._dispatch_tool(name, args, *rest, **kwargs)
        for key in ("error", "info"):
            if isinstance(result.get(key), str):
                result[key] = apply_present(result[key], RESULT_EDITS[self.ablation])
        return result

    async def _run_chapter_loop(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.ablation_chapter_stats = self._new_chapter_stats()
        written = await super()._run_chapter_loop(*args, **kwargs)
        written["ablation_stats"] = copy.deepcopy(self.ablation_chapter_stats)
        return written

    def _save_failed_chapter_trace(self, trace_dir, story_id, chapter_info,
                                   user_msg, trajectory, runtime, error):
        # The core records nudges before _llm edits them; store what the
        # model actually saw.
        edited = []
        for turn in trajectory:
            if isinstance(turn.get("nudge"), str):
                turn = dict(turn)
                turn["nudge"] = apply_present(turn["nudge"], NUDGE_EDITS[self.ablation])
            edited.append(turn)
        return super()._save_failed_chapter_trace(
            trace_dir, story_id, chapter_info, user_msg, edited, runtime,
            f"[ablation={self.ablation}] {error}",
        )


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def record_identity(ablation: str) -> Dict[str, str]:
    return {
        "method": METHOD_NAME,
        "method_id": f"nstagent_ablation_{ablation}",
        "protocol_version": PROTOCOL_VERSION,
        "ablation": ablation,
    }


async def main_async(args) -> None:
    """core.main_async with AblationAgent and ablation-specific record fields."""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = core.setup_logger(
        "nstagent_ablation", f"logs/nstagent_ablation_{args.ablation}_{ts}.log"
    )

    all_prompts = core.load_prompt_items(args.input)
    logger.info(f"Loaded {len(all_prompts)} prompts")
    end = args.end if args.end is not None else len(all_prompts)
    prompts = all_prompts[args.start:end]

    processed_ids = set()
    if not args.no_resume and os.path.exists(args.output):
        processed_ids, incomplete = core.completed_story_ids(args.output)
        logger.info(
            f"Resuming: {len(processed_ids)} already completed"
            + (f", {incomplete} incomplete will be retried" if incomplete else "")
        )

    to_process = [p for p in prompts if p["id"] not in processed_ids]
    logger.info(f"Generating {len(to_process)} stories")
    if not to_process:
        print("Nothing to generate.")
        return

    client = core.AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        max_retries=args.client_max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrent)
    file_lock = asyncio.Lock()
    trace_base_dir, ckpt_dir = core.prepare_output_dirs(args.output)
    ablation_sha = module_sha256()
    pbar = core.atqdm(total=len(to_process), desc="Generating", unit="story")

    async def process_one(item):
        async with semaphore:
            try:
                writer = AblationAgent(
                    client=client, model=args.model,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    word_count=args.word_count, logger=logger,
                    length_control_mode=args.length_control_mode,
                    chapter_token_control=args.chapter_token_control,
                    dynamic_token_ratio_en=args.dynamic_token_ratio_en,
                    dynamic_token_ratio_zh=args.dynamic_token_ratio_zh,
                    dynamic_token_overhead=args.dynamic_token_overhead,
                    dynamic_token_minimum=args.dynamic_token_minimum,
                    max_turns_per_chapter=args.max_turns_per_chapter,
                    request_attempts=args.request_attempts,
                    state_update_mode=args.state_update_mode,
                    narrative_ops_guidance_placement=(
                        args.narrative_ops_guidance_placement
                    ),
                    disable_thinking=args.disable_thinking,
                    outline_cache_dir=args.outline_cache_dir,
                    outline_cache_policy=args.outline_cache_policy,
                    outline_cache_model_id=args.outline_cache_model_id,
                    outline_cache_lock_timeout=args.outline_cache_lock_timeout,
                    ablation=args.ablation,
                )
                record = await core.generate_story_with_auto_resume(
                    writer,
                    item["id"],
                    item["prompt"],
                    trace_base_dir=trace_base_dir,
                    ckpt_dir=ckpt_dir,
                    initial_resume=not args.no_resume,
                    max_auto_resumes=args.auto_resumes,
                    logger=logger,
                )
                record["generation_config"] = {
                    "length_control_mode": args.length_control_mode,
                    "chapter_token_control": args.chapter_token_control,
                    "max_tokens": args.max_tokens,
                    "dynamic_token_ratio_en": args.dynamic_token_ratio_en,
                    "dynamic_token_ratio_zh": args.dynamic_token_ratio_zh,
                    "dynamic_token_overhead": args.dynamic_token_overhead,
                    "dynamic_token_minimum": args.dynamic_token_minimum,
                    "max_turns_per_chapter": args.max_turns_per_chapter,
                    "request_attempts": args.request_attempts,
                    "state_update_mode": args.state_update_mode,
                    "narrative_ops_guidance_placement": (
                        args.narrative_ops_guidance_placement
                    ),
                    "disable_thinking": args.disable_thinking,
                    "outline_cache_policy": args.outline_cache_policy,
                    "outline_cache_model_id": (
                        args.outline_cache_model_id or args.model
                    ),
                    "ablation": args.ablation,
                    "core_sha256": nstagent.current_core_sha256(),
                    "ablation_module_sha256": ablation_sha,
                }
                record.update(record_identity(args.ablation))
                record["language"] = item.get("language", "")
                total_words = sum(
                    core.count_words(ch["content"]) for ch in record["story"]
                )
                record["target_word_count"] = args.word_count
                record["word_count"] = total_words
                whole_min_words = math.ceil(
                    args.word_count * (1 - core.WORD_COUNT_TOLERANCE)
                )
                whole_max_words = math.floor(
                    args.word_count * (1 + core.WORD_COUNT_TOLERANCE)
                )
                record["whole_story_length_ok"] = (
                    whole_min_words <= total_words <= whole_max_words
                )
                record["accepted_word_count_range"] = [
                    whole_min_words, whole_max_words,
                ]
                record["whole_story_length_enforced"] = False
                record["complete"] = bool(record.get("complete"))
                if not record["story"]:
                    logger.error(f"Story {item['id']} produced no chapters; not saved")
                    return
                async with file_lock:
                    with open(args.output, "a", encoding="utf-8") as file:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
                status = "saved" if record["complete"] else "saved INCOMPLETE"
                logger.info(f"Story {item['id']} {status}: {total_words} words, "
                            f"{len(record['story'])}/{len(record['outline'])} chapters")
            except Exception as e:
                logger.error(f"Story {item['id']} failed: {e}", exc_info=True)
            finally:
                pbar.update(1)

    await asyncio.gather(*[process_one(item) for item in to_process])
    pbar.close()
    print(f"Done! Stories saved to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = nstagent.build_arg_parser()
    parser.prog = "nstagent_ablation.py"
    parser.description = "NstAgent component ablations"
    parser.add_argument("--ablation", required=True, choices=ABLATIONS)
    return parser


def check_output_manifest(output: str, ablation: str) -> None:
    """One output directory holds exactly one ablation (checkpoints are shared)."""
    path = os.path.join(os.path.dirname(output) or ".", "ablation_manifest.json")
    manifest = {
        "ablation": ablation,
        "protocol_version": PROTOCOL_VERSION,
    }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != manifest:
            raise SystemExit(
                f"{path} records a different ablation: {existing}"
            )
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    nstagent.validate_args(parser, args)
    if args.ablation != "none":
        items = core.load_prompt_items(args.input, args.start, args.end)
        other = [i["id"] for i in items if core.detect_language(i["prompt"]) != "en"]
        if other:
            parser.error(f"ablations support English prompts only; non-English ids: {other[:5]}")
    check_output_manifest(args.output, args.ablation)
    print(f"{METHOD_NAME} protocol: {PROTOCOL_VERSION}; ablation={args.ablation}")
    print(f"Core SHA-256: {nstagent.current_core_sha256()}")
    print(f"Ablation module SHA-256: {module_sha256()}")
    core.print_run_config(args)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
