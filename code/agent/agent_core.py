#!/usr/bin/env python
# coding: utf-8
"""LongStoryAgent: single-file long-form story generator with structured state.

Sections:
    1. Configuration and constants
    2. Prompts (English and Chinese)
    3. Text, JSON, and tool-call parsing
    4. Checkpoints, tool protocol, and structured narrative state
    5. Multi-stage outline generation
    6. Per-chapter tool loop and full-story generation
    7. Batch processing, auto-resume, and CLI entry point

The agent first builds a four-level outline (premise → detail → acts →
chapters), then writes chapter by chapter with read / search / correct / write
and state-update tools. A checkpoint is saved after every chapter; transient
failures resume from the last complete chapter.
"""

import argparse
import asyncio
import copy
import difflib
import hashlib
import inspect
import json
import logging
import math
import os
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import jieba
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# jieba logs dictionary loading at INFO; silence it to keep run logs clean.
jieba.setLogLevel(logging.WARNING)

# =============================================================================
# 1. Configuration and constants
# =============================================================================

DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENT = 5
DEFAULT_WORD_COUNT = 10000
MAX_READ = 3
MAX_SEARCH = 5
SEARCH_MAX_HITS = 8
MAX_CORRECT_PER_CHAPTER = 3
# Max model/tool turns per chapter for the write/update/DONE loop.
MAX_TURNS_PER_CHAPTER = 50
MAX_JSON_RETRIES = 5
DEFAULT_AUTO_RESUMES = 3
AUTO_RESUME_BACKOFF_SECONDS = 5
WORD_COUNT_TOLERANCE = 0.20
WORD_COUNT_TOLERANCE_PERCENT = 20
LONG_FORM_MIN_WORDS = 10000
DEFAULT_LENGTH_CONTROL_MODE = "compact"
LENGTH_CONTROL_MODES = (
    "compact",
    "tool_target",
    "recent_failure_context",
    "feedback_only",
    "adaptive_recent_failure",
)
ADAPTIVE_LENGTH_MIN_IMPROVEMENT = 0.10
STATE_UPDATE_MODES = (
    "batch", "narrative_ops", "state_grouped", "state_first", "simple",
    "semantic",
)
# Default: the atomic batch update (adds/edits/deletes).
# simple/semantic are opt-in experimental modes.
DEFAULT_STATE_UPDATE_MODE = "batch"
NARRATIVE_OPS_GUIDANCE_PLACEMENTS = ("both", "prompt", "tool")
# Detailed update rules live in the chapter prompt by default; the update
# tool description stays minimal.
DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT = "prompt"

# Outline cache entries intentionally do not include ``state_update_mode`` in
# their identity.  A paired state-update experiment must consume the same
# premise/detail/acts/chapter outline rather than independently sampling the
# planning stages.  Bump this version only for a storage-format migration; the
# prompt/pipeline fingerprint below automatically invalidates planning changes.
OUTLINE_CACHE_SCHEMA_VERSION = 1
OUTLINE_CACHE_POLICIES = ("off", "read-write", "require", "refresh")
DEFAULT_OUTLINE_CACHE_POLICY = "off"
DEFAULT_OUTLINE_CACHE_LOCK_TIMEOUT = 7200.0

# The dynamic chapter budget caps only newly generated tokens, not input
# context. Ratios come from Qwen3.5-4B tool-call probes (~1.38 tokens/word in
# English, ~0.80-0.91 tokens/character in Chinese). The budget uses the
# chapter target, not the upper bound, plus headroom for reasoning and tool JSON.
CHAPTER_TOKEN_CONTROL_MODES = ("fixed", "dynamic")
DEFAULT_CHAPTER_TOKEN_CONTROL = "fixed"
DEFAULT_DYNAMIC_TOKEN_RATIO_EN = 1.35
DEFAULT_DYNAMIC_TOKEN_RATIO_ZH = 0.85
DEFAULT_DYNAMIC_TOKEN_OVERHEAD = 384
DEFAULT_DYNAMIC_TOKEN_MINIMUM = 1024

CHAPTER_ANCHORS = [
    (0, 0),
    (10000, 10),
    (20000, 15),
    (50000, 25),
    (100000, 40),
    (1000000, 100),
]


# =============================================================================
# 2. Prompts (English and Chinese)
# =============================================================================

# System prompts shared by outline planning and chapter writing.
SYSTEM_PROMPTS = {
    "outline_zh": "你是一位经验丰富的小说家，善于精心规划故事结构。",
    "outline_en": "You are an experienced novelist who plans stories carefully.",
    "chapter_zh": (
        "你是一位经验丰富的小说家，正在按章节撰写一部长篇小说。"
        "请使用提供的工具，阅读已写章节、搜寻指定内容、撰写当前章节、"
        "修复不一致，并维护结构化的叙事状态。"
    ),
    "chapter_en": (
        "You are an experienced novelist writing a long novel chapter by chapter. "
        "Use the provided tools to read written chapters, search for specified "
        "content, write the current chapter, fix inconsistencies, and maintain "
        "the structured narrative state."
    ),
}


def get_system_prompt(task: str, lang: str) -> str:
    """Return the system prompt for a task and language; unknown languages use English."""
    key = f"{task}_{lang}"
    return SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS.get(f"{task}_en", ""))


def outline_premise_prompt(lang: str, prompt: str, word_count: int) -> str:
    """Stage 1: generate a ~300-word high-level premise."""
    if lang == "zh":
        return (
            f"故事要求：{prompt}\n\n"
            f"最终成稿约 {word_count} 字。请写一段约 300 字的高层提要，"
            "涵盖核心冲突、主角、设定、中心主题，以及从开篇到结局的整体叙事弧线。"
            "暂不分章。\n\n"
            "只输出提要正文，不要标题或额外说明。"
        )
    return (
        f"Story prompt: {prompt}\n\n"
        f"The final novel will be about {word_count} words. "
        "Write a high-level premise of approximately 300 words. "
        "Cover the core conflict, protagonist, setting, central theme, "
        "and the overall narrative arc from beginning to end. "
        "Do not break it into chapters yet.\n\n"
        "Output ONLY the premise text, no headings or extra commentary."
    )


def outline_detail_prompt(lang: str, prompt: str, premise: str,
                          target_words: int) -> str:
    """Stage 2: expand the premise into a 1000–2500-word detailed synopsis."""
    if lang == "zh":
        return (
            f"故事要求：{prompt}\n\n"
            f"高层提要：\n{premise}\n\n"
            f"将提要扩展为约 {target_words} 字的详细梗概。"
            "梗概应按顺序梳理主要情节节拍，介绍主要人物及其动机，"
            "描写关键场景，列出主要转折与结局。暂不分章。\n\n"
            "只输出梗概正文。"
        )
    return (
        f"Story prompt: {prompt}\n\n"
        f"High-level premise:\n{premise}\n\n"
        f"Expand the premise into a detailed synopsis of approximately "
        f"{target_words} words. The synopsis should walk through the major "
        "story beats in order, introduce the principal characters and "
        "their motivations, describe key locations, and lay out the major "
        "turning points and resolution. Do not break it into chapters yet.\n\n"
        "Output ONLY the synopsis text."
    )


def outline_acts_prompt(lang: str, prompt: str, premise: str,
                        detail: str, word_count: int) -> str:
    """Stage 3: divide the detailed synopsis into 3–6 acts or volumes."""
    if lang == "zh":
        return (
            f"故事要求：{prompt}\n\n"
            f"提要：\n{premise}\n\n"
            f"详细梗概：\n{detail}\n\n"
            f"这是一部约 {word_count} 字的长篇小说。请将故事划分为少量幕或卷"
            "（通常 3-6 个）。每一幕给出标题和 2-4 句描述，概述该幕中发生的事。"
            "各幕必须共同覆盖整个详细梗概，没有遗漏。\n\n"
            "以可读文本格式输出幕级大纲，每一幕清楚编号并加标题，随后是描述。"
        )
    return (
        f"Story prompt: {prompt}\n\n"
        f"Premise:\n{premise}\n\n"
        f"Detailed synopsis:\n{detail}\n\n"
        f"This is a long novel of about {word_count} words. "
        "Divide the story into a small number of acts or volumes "
        "(typically 3-6). For each act, give a title and a 2-4 sentence "
        "description summarizing what happens within that act. The acts "
        "must collectively cover the entire detailed synopsis without "
        "gaps.\n\n"
        "Output the act-level outline as readable text. Each act should be "
        "clearly numbered and titled, followed by its description."
    )


def outline_chapters_prompt(lang: str, prompt: str, context: str,
                            word_count: int, rec: int,
                            per_chapter: int) -> str:
    """Stage 4: generate the chapter-level outline with per-chapter word targets."""
    if lang == "zh":
        rec_line = (
            f"建议章节数：约 {rec} 章（每章平均目标约 {per_chapter} 字）。"
            "如剧情确有需要可以稍作调整，但应贴近这一数字。每章的 word_count 由你决定。"
        ) if rec else (
            "故事过短，无法分章。请返回单个章节，包含整篇内容。"
        )
        return (
            f"故事要求：{prompt}\n\n"
            f"前述大纲：\n{context}\n\n"
            f"整部小说的目标总字数为 {word_count} 字。{rec_line}\n\n"
            "请为每一章提供：\n"
            "  - id: 整数，从 0 开始递增。\n"
            "  - name: 章节标题（简短）。\n"
            "  - description: 该章的大纲描述，概述关键事件、人物时刻、"
            "情节发展。\n"
            "  - word_count: 该章的目标字数。所有章节 word_count 之和"
            f"应在总目标字数的 ±{WORD_COUNT_TOLERANCE_PERCENT}% 以内。"
            "允许重点章节分配更多字数，过渡章节更少。\n\n"
            "以 JSON 数组格式输出章节大纲：\n"
            "[\n"
            '  {"id": 0, "name": "...", "description": "...", "word_count": <int>},\n'
            '  {"id": 1, "name": "...", "description": "...", "word_count": <int>},\n'
            "  ...\n"
            "]\n\n"
            "确保每章描述覆盖该章的核心内容。"
        )
    rec_line = (
        f"Recommended number of chapters: approximately {rec} chapters "
        f"(average target ~{per_chapter} words per chapter). "
        "You may adjust slightly if the plot requires it, but stay close. "
        "The word_count for each chapter is up to you."
    ) if rec else (
        "The story is too short to break into chapters. "
        "Return a single chapter containing the entire story."
    )
    return (
        f"Story prompt: {prompt}\n\n"
        f"Prior outline:\n{context}\n\n"
        f"The total target word count for the entire novel is {word_count}. "
        f"{rec_line}\n\n"
        "For each chapter, provide:\n"
        "  - id: an integer, starting from 0 and incrementing.\n"
        "  - name: a short chapter title.\n"
        "  - description: an outline description summarizing "
        "the key events, character moments, and plot progression in this chapter.\n"
        "  - word_count: the target word count for this chapter. The sum of all "
        f"chapter word_count values must stay within "
        f"±{WORD_COUNT_TOLERANCE_PERCENT}% of the total target. "
        "You may assign more words to important chapters and fewer to transitional ones.\n\n"
        "Output the chapter-level outline as a JSON array:\n"
        "[\n"
        '  {"id": 0, "name": "...", "description": "...", "word_count": <int>},\n'
        '  {"id": 1, "name": "...", "description": "...", "word_count": <int>},\n'
        "  ...\n"
        "]\n\n"
        "Ensure each description covers the essential content of the chapter."
    )


def requires_dialogue_only(story_contract: str) -> bool:
    """Detect hard dialogue-only format requirements that can be reliably enforced."""
    text = " ".join((story_contract or "").casefold().split())
    markers = (
        "dialogue only",
        "dialogue-only",
        "entire narrative in dialogue",
        "entirely in dialogue",
        "entirely as dialogue",
        "no narration",
        "纯对话",
        "仅用对话",
        "只用对话",
        "无旁白",
    )
    return any(marker in text for marker in markers)


def visible_outline_for_chapter(
        outline: List[Dict[str, Any]], chapter_info: Dict[str, Any],
        state_update_mode: str) -> List[Dict[str, Any]]:
    """Keep unreached plot revelations out of the prose model's context.

    The controller retains the full outline.  In semantic mode the prose model
    receives the immutable contract, compact past state, and only the current
    chapter plan.  This is sufficient to execute the chapter and prevents a
    small model from copying future twists or juggling dozens of irrelevant
    chapter descriptions while writing the present scene.
    """
    if state_update_mode == "semantic":
        return [copy.deepcopy(chapter_info)]
    return outline


def explicit_contract_checklist(story_contract: str, ch_id: int) -> str:
    """Return a short deterministic checklist for common hard constraints.

    The original prompt remains the source of truth.  This helper does not ask
    a small model to rewrite the contract (which can silently drop details); it
    merely makes a few high-risk, mechanically recognizable constraints hard
    to overlook at chapter-writing time.
    """
    text = " ".join((story_contract or "").casefold().split())
    rules: List[str] = []
    if "teenage" in text or "teenager" in text:
        rules.append(
            "Any character described by the prompt as a teenager must be "
            "13-19 at that stage of the story; never make that character "
            "twelve or younger."
        )
    if (
        "adolescence to adulthood" in text
        or "teenager to adult" in text
        or "teenage girl to" in text
    ):
        rules.append(
            "The outline and finished narrative must visibly cover the "
            "requested progression from adolescence into adulthood rather "
            "than compressing the whole arc into days or weeks."
        )
    if requires_dialogue_only(story_contract):
        rules.append(
            "The stored prose itself must contain dialogue only; evaluation "
            "or serialization must not inject chapter headings into it."
        )
        rules.append(
            "Different speakers must remain distinguishable without names or "
            "tags through stable differences in diction, rhythm, knowledge, "
            "and worldview; avoid recursive question-and-answer loops, repeated "
            "neighboring utterances, and long runs of one-clause fragments."
        )
    if "slowly reveal" in text or "gradually reveal" in text:
        rules.append(
            "Do not disclose the central answer in the opening chapters. "
            "Preserve the planned reveal for its assigned midpoint-or-later "
            "chapter and use only partial evidence beforehand."
        )
    if not rules:
        return ""
    heading = "# Deterministic Contract Checklist\n"
    return heading + "\n".join(f"- {rule}" for rule in rules) + "\n\n"


def chapter_prompt(lang: str, outline: str, state: str, ch_id: int,
                  ch_name: str, ch_desc: str, ch_target: int,
                  completed: int, completed_words: int,
                  state_update_mode: str = DEFAULT_STATE_UPDATE_MODE,
                  narrative_ops_guidance_placement: str =
                  DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT,
                  story_contract: str = "") -> str:
    """Build the chapter prompt, including tool order and state-maintenance rules."""
    if narrative_ops_guidance_placement not in \
            NARRATIVE_OPS_GUIDANCE_PLACEMENTS:
        raise ValueError(
            "unknown narrative_ops_guidance_placement "
            f"{narrative_ops_guidance_placement!r}"
        )
    state_grouped_updates = state_update_mode == "state_grouped"
    state_first_updates = state_update_mode == "state_first"
    narrative_ops_updates = state_update_mode == "narrative_ops"
    simple_updates = state_update_mode == "simple"
    semantic_updates = state_update_mode == "semantic"
    dialogue_only = semantic_updates and requires_dialogue_only(story_contract)
    dialogue_min_lines = max(20, math.ceil(ch_target / 20))
    dialogue_max_lines = max(dialogue_min_lines, math.ceil(ch_target / 8))
    if lang == "zh":
        if semantic_updates:
            update_step = (
                "4. 正文写入成功后会进入独立状态阶段：必须调用一次 "
                "record_chapter_summary，只记录本章的一条因果摘要；再按需更新角色、"
                "新增或兑现未来回报。\n"
            )
            update_rules = (
                "- record_chapter_summary 每章只保留一条不超过 100 个汉字的因果摘要，"
                "不要逐场景追加事件；update_character 必须给出角色完整的当前状态，"
                "包括仍然有效的目标、关系、知识及身心处境，并覆盖本章新登场或有"
                "实质变化的每个具名常驻角色，且不超过 140 个汉字，不复述生平或整章；"
                "add_requirement 只记录本章新引入、"
                "以后必须用一个具体动作兑现的义务、未解问题或承诺回报，每章至多"
                " 1 条；description 必须明确未来要揭示、回答、返回、面对、兑现或"
                "解决什么，且不超过 80 个汉字。"
                "不得把持续威胁、背景事实、已经完成的发现、角色快照中已有的目标，"
                "或当前状态已经表达的同一事项重复加入；"
                "兑现时用 resolve_requirement_by_text 复制或概括该事项文本，"
                "不要引用数字 id。\n"
            )
        elif state_grouped_updates:
            update_step = (
                "4. 正文确定后只调用一次 update；在 state_updates 中先按 "
                "state_name 对全部变化分组，再为每组填写 adds/deletes/edits，"
                "整个调用原子提交。\n"
            )
            update_rules = (
                "- character_states、past_events、future_requirements 的变化必须"
                "合并在同一次 update 中；每个状态集合最多出现一次，空操作组传 []。"
                "状态只保留后文真正需要的信息，兑现后及时 edit/delete。\n"
            )
        elif state_first_updates:
            update_step = (
                "4. 正文确定后，按发生变化的状态类型分别调用 update；每次调用先选 "
                "state_name，再填写 adds/deletes/edits 三个数组，同一状态类型的变化"
                "在一次调用中原子提交。\n"
            )
            update_rules = (
                "- character_states、past_events、future_requirements 必须分开"
                "调用 update；空操作组传 []。状态只保留后文真正需要的信息，兑现后"
                "及时 edit/delete。\n"
            )
        elif narrative_ops_updates:
            update_step = (
                "4. 正文确定后只调用一次 update；在同一次调用中填写角色状态"
                " upsert、新增过去事件、新增未来要求和兑现未来要求四个数组。\n"
            )
            if narrative_ops_guidance_placement in ("both", "prompt"):
                update_rules = (
                    "- upsert_character_state 使用 {name, description} 写入角色完整的当前"
                    "位置、目标、关系、知识、持有物和身心状况，同名角色自动替换；"
                    "add_past_event 只添加大纲中没有明示且可能影响后续情节的本章事件，"
                    "并为它设置稳定的 snake_case key；add_future_requirement 只添加以后"
                    "必须兑现的具体事项，也使用稳定 key；resolve_future_requirement "
                    "传入已兑现事项的 key。四个字段都必须是原生 JSON 数组，无变化时传 []。\n"
                )
            else:
                update_rules = (
                    "- 按 update 工具的说明和参数结构提交本章全部叙事状态变化。\n"
                )
        elif simple_updates:
            update_step = (
            "4. 正文确定后，对每项变化调用一个对应的简单状态工具；"
            "同一轮可返回多个工具调用。\n"
            )
            update_rules = (
            "- 用 update_character 替换角色的当前状态（新角色会自动创建）；"
            "用 record_past_event 记录每章 1-4 个关键过去事件；"
            "用 add_requirement 添加尚未兑现的伏笔或承诺；"
            "兑现后用 resolve_requirement 和现有 id 删除。\n"
            )
        else:
            update_step = "4. 正文确定后，使用 update 批量更新本章造成的状态变化。\n"
            update_rules = (
            "- 状态只保留后文真正需要的信息：角色当前处境的实质变化、每章 1-4 条"
            "关键事件，以及尚未兑现的伏笔或承诺；兑现后及时 edit/delete。\n"
            )
        story_prompt_section = (
            f"# 原始故事提示\n{story_contract}\n\n"
            if story_contract else ""
        )
        contract_guidance = (
            "# 故事契约解释\n"
            "其中逐章约束（格式、视角、文体和禁用内容）在每章都必须遵守；"
            "全书级要求（如中点反转、逐步揭示和结局）只在大纲指定的阶段兑现，"
            "不要强迫当前章提前完成。两类约束都优先于大纲中的泛化措辞。\n\n"
            if semantic_updates and story_contract else ""
        )
        format_guard = (
            "# 强制纯对话格式\n"
            "write.content 的每一个非空行都必须是一条以破折号“—”开头的完整"
            "台词；推荐使用 —\"完整台词\"，若小模型难以稳定输出引号，也允许"
            " —完整台词。每行只能有一条不标注说话者、长度不超过 160 词的台词。"
            "不得把整章塞进一条超长"
            "台词；不得出现标题、旁白、场景描写、动作说明、"
            "舞台指示、说话者姓名或“某人说”等标签。动作、环境、情绪和心理信息"
            "只能通过台词表达。相邻台词不得相同，全章至少 90% 台词应互不重复；"
            f"本章建议使用约 {dialogue_min_lines}–{dialogue_max_lines} 条有实质"
            "内容的台词，而不是数百条单句碎片。每条通常表达新的事实、立场或"
            "决定。禁止示例：连续两行都是“—你明白吗？”。write 工具会拒绝"
            "格式合规但机械重复的正文。\n\n"
            if dialogue_only else ""
        )
        prose_expansion_rule = (
            "- 大纲只是计划；只能在故事契约允许的形式内展开。若契约限制格式，"
            "场景、动作、环境、感官和心理信息也必须只用该形式表达。\n"
            if semantic_updates else
            "- 章节大纲只是计划，请扩展为包含场景、对话、感官细节和内心活动的正文。\n"
        )
        return (
            f"你正在创作一部长篇小说的第 {ch_id} 章。\n\n"
            f"{story_prompt_section}"
            f"{contract_guidance}"
            f"{format_guard}"
        f"#{' 当前章节计划' if semantic_updates else ' 完整冻结大纲'}\n{outline}\n\n"
            f"# 当前叙事状态\n{state}\n\n"
            f"# 已完成章节\n"
            f"前 {completed} 章已经写完。\n\n"
            f"# 当前任务\n"
            f"章节 ID: {ch_id}\n"
            f"章节名: {ch_name}\n"
            f"大纲: {ch_desc}\n"
            f"目标字数: {ch_target} 字。请严格控制在该目标字数的 "
            f"±{WORD_COUNT_TOLERANCE_PERCENT}% 以内。\n\n"
            "执行顺序：\n"
            "1. 按需使用 read 回顾指定章节，或使用 search 定位前文事实。\n"
            "2. 使用 write 写入当前章的完整正文。\n"
            "3. 如有必要，使用 correct 精确修正当前章或前文的一致性错误。\n"
            f"{update_step}"
            "5. 全部完成后，不调用工具，只输出 DONE；其他文本不会结束本章。\n\n"
            "写作与状态要求：\n"
            f"{prose_expansion_rule}"
            f"{update_rules}"
            "- 具体调用条件、次数和参数以工具说明为准。"
        )
    if semantic_updates:
        update_step = (
            "4. After write succeeds, the controller will switch to a separate "
            "state phase and provide the state instructions then. Focus only on "
            "producing excellent chapter prose during the current phase.\n"
        )
        update_rules = (
            "- Do not plan or call state tools until the controller explicitly "
            "announces the state phase.\n"
        )
    elif state_grouped_updates:
        update_step = (
            "4. Once the prose is final, make exactly one atomic update call. "
            "Inside state_updates, group every change by state_name first, "
            "then provide adds, deletes, and edits for that collection.\n"
        )
        update_rules = (
            "- Put character_states, past_events, and future_requirements "
            "changes in the same update call, with at most one group per "
            "state_name and [] for unused operation groups. Keep only state "
            "future chapters genuinely need; edit/delete it after payoff.\n"
        )
    elif state_first_updates:
        update_step = (
            "4. Once the prose is final, call update separately for each "
            "changed state_name. In every call select state_name first, then "
            "provide adds, deletes, and edits; changes within that state "
            "collection are atomic.\n"
        )
        update_rules = (
            "- Call character_states, past_events, and future_requirements "
            "updates separately, and pass [] for unused operation groups. Keep "
            "only state future chapters genuinely need; edit/delete it after "
            "payoff.\n"
        )
    elif narrative_ops_updates:
        update_step = (
            "4. Once the prose is final, make exactly one update call containing "
            "character-state upserts, newly established completed events, "
            "new future "
            "requirements, and resolved future-requirement keys.\n"
        )
        if narrative_ops_guidance_placement in ("both", "prompt"):
            update_rules = (
                "- In upsert_character_state, provide {name, description} with each "
                "character's complete current location, goal, relationships, "
                "knowledge, possessions, and physical or emotional condition; an "
                "existing name is replaced rather than appended. In add_past_event, "
                "record a completed event only when the fact is "
                "not already explicit in the frozen outline and may affect later "
                "plot; give it a stable snake_case key. Add only concrete later "
                "obligations to add_future_requirement, also with stable keys, and "
                "pass the keys of requirements fulfilled in this chapter to "
                "resolve_future_requirement. Every field must "
                "be a native JSON array; "
                "never serialize or quote an array as a string. Include all four "
                "arrays and use [] when one has no changes.\n"
            )
        else:
            update_rules = (
                "- Submit every narrative-state change from this chapter using "
                "the update tool's description and parameter schema.\n"
            )
    elif simple_updates:
        update_step = (
        "4. Once the prose is final, call one matching simple state tool for "
        "each change; you may return multiple tool calls in the same turn.\n"
        )
        update_rules = (
        "- Use update_character to replace a character's current state (a new "
        "name is created automatically), record_past_event for 1-4 pivotal events "
        "that happened, add_requirement for an unresolved setup or promise, "
        "and resolve_requirement with its existing id after payoff.\n"
        )
    else:
        update_step = (
            "4. Once the prose is final, use update to batch-apply state changes "
            "caused by this chapter.\n"
        )
        update_rules = (
        "- Keep only state that future chapters genuinely need: substantive "
        "changes in a character's current situation, 1-4 pivotal events per "
        "chapter, and unresolved setups or promises. Edit/delete them after payoff.\n"
        )
    story_prompt_section = (
        f"# Original Story Prompt\n{story_contract}\n\n"
        if story_contract else ""
    )
    contract_guidance = (
        "# Story-Contract Interpretation\n"
        "Obey chapter-local constraints (format, point of view, style, and "
        "prohibitions) in every chapter. Fulfill story-level arc requirements "
        "such as a midpoint twist, gradual reveal, or ending only in the "
        "outline-assigned chapters; do not force them into the current chapter. "
        "Both kinds of constraint override generic wording in the outline.\n\n"
        if semantic_updates and story_contract else ""
    )
    contract_checklist = (
        explicit_contract_checklist(story_contract, ch_id)
        if semantic_updates else ""
    )
    format_guard = (
        "# Enforced Dialogue-Only Format\n"
        "Every non-empty line of write.content MUST be one complete spoken "
        "utterance beginning with an em dash. Prefer —\"complete utterance\"; "
        "if quoting is unreliable, —complete utterance is also valid. Each line "
        "must contain one unattributed utterance of at most 160 words. "
        "Do not put the chapter into one giant utterance. Do not use headings, "
        "narration, scene description, action beats, stage directions, speaker "
        "names, colons as labels, or tags such as 'she said'. Convey action, "
        "setting, emotion, and inner experience only through the spoken words. "
        "Never repeat the same utterance on neighboring lines; at least 90% of "
        f"utterances must be unique. For this chapter, use roughly "
        f"{dialogue_min_lines}-{dialogue_max_lines} substantive utterances, with "
        "each line normally adding a new fact, position, or decision. Internally "
        "assign every recurring voice a distinct diction, rhythm, knowledge, and "
        "attitude before writing, but do not print that plan or any speaker label. "
        "Do not pad the chapter by paraphrasing the preceding utterance, repeating "
        "a question, or cycling through abstract words with small variations. "
        "The write tool rejects dialogue that is formally valid but mechanically "
        "repetitive.\n\n"
        if dialogue_only else ""
    )
    prose_expansion_rule = (
        "- The outline is a plan; expand it only through the form permitted by "
        "the Story Contract. When the contract restricts format, convey scene, "
        "action, setting, sensation, and inner life only through that form.\n"
        if semantic_updates else
        "- The outline is a plan, not prose; expand it into scenes with dialogue, "
        "sensory detail, and internal experience.\n"
    )
    if semantic_updates:
        prose_expansion_rule += (
            "- Write only the events and discoveries assigned to the current "
            "chapter. The later outline is private planning context: never "
            "prematurely reveal a twist, answer, identity, or resolution that "
            "belongs to a future chapter.\n"
        )
    return (
        f"You are writing Chapter {ch_id} of a long novel.\n\n"
        f"{story_prompt_section}"
        f"{contract_guidance}"
        f"{contract_checklist}"
        f"{format_guard}"
        f"# {'Current Chapter Plan' if semantic_updates else 'Full Frozen Outline'}\n"
        f"{outline}\n\n"
        f"# Current Narrative State\n{state}\n\n"
        f"# Completed Chapters\n"
        f"The first {completed} chapters have been completed.\n\n"
        f"# Current Task\n"
        f"Chapter ID: {ch_id}\n"
        f"Chapter Name: {ch_name}\n"
        f"Description: {ch_desc}\n"
        f"Target word count for this chapter: {ch_target}. "
        f"Keep the chapter strictly within ±{WORD_COUNT_TOLERANCE_PERCENT}% "
        "of this target word count.\n\n"
        "Execution order:\n"
        "1. Use read to revisit a specific chapter, or search to locate prior "
        "facts, as needed.\n"
        "2. Use write to write the complete current chapter.\n"
        "3. If necessary, use correct to precisely fix consistency errors in "
        "the current or a prior chapter.\n"
        f"{update_step}"
        "5. When everything is complete, make no tool call and output only "
        "DONE; any other text does not finish the chapter.\n\n"
        "Writing and state requirements:\n"
        f"{prose_expansion_rule}"
        f"{update_rules}"
        "- Follow each tool's own conditions, limits, and schema."
    )


def semantic_state_phase_prompt(lang: str, chapter_id: int) -> str:
    """Give the small model state instructions only after prose is accepted."""
    if lang == "zh":
        return (
            f"第 {chapter_id} 章正文已通过并锁定。现在只维护紧凑叙事状态：先调用"
            "一次 record_chapter_summary，写一条不超过 100 个汉字的因果摘要；"
            "再只为本章新登场或有实质变化的具名常驻角色调用 update_character。"
            "只有本章新引入、以后必须用具体动作兑现的事项才调用一次 "
            "add_requirement；已兑现事项按文本调用 resolve_requirement_by_text。"
            "不要记录背景事实、持续规则、已完成发现或重复事项。完成后只输出 DONE。"
        )
    return (
        f"Chapter {chapter_id} prose is accepted and locked. Now maintain only "
        "compact narrative state. First call record_chapter_summary exactly once "
        "with one causal summary of at most 60 words. Then call update_character "
        "only for a named recurring character introduced or materially changed "
        "here. Call add_requirement at most once, only for a new item requiring "
        "a concrete future payoff; resolve a paid-off item by text. Do not store "
        "background facts, ongoing rules, completed discoveries, or duplicates. "
        "When no material state work remains, output only DONE."
    )


# =============================================================================
# 3. Logging, text, JSON, and tool-call parsing
# =============================================================================

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r'```json\s*\n?(.*?)\n?\s*```', r'```\s*\n?(.*?)\n?\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    best: Any = None
    best_len = -1
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        for start in (i for i, c in enumerate(text) if c == start_char):
            depth, in_string, escape = 0, False, False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
                        size = i + 1 - start
                        if size > best_len:
                            best, best_len = parsed, size
                        break
    if best_len > 0:
        return best
    raise ValueError(f"Could not extract JSON: {text[:300]}...")


# OpenAI-compatible endpoints name the reasoning field differently: DeepSeek/Qwen
# usually use reasoning_content, others use reasoning; try both in order.
REASONING_FIELDS = ("reasoning_content", "reasoning")


def extract_content(message) -> str:
    content = message.content
    if content:
        return content.strip()
    d = message.model_dump()
    for field in REASONING_FIELDS:
        reasoning = d.get(field, "") or ""
        if not reasoning:
            continue
        if "</think>" in reasoning:
            return reasoning.split("</think>", 1)[1].strip()
        return reasoning.strip()
    raise ValueError("Empty response: no content or reasoning")


_CJK_CHAR = re.compile(
    r"[㐀-䶿一-鿿豈-﫿]"
    r"|[\U00020000-\U0002ffff]"
)
_ASCII_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
CJK_RANGE = _CJK_CHAR


def count_words(text: str) -> int:
    """Count mixed-language words: each CJK character and each ASCII word counts 1."""
    if not text:
        return 0
    return len(_CJK_CHAR.findall(text)) + len(_ASCII_WORD.findall(text))


def semantic_state_text_too_long(
        text: str, *, max_words: int, max_cjk_chars: int) -> bool:
    """Use units a model can reliably follow when writing compact state.

    English limits are expressed in words.  For predominantly Chinese text,
    the prompt naturally says ``字``; counting every Han character as an
    English word made already-compact Chinese summaries fail repeatedly.
    """
    if detect_language(text) == "zh":
        return len(_CJK_CHAR.findall(text)) > max_cjk_chars
    return count_words(text) > max_words


def normalize_outline_word_targets(
        outline: List[Dict[str, Any]], total_target: int) -> List[Dict[str, Any]]:
    """Scale chapter targets to sum exactly to the requested book length.

    The outliner is allowed a broad whole-book tolerance.  If every chapter
    then lands near its own upper bound, those deviations compound and can
    push an otherwise valid story outside the final tolerance.  Proportional
    normalization preserves the outline's pacing while removing that drift.
    """
    if not outline or total_target <= 0:
        return outline
    raw = [max(1, int(item.get("word_count") or 1)) for item in outline]
    raw_total = sum(raw)
    exact = [value * total_target / raw_total for value in raw]
    scaled = [max(1, math.floor(value)) for value in exact]
    difference = total_target - sum(scaled)
    order = sorted(
        range(len(outline)),
        key=lambda index: exact[index] - math.floor(exact[index]),
        reverse=difference > 0,
    )
    step = 1 if difference > 0 else -1
    for offset in range(abs(difference)):
        index = order[offset % len(order)]
        if step < 0 and scaled[index] <= 1:
            continue
        scaled[index] += step
    normalized = copy.deepcopy(outline)
    for item, target in zip(normalized, scaled):
        item["word_count"] = target
    return normalized


def rebalance_remaining_outline_word_targets(
        outline: List[Dict[str, Any]], chapters: List[Dict[str, Any]],
        total_target: int) -> Dict[int, Tuple[int, int]]:
    """Return accepted length drift to the remaining chapter budget.

    Per-chapter tolerance should not turn into a systematic whole-book
    overshoot.  Once a chapter is accepted, redistribute the exact remaining
    book budget over unwritten chapters while preserving their relative pacing
    weights.  The operation is deterministic and safe to repeat after loading
    a checkpoint.
    """
    completed_ids = {
        item.get("id") for item in chapters
        if isinstance(item.get("id"), int)
    }
    completed_words = sum(
        count_words(str(item.get("content", ""))) for item in chapters
    )
    remaining = [
        item for item in outline if item.get("id") not in completed_ids
    ]
    remaining_budget = total_target - completed_words
    if not remaining or remaining_budget < len(remaining):
        return {}
    normalized = normalize_outline_word_targets(remaining, remaining_budget)
    changes: Dict[int, Tuple[int, int]] = {}
    for item, replacement in zip(remaining, normalized):
        old_target = max(1, int(item.get("word_count") or 1))
        new_target = max(1, int(replacement.get("word_count") or 1))
        if old_target != new_target:
            changes[int(item["id"])] = (old_target, new_target)
            item["word_count"] = new_target
    return changes


def detect_language(text: str) -> str:
    """Classify text as Chinese when CJK makes up at least 30% of letter characters."""
    if not text:
        return "en"
    cjk = len(CJK_RANGE.findall(text))
    letters = sum(1 for c in text if c.isalpha() or CJK_RANGE.match(c))
    if letters == 0:
        return "en"
    return "zh" if cjk / letters >= 0.30 else "en"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[。！？!?\.\n])\s+|(?<=[。！？!?])(?=[^\s])', text)
    return [p.strip() for p in parts if p and p.strip()]


def fuzzy_locate(content: str, old_text: str,
                 anchor_chars: int = 20) -> Optional[Tuple[int, int]]:
    """Locate text to replace; fall back to whitespace-tolerant head/tail anchor matching."""
    if not old_text:
        return None
    idx = content.find(old_text)
    if idx != -1:
        return idx, idx + len(old_text)
    if len(old_text) <= 2 * anchor_chars + 4:
        return None

    def ws_pattern(s: str) -> str:
        parts = re.split(r'\s+', s)
        return r'\s+'.join(re.escape(p) for p in parts if p)

    head_pat = ws_pattern(old_text[:anchor_chars])
    tail_pat = ws_pattern(old_text[-anchor_chars:])
    if not head_pat or not tail_pat:
        return None
    span_limit = max(len(old_text) * 2, len(old_text) + 200)
    pattern = head_pat + r'.{0,' + str(span_limit) + r'}?' + tail_pat
    m = re.search(pattern, content, flags=re.DOTALL)
    if m:
        return m.start(), m.end()
    return None


class ChapterSearchIndex:
    """Incremental full-text index over completed chapters (SQLite FTS5 + jieba)."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE chapter_passages USING fts5(
                    chapter_id UNINDEXED,
                    sentence_id UNINDEXED,
                    before_text UNINDEXED,
                    sentence_text UNINDEXED,
                    after_text UNINDEXED,
                    tokens,
                    tokenize='unicode61'
                )
                """
            )
        except sqlite3.OperationalError as exc:
            self._conn.close()
            raise RuntimeError(
                "SQLite FTS5 is required for the search tool"
            ) from exc
        self._indexed_ids: set = set()
        self._dirty_ids: set = set()

    @staticmethod
    def _tokenize(text: str, *, query: bool = False) -> List[str]:
        """Tokenize with jieba search mode plus CJK unigrams/bigrams for invented names."""
        normalized = unicodedata.normalize("NFKC", text).lower()
        tokens: List[str] = []
        seen = set()

        def add_token(token: str) -> None:
            if token and token not in seen:
                tokens.append(token)
                seen.add(token)

        for raw_token in jieba.lcut_for_search(normalized):
            token = raw_token.strip()
            if not token or not any(char.isalnum() for char in token):
                continue
            add_token(token)

        # jieba's general dictionary may split "莉拉站在" as "莉拉站/在", so a query for
        # "莉拉" would miss. Add unigrams and bigrams for runs of CJK characters, still
        # ranked by FTS5/BM25; multi-token queries drop single-character noise.
        cjk_run: List[str] = []

        def flush_cjk_run() -> None:
            for char in cjk_run:
                add_token(char)
            for index in range(len(cjk_run) - 1):
                add_token("".join(cjk_run[index:index + 2]))
            cjk_run.clear()

        for char in normalized:
            if _CJK_CHAR.fullmatch(char):
                cjk_run.append(char)
            else:
                flush_cjk_run()
        flush_cjk_run()

        if not query:
            return tokens

        # jieba splits off the English possessive 's' and emits some single-character
        # Chinese tokens. Multi-token queries keep tokens of length >= 2; queries made
        # only of single characters still search.
        preferred = [
            token for token in tokens
            if len(token) >= 2 or token.isdigit()
        ]
        return preferred or tokens

    @classmethod
    def _chapter_rows(cls, chapter: Dict[str, Any]) -> List[Tuple[Any, ...]]:
        sentences = split_sentences(chapter.get("content", ""))
        rows: List[Tuple[Any, ...]] = []
        for index, sentence in enumerate(sentences):
            before = sentences[index - 1] if index > 0 else ""
            after = sentences[index + 1] if index + 1 < len(sentences) else ""
            # Windows include neighboring sentences; center-sentence tokens are repeated
            # three times so the sentence holding the fact ranks higher under BM25.
            center_tokens = cls._tokenize(sentence)
            context_tokens = cls._tokenize(before) + cls._tokenize(after)
            tokens = " ".join(center_tokens * 3 + context_tokens)
            if tokens:
                rows.append((
                    chapter["id"], index, before, sentence, after, tokens,
                ))
        return rows

    def reset(self) -> None:
        """Clear the in-memory index; it is rebuilt lazily from checkpointed chapters."""
        with self._conn:
            self._conn.execute("DELETE FROM chapter_passages")
        self._indexed_ids.clear()
        self._dirty_ids.clear()

    def mark_dirty(self, chapter_id: int) -> None:
        """Mark a chapter edited by correct for reindexing before the next search."""
        self._dirty_ids.add(chapter_id)

    def _sync(self, chapters: List[Dict[str, Any]]) -> None:
        """Insert new chapters and reindex chapters modified by correct."""
        chapter_by_id = {chapter["id"]: chapter for chapter in chapters}
        current_ids = set(chapter_by_id)
        stale_ids = self._indexed_ids - current_ids
        changed_ids = {
            chapter_id for chapter_id in current_ids
            if chapter_id not in self._indexed_ids
            or chapter_id in self._dirty_ids
        }
        if not stale_ids and not changed_ids:
            return

        with self._conn:
            for chapter_id in stale_ids | changed_ids:
                self._conn.execute(
                    "DELETE FROM chapter_passages WHERE chapter_id = ?",
                    (chapter_id,),
                )
            for chapter_id in changed_ids:
                rows = self._chapter_rows(chapter_by_id[chapter_id])
                if rows:
                    self._conn.executemany(
                        """
                        INSERT INTO chapter_passages(
                            chapter_id, sentence_id, before_text,
                            sentence_text, after_text, tokens
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )

        self._indexed_ids = current_ids
        self._dirty_ids.difference_update(changed_ids | stale_ids)

    @staticmethod
    def _fts_query(tokens: List[str]) -> str:
        """Join tokens into a safe FTS5 OR query; BM25 ranks relevance."""
        quoted = []
        for token in tokens:
            escaped = token.replace('"', '""')
            quoted.append(f'"{escaped}"')
        return " OR ".join(quoted)

    def search(self, chapters: List[Dict[str, Any]], query: str,
               max_hits: int = SEARCH_MAX_HITS) -> List[Dict[str, Any]]:
        """Return the most relevant sentence windows in the search tool's ``hits`` format."""
        if not query or not chapters:
            return []
        self._sync(chapters)
        query_tokens = self._tokenize(query, query=True)
        if not query_tokens:
            return []

        rows = self._conn.execute(
            """
            SELECT chapter_id, before_text, sentence_text, after_text,
                   bm25(chapter_passages) AS relevance
            FROM chapter_passages
            WHERE tokens MATCH ?
            ORDER BY relevance, chapter_id DESC, sentence_id
            LIMIT ?
            """,
            (self._fts_query(query_tokens), max_hits),
        ).fetchall()
        return [
            {
                "chapter_id": int(chapter_id),
                "sentence": sentence,
                "before": before,
                "after": after,
            }
            for chapter_id, before, sentence, after, relevance in rows
        ]

    def close(self) -> None:
        self._conn.close()


def search_chapters(chapters: List[Dict[str, Any]], query: str,
                    max_hits: int = SEARCH_MAX_HITS) -> List[Dict[str, Any]]:
    """Standalone search helper; the agent uses a reusable incremental index."""
    index = ChapterSearchIndex()
    try:
        return index.search(chapters, query, max_hits)
    finally:
        index.close()


# --- XML tool-call fallback --------------------------------------------------

def _coerce_arg(name: str, value: str) -> Any:
    """Coerce an XML argument to an int or JSON object when possible."""
    v = value.strip()
    if name in ("chapter_id", "state_id"):
        try:
            return int(v)
        except ValueError:
            return v
    if v and v[0] in "{[":
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
    return v


def parse_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Convert XML-style tool calls in model output to the OpenAI tool_calls shape."""
    if not text or "<tool_call>" not in text:
        return []
    calls: List[Dict[str, Any]] = []
    block_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    fn_re = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", re.DOTALL)
    fn_open_re = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*)$", re.DOTALL)
    param_re = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", re.DOTALL)
    for m in block_re.finditer(text):
        body = m.group(1)
        fm = fn_re.search(body) or fn_open_re.search(body)
        if not fm:
            continue
        fname, inner = fm.group(1), fm.group(2)
        args: Dict[str, Any] = {}
        for pm in param_re.finditer(inner):
            args[pm.group(1)] = _coerce_arg(pm.group(1), pm.group(2))
        calls.append({
            "id": f"call_xml_{len(calls)}",
            "type": "function",
            "function": {"name": fname, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return calls


class _SynthToolCall:
    def __init__(self, d: Dict[str, Any]):
        self.id = d["id"]
        self.type = "function"

        class _Fn:
            pass
        self.function = _Fn()
        self.function.name = d["function"]["name"]
        self.function.arguments = d["function"]["arguments"]


# =============================================================================
# 4. Checkpoints, tool protocol, and structured narrative state
# =============================================================================

# --- 4.1 Chapter-level checkpoints -------------------------------------------

CHECKPOINT_VERSION = 3


def checkpoint_path(ckpt_dir: str, story_id: Any) -> str:
    return os.path.join(ckpt_dir, f"story_{story_id}.json")


def checkpoint_fingerprint(
        prompt: str, word_count: int,
        state_update_mode: str = "batch",
        narrative_ops_guidance_placement: str =
        DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT,
) -> str:
    """Key a checkpoint by version, target length, and prompt so stories never mix."""
    h = hashlib.sha256()
    payload = f"{CHECKPOINT_VERSION}\x00{word_count}\x00{prompt}"
    # batch mode keeps the base fingerprint; other modes are isolated from it.
    if state_update_mode != "batch":
        payload += f"\x00state_update={state_update_mode}"
    # The "both" placement keeps the unsuffixed fingerprint; single-surface
    # placements are isolated. This does not depend on the current CLI default.
    if (state_update_mode == "narrative_ops"
            and narrative_ops_guidance_placement != "both"):
        payload += (
            "\x00narrative_ops_guidance="
            f"{narrative_ops_guidance_placement}"
        )
    h.update(payload.encode("utf-8"))
    return h.hexdigest()[:16]


def save_checkpoint(ckpt_dir: str, story_id: Any, data: Dict[str, Any],
                    logger) -> None:
    """Write a checkpoint atomically; failures are logged and do not stop generation."""
    path = checkpoint_path(ckpt_dir, story_id)
    tmp = f"{path}.tmp"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)  # Atomic replace: a crash mid-write leaves the previous complete file.
    except Exception as e:
        logger.warning(f"  failed to save checkpoint for story {story_id}: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_checkpoint(ckpt_dir: str, story_id: Any, fingerprint: str,
                    logger) -> Optional[Dict[str, Any]]:
    """Load a usable checkpoint; return None if missing, stale, or corrupt."""
    path = checkpoint_path(ckpt_dir, story_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  ignoring unreadable checkpoint {path}: {e}")
        return None
    if data.get("fingerprint") != fingerprint:
        logger.warning(
            f"  ignoring stale checkpoint {path}: written for a different "
            "prompt/word_count"
        )
        return None
    if not data.get("outline"):
        return None
    return data


def discard_checkpoint(ckpt_dir: str, story_id: Any) -> None:
    try:
        os.remove(checkpoint_path(ckpt_dir, story_id))
    except OSError:
        pass


# --- 4.2 Tool schemas exposed to the model -----------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a written chapter in full. Prior chapters are always "
                "available; the current chapter becomes available after a "
                "successful write. Counts against "
                f"the {MAX_READ}-call read budget, but only "
                "when the call succeeds — rejected calls (bad chapter_id, "
                "unknown chapter) are free."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_id": {"type": "integer"},
                },
                "required": ["chapter_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search prior chapters using one or more keywords or a short "
                "phrase. The query is segmented into terms, sentence windows "
                "matching any term are ranked by BM25 relevance, and every "
                f"valid executed search counts against the {MAX_SEARCH}-call "
                f"search budget; at most {SEARCH_MAX_HITS} ranked windows "
                "are returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct",
            "description": (
                "Fix a factual or continuity error in an already-written "
                "chapter (current or prior). Style polishing is not a "
                f"valid use. At most {MAX_CORRECT_PER_CHAPTER} correct calls "
                "per chapter. Correcting the CURRENT chapter is final: once "
                "you correct it you can no longer rewrite it with write, so "
                "settle the chapter's length first and correct last."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_id": {"type": "integer"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["chapter_id", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Write the full text of the current chapter. Only one "
                "successful write is allowed per chapter. A draft rejected "
                "for length does not consume that opportunity, so rewrite it "
                "until the tool returns ok=true. After that, use correct for "
                "any changes; another write will be rejected. The tool returns "
                f"ok=true only when the submitted chapter is within "
                f"±{WORD_COUNT_TOLERANCE_PERCENT}% of its target length."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["chapter_id", "title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update",
            "description": (
                "Atomically batch state changes using adds, edits, and "
                "deletes; use an empty array when a group is not needed. "
                "Edits and deletes require an existing state_id from the "
                "current state; name is used only for character_states and "
                "must otherwise be null."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "adds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "state_name": {
                                    "type": "string",
                                    "enum": [
                                        "character_states",
                                        "past_events",
                                        "future_requirements",
                                    ],
                                },
                                "name": {"type": ["string", "null"]},
                                "description": {"type": "string"},
                            },
                            "required": [
                                "state_name", "name", "description",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "state_name": {
                                    "type": "string",
                                    "enum": [
                                        "character_states",
                                        "past_events",
                                        "future_requirements",
                                    ],
                                },
                                "state_id": {"type": "integer"},
                                "name": {"type": ["string", "null"]},
                                "description": {"type": "string"},
                            },
                            "required": [
                                "state_name", "state_id", "name",
                                "description",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "deletes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "state_name": {
                                    "type": "string",
                                    "enum": [
                                        "character_states",
                                        "past_events",
                                        "future_requirements",
                                    ],
                                },
                                "state_id": {"type": "integer"},
                            },
                            "required": ["state_name", "state_id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["adds", "edits", "deletes"],
                "additionalProperties": False,
            },
        },
    },
]


# Layout-only batch variant: ``state_name`` is the first top-level argument
# and one call touches exactly one state collection. The operation groups
# stay atomic within the call and appear in adds/deletes/edits order.
STATE_FIRST_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update",
        "description": (
            "Atomically update exactly one narrative-state collection. Select "
            "state_name first, then provide adds, deletes, and edits; use an "
            "empty array when a group is not needed. Call update separately for "
            "each state_name changed by the chapter. Edits and deletes require "
            "an existing state_id; name is used only for character_states and "
            "must otherwise be null."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "state_name": {
                    "type": "string",
                    "enum": [
                        "character_states",
                        "past_events",
                        "future_requirements",
                    ],
                },
                "adds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "deletes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "state_id": {"type": "integer"},
                        },
                        "required": ["state_id"],
                        "additionalProperties": False,
                    },
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "state_id": {"type": "integer"},
                            "name": {"type": ["string", "null"]},
                            "description": {"type": "string"},
                        },
                        "required": ["state_id", "name", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["state_name", "adds", "deletes", "edits"],
            "additionalProperties": False,
        },
    },
}


# Layout-only inverse of the batch schema above.  One
# atomic tool call can still touch every state collection; only the nesting
# changes from operation -> state_name to state_name -> operation groups.
STATE_GROUPED_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update",
        "description": (
            "Atomically batch every narrative-state change in one update call. "
            "In state_updates, group changes by state_name first, then provide "
            "adds, deletes, and edits for that collection; use an empty array "
            "when an operation group is not needed. Include every changed "
            "collection in this single call. Edits and deletes require an "
            "existing state_id; name is used only for character_states and "
            "must otherwise be null."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "state_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "state_name": {
                                "type": "string",
                                "enum": [
                                    "character_states",
                                    "past_events",
                                    "future_requirements",
                                ],
                            },
                            "adds": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": ["string", "null"]},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["name", "description"],
                                    "additionalProperties": False,
                                },
                            },
                            "deletes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "state_id": {"type": "integer"},
                                    },
                                    "required": ["state_id"],
                                    "additionalProperties": False,
                                },
                            },
                            "edits": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "state_id": {"type": "integer"},
                                        "name": {"type": ["string", "null"]},
                                        "description": {"type": "string"},
                                    },
                                    "required": [
                                        "state_id", "name", "description",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "state_name", "adds", "deletes", "edits",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["state_updates"],
            "additionalProperties": False,
        },
    },
}


# A single-call semantic layout that removes the generic state/action matrix.
# Character entries are deterministic upserts; past events and future
# requirements use stable semantic keys so resolution never depends on an
# implementation-generated integer id. ``batch`` remains the core CLI
# default; nstagent.py fixes this mode.
NARRATIVE_OPS_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update",
        "description": (
            "Atomically apply all narrative-state changes from the completed "
            "chapter. Upsert each changed character's complete current state, "
            "record a completed event only when it is absent from the frozen "
            "outline and may affect later plot, add concrete unresolved future "
            "requirements, "
            "and resolve requirements by stable key after payoff. Include all "
            "four arrays and use [] when an operation has no items. Every field "
            "must be a native JSON array, never a quoted or serialized array "
            "string. Do not append another snapshot for an existing character."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "upsert_character_state": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "add_past_event": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["key", "description"],
                        "additionalProperties": False,
                    },
                },
                "add_future_requirement": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["key", "description"],
                        "additionalProperties": False,
                    },
                },
                "resolve_future_requirement": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "upsert_character_state",
                "add_past_event",
                "add_future_requirement",
                "resolve_future_requirement",
            ],
            "additionalProperties": False,
        },
    },
}

NARRATIVE_OPS_UPDATE_DESCRIPTIONS = {
    "minimal_zh": (
        "原子提交已完成章节造成的全部叙事状态变化。请遵循参数结构，并在一次调用中"
        "填写全部四个数组。"
    ),
    "minimal_en": (
        "Atomically submit every narrative-state change caused by the completed "
        "chapter. Follow the parameter schema and include all four arrays in one call."
    ),
    "detailed_zh": (
        "原子提交已完成章节造成的全部叙事状态变化。upsert_character_state 使用 "
        "{name, description} 写入角色完整的当前位置、目标、关系、知识、持有物和身心"
        "状况，同名角色自动替换；add_past_event 只添加大纲中没有明示且可能影响后续"
        "情节的本章事件，并为它设置稳定的 snake_case key；"
        "add_future_requirement 只添加以后必须兑现的具体事项，也使用稳定 key；"
        "resolve_future_requirement 传入已兑现事项的 key。四个字段都必须是原生 "
        "JSON 数组，无变化时传 []。"
    ),
    "detailed_en": (
        "Atomically submit every narrative-state change caused by the completed "
        "chapter. In upsert_character_state, use {name, description} to provide "
        "each character's complete current location, goal, relationships, knowledge, "
        "possessions, and physical or emotional condition; an existing name is "
        "replaced automatically. In add_past_event, add a chapter event only when it "
        "is not explicit in the outline and may affect later plot, and give it a "
        "stable snake_case key. In add_future_requirement, add only a concrete matter "
        "that must be fulfilled later, also with a stable key. Pass fulfilled keys to "
        "resolve_future_requirement. All four fields must be native JSON arrays; use "
        "[] when a field has no changes."
    ),
}


def narrative_ops_update_tool(
        lang: str,
        guidance_placement: str = DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT,
) -> Dict[str, Any]:
    """Return a localized schema with detailed guidance on the chosen surface."""
    if guidance_placement not in NARRATIVE_OPS_GUIDANCE_PLACEMENTS:
        raise ValueError(
            f"unknown narrative_ops_guidance_placement {guidance_placement!r}"
        )
    tool = copy.deepcopy(NARRATIVE_OPS_UPDATE_TOOL)
    detail = "minimal" if guidance_placement == "prompt" else "detailed"
    language = "zh" if lang == "zh" else "en"
    tool["function"]["description"] = (
        NARRATIVE_OPS_UPDATE_DESCRIPTIONS[f"{detail}_{language}"]
    )
    return tool


# Generic CRUD suits large models but asks small models to pick the state
# type, action, id, and nullable fields at once. The tools below encode type
# and action in the tool name, handle one change per call, and use no arrays
# or nested objects.
SIMPLE_STATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_character",
            "description": (
                "Set one character's complete current state. If the name "
                "already exists (case-insensitive), its description is "
                "replaced; otherwise the character is created. Call once "
                "per character whose current situation materially changed."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_past_event",
            "description": (
                "Append one pivotal past event that actually happened in this "
                "chapter. Call separately for each event; keep the chapter "
                "total to 1-4 events."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_requirement",
            "description": (
                "Add one unresolved setup, promise, mystery, or future "
                "obligation introduced by this chapter. Do not add events "
                "that are already complete."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_requirement",
            "description": (
                "Remove one existing future requirement after it has been "
                "fulfilled or made irrelevant. Use its exact id from the "
                "current narrative state."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"state_id": {"type": "integer"}},
                "required": ["state_id"],
                "additionalProperties": False,
            },
        },
    },
]

SIMPLE_STATE_TOOL_NAMES = {
    tool["function"]["name"] for tool in SIMPLE_STATE_TOOLS
}


# ``simple`` splits generic CRUD into four flat tools but still has the model
# append events scene by scene and resolve requirements by error-prone numeric
# ids. ``semantic`` keeps equally flat arguments but builds event compression
# and reference resolution into the tools themselves.
SEMANTIC_STATE_TOOLS = [
    copy.deepcopy(SIMPLE_STATE_TOOLS[0]),
    {
        "type": "function",
        "function": {
            "name": "record_chapter_summary",
            "description": (
                "Store the single causal summary for the chapter that was just "
                "written. Include the consequential actions, discoveries, and "
                "state changes future chapters need, in at most 60 English words "
                "or 100 Han characters. "
                "Call exactly once after write succeeds; a repeated call replaces "
                "this chapter's prior summary instead of appending another event."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
    copy.deepcopy(SIMPLE_STATE_TOOLS[2]),
    {
        "type": "function",
        "function": {
            "name": "resolve_requirement_by_text",
            "description": (
                "Resolve one open future requirement after payoff. Copy or briefly "
                "paraphrase its description from Current Narrative State; no numeric "
                "id is needed. Repeating an already resolved request is harmless."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": False,
            },
        },
    },
]

SEMANTIC_STATE_TOOLS[0]["function"]["description"] = (
    "Set one named recurring character's complete current state, including "
    "still-valid goals, relationships, knowledge, possessions, and physical or "
    "emotional condition, in at most 90 English words or 140 Han characters "
    "with no biography or scene recap. "
    "Call for every recurring character newly introduced "
    "or materially changed; the name is created or case-insensitively replaced."
)
SEMANTIC_STATE_TOOLS[2]["function"]["description"] = (
    "Add one concrete cross-chapter obligation, unanswered question, or promised "
    "payoff newly introduced in this chapter (zero or at most one call). Its text "
    "must name a specific later action such as reveal, answer, return, confront, "
    "fulfill, resolve, discover, or recover, in at most 50 English words or "
    "80 Han characters. Do not "
    "add ongoing threats, background facts, completed discoveries, character goals "
    "already in a snapshot, or an open loop already represented in current state."
)

SEMANTIC_STATE_TOOL_NAMES = {
    tool["function"]["name"] for tool in SEMANTIC_STATE_TOOLS
}
ITEM_STATE_TOOL_NAMES = SIMPLE_STATE_TOOL_NAMES | SEMANTIC_STATE_TOOL_NAMES
CONTENT_TOOL_NAMES = {"read", "search", "correct", "write"}


def chapter_tools(chapter_target_words: int, lang: str,
                  length_control_mode: str,
                  state_update_mode: str = DEFAULT_STATE_UPDATE_MODE,
                  narrative_ops_guidance_placement: str =
                  DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT,
                  ) -> List[Dict[str, Any]]:
    """Build this chapter's tool schemas for the mode without mutating ``TOOLS``.

    ``tool_target`` writes the chapter's exact target and allowed range into
    the write description; other modes use the standard descriptions so only
    one variable changes.
    """
    if state_update_mode not in STATE_UPDATE_MODES:
        raise ValueError(f"unknown state_update_mode {state_update_mode!r}")
    if narrative_ops_guidance_placement not in \
            NARRATIVE_OPS_GUIDANCE_PLACEMENTS:
        raise ValueError(
            "unknown narrative_ops_guidance_placement "
            f"{narrative_ops_guidance_placement!r}"
        )
    if state_update_mode == "batch":
        base_tools = TOOLS
    elif state_update_mode == "narrative_ops":
        base_tools = [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") != "update"
        ] + [narrative_ops_update_tool(
            lang, narrative_ops_guidance_placement
        )]
    elif state_update_mode == "state_grouped":
        base_tools = [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") != "update"
        ] + [STATE_GROUPED_UPDATE_TOOL]
    elif state_update_mode == "state_first":
        base_tools = [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") != "update"
        ] + [STATE_FIRST_UPDATE_TOOL]
    elif state_update_mode == "simple":
        base_tools = [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") != "update"
        ] + SIMPLE_STATE_TOOLS
    else:
        base_tools = [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") != "update"
        ] + SEMANTIC_STATE_TOOLS

    if length_control_mode != "tool_target":
        return base_tools

    min_words = math.ceil(
        chapter_target_words * (1 - WORD_COUNT_TOLERANCE)
    )
    max_words = math.floor(
        chapter_target_words * (1 + WORD_COUNT_TOLERANCE)
    )
    tools = copy.deepcopy(base_tools)
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name") != "write":
            continue
        if lang == "zh":
            target_note = (
                f" 当前章节的精确目标为 {chapter_target_words} 字；"
                f"完整正文为 {min_words}–{max_words} 字时才会返回 ok=true。"
            )
        else:
            target_note = (
                f" For the current chapter, the exact target is "
                f"{chapter_target_words} words; write returns ok=true only "
                f"when the complete draft contains {min_words}-{max_words} "
                "words."
            )
        function["description"] += target_note
        break
    return tools


def chapter_tools_for_phase(tools: List[Dict[str, Any]],
                            state_update_mode: str,
                            write_succeeded: bool,
                            semantic_summary_done: bool = False,
                            ) -> List[Dict[str, Any]]:
    """In semantic mode, expose tools by phase to ease tool choice for small models."""
    if state_update_mode != "semantic":
        return tools
    allowed = (
        SEMANTIC_STATE_TOOL_NAMES | {"correct"}
        if write_succeeded else CONTENT_TOOL_NAMES
    )
    if semantic_summary_done:
        allowed = allowed - {"record_chapter_summary"}
    return [
        tool for tool in tools
        if tool.get("function", {}).get("name") in allowed
    ]


# --- 4.3 Structured narrative state ------------------------------------------

def _maybe_json(value: Any) -> Any:
    """Recover the structure when the model passes an object as a JSON string."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s[0] not in "{[":
        return value
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return value


def _unwrap_text(content: Any) -> str:
    """Extract plain text from nested description/text/content or a JSON string."""
    content = _maybe_json(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("description", "text", "content"):
            if key in content:
                return _unwrap_text(content[key])
    return str(content).strip()


class StoryState:
    """Character, past-event, and future-requirement buckets with auto-increment ids."""

    def __init__(self):
        self.character_states: List[Dict[str, Any]] = []
        self.past_events: List[Dict[str, Any]] = []
        self.future_requirements: List[Dict[str, Any]] = []
        self._next_ids = {
            "character_states": 1,
            "past_events": 1,
            "future_requirements": 1,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_states": self.character_states,
            "past_events": self.past_events,
            "future_requirements": self.future_requirements,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StoryState":
        """Restore state from a checkpoint and rebuild id counters from the max id."""
        s = cls()
        for bucket in s._next_ids:
            raw_entries = d.get(bucket)
            # Fall back to ``occurred_events`` when ``past_events`` is absent; every
            # newly written checkpoint and result uses ``past_events``.
            if bucket == "past_events" and raw_entries is None:
                raw_entries = d.get("occurred_events")
            entries = [e for e in (raw_entries or []) if isinstance(e, dict)]
            setattr(s, bucket, entries)
            ids = [e["id"] for e in entries if isinstance(e.get("id"), int)]
            s._next_ids[bucket] = (max(ids) + 1) if ids else 1
        return s

    def _bucket(self, state_name: str) -> List[Dict[str, Any]]:
        return getattr(self, state_name)

    def apply(self, state_name: str, action: str,
              state_id: Optional[int], content: Any) -> Tuple[bool, str]:
        if state_name not in self._next_ids:
            return False, f"unknown state_name {state_name!r}"
        if action not in ("add", "edit", "delete"):
            return False, f"unknown action {action!r}"
        bucket = self._bucket(state_name)

        content = _maybe_json(content)
        if isinstance(content, str) and content.strip():
            content = {"description": content.strip()}

        if action == "add":
            if not isinstance(content, dict):
                return False, "add requires content object"
            new_id = self._next_ids[state_name]
            self._next_ids[state_name] += 1
            entry: Dict[str, Any] = {"id": new_id}
            if state_name == "character_states":
                entry["name"] = str(content.get("name", "")).strip()
                entry["description"] = _unwrap_text(content.get("description"))
                if not entry["name"] and not entry["description"]:
                    return False, "character_states add needs name or description"
            else:
                entry["description"] = _unwrap_text(content.get("description") or content)
                if not entry["description"]:
                    return False, f"{state_name} add needs description"
                key = content.get("key")
                if key is not None:
                    entry["key"] = str(key).strip()
            bucket.append(entry)
            return True, f"added id={new_id}"

        if state_id is None:
            return False, f"{action} requires state_id"
        idx = next((i for i, e in enumerate(bucket) if e.get("id") == state_id), -1)
        if idx == -1:
            return False, f"id {state_id} not found in {state_name}"

        if action == "edit":
            if not isinstance(content, dict):
                return False, "edit requires content object"
            entry = bucket[idx]
            if state_name == "character_states":
                if "name" in content:
                    entry["name"] = str(content["name"]).strip()
                if "description" in content:
                    entry["description"] = _unwrap_text(content["description"])
            else:
                desc = _unwrap_text(content.get("description") or content)
                if desc:
                    entry["description"] = desc
            return True, f"edited id={state_id}"

        bucket.pop(idx)
        return True, f"deleted id={state_id}"


# =============================================================================
# 5. Multi-stage outline generation
# =============================================================================

def recommend_chapter_count(word_count: int) -> int:
    if word_count <= 0:
        return 0
    anchors = CHAPTER_ANCHORS
    if word_count >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        w0, c0 = anchors[i]
        w1, c1 = anchors[i + 1]
        if w0 <= word_count <= w1:
            if w1 == w0:
                return c0
            ratio = (word_count - w0) / (w1 - w0)
            return max(1, math.ceil(c0 + ratio * (c1 - c0)))
    return anchors[-1][1]


def stage_count_for_words(word_count: int) -> int:
    if word_count < 10000:
        return 1
    if word_count < 20000:
        return 2
    if word_count < 100000:
        return 3
    return 4


class IncompleteResponseError(ValueError):
    """The API response did not finish normally (e.g. length or content_filter)."""


def require_finish_reason(choice: Any, allowed: Tuple[str, ...]) -> str:
    """Reject responses that were truncated, filtered, or otherwise unfinished."""
    reason = getattr(choice, "finish_reason", None)
    if reason not in allowed:
        raise IncompleteResponseError(
            f"unexpected finish_reason={reason!r}; expected one of {allowed}"
        )
    return reason


async def llm_call_text(client: AsyncOpenAI, model: str,
                        messages: List[Dict], max_tokens: int,
                        temperature: float,
                        disable_thinking: bool = False) -> str:
    for retry in range(3):
        try:
            kwargs: Dict[str, Any] = dict(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            if disable_thinking:
                # Qwen3.5's official template supports this request-scoped
                # switch. It is opt-in; by default the model's standard
                # thinking behavior is used.
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            require_finish_reason(choice, ("stop",))
            return extract_content(choice.message)
        except IncompleteResponseError:
            # finish_reason failures are retried by the outer outline loop,
            # so retry counts do not multiply.
            raise
        except Exception:
            if retry == 2:
                raise
            await asyncio.sleep(5 * (retry + 1))
    raise RuntimeError("text LLM call exhausted retries")


class OutlineAgent:
    """Generate a hierarchical outline for the target length and validate it."""

    def __init__(self, client: AsyncOpenAI, model: str,
                 max_tokens: int, temperature: float,
                 word_count: int, logger: logging.Logger,
                 lang: str = "en",
                 disable_thinking: bool = False):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.word_count = word_count
        self.logger = logger
        self.lang = lang
        self.disable_thinking = disable_thinking
        self.system_msg = {
            "role": "system",
            "content": get_system_prompt("outline", lang),
        }

    async def _chat(self, user_msg: str) -> str:
        messages = [self.system_msg, {"role": "user", "content": user_msg}]
        for attempt in range(MAX_JSON_RETRIES):
            try:
                return await llm_call_text(
                    self.client, self.model, messages,
                    self.max_tokens, self.temperature,
                    self.disable_thinking,
                )
            except IncompleteResponseError as e:
                self.logger.warning(
                    f"  Outline response incomplete (attempt {attempt+1}): {e}"
                )
                if attempt == MAX_JSON_RETRIES - 1:
                    raise
        raise RuntimeError("outline generation exhausted retries")

    async def _chat_json(self, user_msg: str, expect_list: bool = False,
                         validator: Optional[Callable[[Any], None]] = None) -> Any:
        """Request JSON; regenerate on parse or validation failure."""
        messages = [self.system_msg, {"role": "user", "content": user_msg}]
        for attempt in range(MAX_JSON_RETRIES):
            try:
                response = await llm_call_text(
                    self.client, self.model, messages,
                    self.max_tokens, self.temperature,
                    self.disable_thinking,
                )
                data = extract_json(response)
                if expect_list:
                    data = self._coerce_json_list(data)
                if validator is not None:
                    validator(data)
                return data
            except (ValueError, json.JSONDecodeError) as e:
                self.logger.warning(
                    f"  Outline response validation failed "
                    f"(attempt {attempt+1}): {e}"
                )
                if attempt == MAX_JSON_RETRIES - 1:
                    raise
                if "response" in locals() and isinstance(response, str):
                    messages.extend([
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                "The JSON outline was rejected by deterministic "
                                f"validation: {e}. Regenerate the complete JSON "
                                "array, correcting that exact problem while "
                                "preserving all original Story prompt constraints. "
                                "Output JSON only."
                            ),
                        },
                    ])
        raise RuntimeError("JSON outline generation exhausted retries")

    @staticmethod
    def _coerce_json_list(data: Any) -> List[Any]:
        """Unwrap common JSON wrappers; lift a single chapter object to a list for validation."""
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON array, got {type(data)}")

        for key in ("chapters", "outline", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        if {"id", "name", "description"} <= set(data):
            return [data]
        raise ValueError(
            f"Expected JSON array, got dict with keys {list(data.keys())[:5]}"
        )

    async def gen_premise(self, prompt: str) -> str:
        self.logger.info("Stage: premise (~300 words)")
        return await self._chat(outline_premise_prompt(self.lang, prompt, self.word_count))

    async def gen_detail(self, prompt: str, premise: str, target_words: int) -> str:
        self.logger.info(f"Stage: detail (~{target_words} words)")
        return await self._chat(outline_detail_prompt(self.lang, prompt, premise, target_words))

    async def gen_acts(self, prompt: str, premise: str, detail: str) -> str:
        self.logger.info("Stage: acts/volumes")
        return await self._chat(outline_acts_prompt(self.lang, prompt, premise, detail, self.word_count))

    def _validate_chapter_outline(
            self, chapters: Any, story_contract: str = "") -> None:
        """Validate the chapter outline structure and the exactly checkable length constraints."""
        if not isinstance(chapters, list) or not chapters:
            raise ValueError("chapter outline must be a non-empty JSON array")
        if self.word_count >= LONG_FORM_MIN_WORDS and len(chapters) == 1:
            raise ValueError(
                f"a long-form target ({self.word_count} words) cannot use a "
                "single-chapter outline"
            )

        planned_words = 0
        for i, chapter in enumerate(chapters):
            if not isinstance(chapter, dict):
                raise ValueError(f"chapter {i} must be a JSON object")

            description = chapter.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"chapter {i} must have a non-empty description")

            chapter_words = chapter.get("word_count")
            if (isinstance(chapter_words, bool)
                    or not isinstance(chapter_words, int)
                    or chapter_words <= 0):
                raise ValueError(
                    f"chapter {i} word_count must be a positive integer"
                )
            planned_words += chapter_words

        min_words = self.word_count * (1 - WORD_COUNT_TOLERANCE)
        max_words = self.word_count * (1 + WORD_COUNT_TOLERANCE)
        if not min_words <= planned_words <= max_words:
            raise ValueError(
                f"outline plans {planned_words} words; target {self.word_count} "
                f"allows {round(min_words)}-{round(max_words)} "
                f"(±{WORD_COUNT_TOLERANCE_PERCENT}%)"
            )

        contract = " ".join((story_contract or "").casefold().split())
        outline_text = json.dumps(chapters, ensure_ascii=False).casefold()
        if ("teenage" in contract or "teenager" in contract) and not re.search(
                r"\b(?:12|twelve)\b", contract):
            if re.search(
                r"\b(?:at|aged?)\s+(?:12|twelve)\b|"
                r"\b(?:12|twelve)[-\s]year[-\s]old\b",
                outline_text,
            ):
                raise ValueError(
                    "outline contradicts the prompt's teenage-age constraint "
                    "by making the protagonist twelve"
                )

    def _normalise_chapters(self, chapters: List[Dict[str, Any]],
                            default_words: int) -> None:
        """Normalize chapter ids and fill in non-essential fields the model omitted."""
        for index, chapter in enumerate(chapters):
            model_id = chapter.get("id")
            if model_id != index:
                self.logger.debug(
                    f"  chapter {index}: reindexed id {model_id!r} -> {index}"
                )
            chapter["id"] = index
            chapter.setdefault("name", f"Chapter {index}")
            chapter.setdefault("description", "")
            chapter.setdefault("word_count", default_words or 1000)

    async def gen_chapters(self, prompt: str, context_blocks: List[str]) -> List[Dict]:
        rec = recommend_chapter_count(self.word_count)
        self.logger.info(f"Stage: chapter outline (recommended {rec} chapters)")
        per_chapter = max(1, round(self.word_count / max(1, rec))) if rec else 0
        context = "\n\n".join(context_blocks) if context_blocks else ""
        user_msg = outline_chapters_prompt(
            self.lang, prompt, context, self.word_count, rec, per_chapter
        )
        chapters = await self._chat_json(
            user_msg,
            expect_list=True,
            validator=lambda value: self._validate_chapter_outline(
                value, prompt
            ),
        )
        # ids drive read/correct bounds checks, so model output cannot be trusted as is.
        self._normalise_chapters(chapters, per_chapter)
        self.logger.info(f"  Outline: {len(chapters)} chapters")
        return chapters

    async def generate(self, prompt: str) -> Dict[str, Any]:
        stages = stage_count_for_words(self.word_count)
        self.logger.info(f"Outline stages: {stages} (word_count={self.word_count})")
        artifacts: Dict[str, Any] = {"stages": stages}
        context_blocks: List[str] = []

        if stages >= 2:
            premise = await self.gen_premise(prompt)
            artifacts["premise"] = premise
            context_blocks.append("Premise:\n" + premise)

        if stages >= 3:
            detail_words = 1000 if stages == 3 else 2500
            detail = await self.gen_detail(prompt, artifacts["premise"], detail_words)
            artifacts["detail"] = detail
            context_blocks.append(f"Detailed synopsis (~{detail_words} words):\n{detail}")

        if stages >= 4:
            acts = await self.gen_acts(prompt, artifacts["premise"], artifacts["detail"])
            artifacts["acts"] = acts
            context_blocks.append("Act/volume outline:\n" + acts)

        chapters = await self.gen_chapters(prompt, context_blocks)
        artifacts["outline"] = chapters
        return artifacts


def outline_pipeline_fingerprint(lang: str) -> str:
    """Hash every input that can change the sampled outline pipeline.

    The cache key must change when a planning prompt or the multi-stage
    assembly logic changes.  Hashing source here avoids relying on a manually
    maintained experiment label that can be forgotten during prompt edits.
    """
    components = [
        f"schema={OUTLINE_CACHE_SCHEMA_VERSION}",
        get_system_prompt("outline", lang),
        inspect.getsource(outline_premise_prompt),
        inspect.getsource(outline_detail_prompt),
        inspect.getsource(outline_acts_prompt),
        inspect.getsource(outline_chapters_prompt),
        inspect.getsource(stage_count_for_words),
        inspect.getsource(recommend_chapter_count),
        inspect.getsource(OutlineAgent.generate),
        inspect.getsource(OutlineAgent.gen_chapters),
        inspect.getsource(OutlineAgent._validate_chapter_outline),
        inspect.getsource(OutlineAgent._normalise_chapters),
    ]
    payload = "\n\n".join(components).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_cache_component(value: Any, fallback: str = "item") -> str:
    """Return a readable, traversal-safe component for cache paths."""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return (text or fallback)[:96]


class OutlineCache:
    """Immutable per-model/per-prompt outline-artifact cache.

    Each entry is an ordinary JSON file so it remains inspectable and rsync
    friendly on the cluster filesystem.  A per-key O_EXCL lock plus atomic
    ``os.replace`` prevents batch and experimental writers from sampling the
    same missing outline concurrently.
    """

    def __init__(self, cache_dir: Optional[str], policy: str,
                 model_id: str, model_request_name: str,
                 word_count: int, max_tokens: int, temperature: float,
                 disable_thinking: bool,
                 lock_timeout: float = DEFAULT_OUTLINE_CACHE_LOCK_TIMEOUT):
        if policy not in OUTLINE_CACHE_POLICIES:
            raise ValueError(f"unknown outline cache policy {policy!r}")
        if policy != "off" and not cache_dir:
            raise ValueError(
                "outline cache directory is required unless policy is off"
            )
        if lock_timeout <= 0:
            raise ValueError("outline cache lock timeout must be > 0")
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.policy = policy
        self.model_id = model_id
        self.model_request_name = model_request_name
        self.word_count = word_count
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.disable_thinking = bool(disable_thinking)
        self.lock_timeout = float(lock_timeout)

    def descriptor(self, story_id: Any, prompt: str,
                   lang: str) -> Dict[str, Any]:
        """Build the exact cache identity; state-update mode is excluded."""
        return {
            "schema_version": OUTLINE_CACHE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "story_id": str(story_id),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "word_count": self.word_count,
            "language": lang,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "disable_thinking": self.disable_thinking,
            "outline_pipeline_sha256": outline_pipeline_fingerprint(lang),
        }

    @staticmethod
    def cache_key(descriptor: Dict[str, Any]) -> str:
        encoded = json.dumps(
            descriptor, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def entry_path(self, descriptor: Dict[str, Any]) -> str:
        if self.cache_dir is None:
            raise RuntimeError("outline cache is disabled")
        model_dir = _safe_cache_component(self.model_id, "model")
        story_id = _safe_cache_component(descriptor["story_id"], "story")
        key = self.cache_key(descriptor)
        return os.path.join(
            self.cache_dir,
            model_dir,
            f"{self.word_count}w",
            f"{story_id}--{key[:24]}.json",
        )

    @staticmethod
    def _artifacts_sha256(artifacts: Dict[str, Any]) -> str:
        encoded = json.dumps(
            artifacts, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load(self, path: str, descriptor: Dict[str, Any],
              validator: Callable[[Dict[str, Any]], None],
              ) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            entry = json.load(file)
        if entry.get("descriptor") != descriptor:
            raise ValueError(f"outline cache descriptor mismatch: {path}")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"outline cache artifacts are not an object: {path}")
        actual_hash = self._artifacts_sha256(artifacts)
        validator(artifacts)
        return {
            "artifacts": copy.deepcopy(artifacts),
            "cache_key": self.cache_key(descriptor),
            "artifacts_sha256": actual_hash,
            "path": path,
            "created_at": entry.get("created_at"),
        }

    def _store(self, path: str, descriptor: Dict[str, Any],
               artifacts: Dict[str, Any]) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifacts_copy = copy.deepcopy(artifacts)
        artifacts_hash = self._artifacts_sha256(artifacts_copy)
        entry = {
            "schema_version": OUTLINE_CACHE_SCHEMA_VERSION,
            "descriptor": descriptor,
            "provenance": {
                "model_request_name": self.model_request_name,
            },
            "artifacts_sha256": artifacts_hash,
            "created_at": datetime.now().astimezone().isoformat(),
            "artifacts": artifacts_copy,
        }
        temp_path = (
            f"{path}.tmp.{os.getpid()}."
            f"{time.time_ns()}"
        )
        try:
            with open(temp_path, "x", encoding="utf-8") as file:
                json.dump(entry, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, path)
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
        return {
            "artifacts": copy.deepcopy(artifacts_copy),
            "cache_key": self.cache_key(descriptor),
            "artifacts_sha256": artifacts_hash,
            "path": path,
            "created_at": entry["created_at"],
        }

    async def _acquire_lock(self, lock_path: str) -> None:
        started = time.monotonic()
        while True:
            try:
                fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump({
                        "pid": os.getpid(),
                        "created_at": datetime.now().astimezone().isoformat(),
                    }, file)
                return
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                    if age > self.lock_timeout:
                        os.remove(lock_path)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() - started >= self.lock_timeout:
                    raise TimeoutError(
                        f"timed out waiting for outline cache lock {lock_path}"
                    )
                await asyncio.sleep(1.0)

    async def get_or_create(
            self, story_id: Any, prompt: str, lang: str,
            generator: Callable[[], Awaitable[Dict[str, Any]]],
            validator: Callable[[Dict[str, Any]], None],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load one immutable entry or generate it once under a key lock."""
        if self.policy == "off":
            artifacts = await generator()
            validator(artifacts)
            return artifacts, {"policy": "off", "status": "uncached"}

        descriptor = self.descriptor(story_id, prompt, lang)
        path = self.entry_path(descriptor)
        if self.policy != "refresh":
            loaded = self._load(path, descriptor, validator)
            if loaded is not None:
                metadata = {key: value for key, value in loaded.items()
                            if key != "artifacts"}
                metadata.update({"policy": self.policy, "status": "hit"})
                return loaded["artifacts"], metadata
        if self.policy == "require":
            raise FileNotFoundError(
                f"required outline cache entry is missing: {path}"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        lock_path = path + ".lock"
        await self._acquire_lock(lock_path)
        try:
            # Another process may have populated the entry while this process
            # waited.  Refresh is the only policy allowed to overwrite it.
            if self.policy != "refresh":
                loaded = self._load(path, descriptor, validator)
                if loaded is not None:
                    metadata = {key: value for key, value in loaded.items()
                                if key != "artifacts"}
                    metadata.update({
                        "policy": self.policy,
                        "status": "hit_after_wait",
                    })
                    return loaded["artifacts"], metadata
            artifacts = await generator()
            validator(artifacts)
            stored = self._store(path, descriptor, artifacts)
            metadata = {key: value for key, value in stored.items()
                        if key != "artifacts"}
            metadata.update({"policy": self.policy, "status": "generated"})
            return stored["artifacts"], metadata
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


# =============================================================================
# 6. Chapter execution context and full-story generation
# =============================================================================


def normalise_model_tool_calls(message: Any) -> Tuple[List[Any], str]:
    """Normalize standard OpenAI tool_calls and occasional XML tool calls."""
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    content = message.content or ""
    if tool_calls or "<tool_call>" not in content:
        return tool_calls, content

    xml_calls = parse_xml_tool_calls(content)
    if not xml_calls:
        return tool_calls, content
    content = re.sub(
        r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL
    ).strip()
    return [_SynthToolCall(call) for call in xml_calls], content


def assistant_history_entry(tool_calls: List[Any], content: str) -> Dict[str, Any]:
    """Convert an SDK message into a plain dict that can be resent to the API."""
    entry: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in tool_calls
        ]
    return entry


def compact_rejected_write_in_history(
        assistant_entry: Dict[str, Any],
        tool_call_id: str,
        result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Remove length-rejected prose from the working context; return a record.

    Full arguments remain in ``trajectory`` and ``runtime.tool_trace`` for
    diagnostics. Only the messages resent to the model are compacted, so
    invalid long drafts are not fed back repeatedly, where they would act as
    copy attractors and fill the context.
    """
    if result.get("ok") or "word_count" not in result:
        return None

    for call in assistant_entry.get("tool_calls") or []:
        if call.get("id") != tool_call_id:
            continue
        function = call.get("function") or {}
        if function.get("name") != "write":
            return None
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None
        content = args.get("content")
        if not isinstance(content, str) or not content:
            return None

        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        actual = result.get("word_count")
        target = result.get("target_word_count")
        args["content"] = (
            "[rejected write content omitted from working context; "
            f"word_count={actual}; target_word_count={target}; "
            f"sha256={digest[:16]}]"
        )
        function["arguments"] = json.dumps(args, ensure_ascii=False)
        return {
            "tool_call_id": tool_call_id,
            "word_count": actual,
            "target_word_count": target,
            "sha256": digest,
        }
    return None


def rejected_write_content_sha256(
        assistant_entry: Dict[str, Any],
        tool_call_id: str,
) -> Optional[str]:
    """Hash a rejected draft; used only to detect verbatim resubmissions."""
    for call in assistant_entry.get("tool_calls") or []:
        if call.get("id") != tool_call_id:
            continue
        function = call.get("function") or {}
        if function.get("name") != "write":
            return None
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None
        content = args.get("content")
        if not isinstance(content, str) or not content:
            return None
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    return None


def rejected_write_distance(result: Dict[str, Any]) -> Optional[float]:
    """Word distance from a rejected draft to the allowed range (zero inside it)."""
    actual = result.get("word_count")
    min_words = result.get("min_word_count")
    max_words = result.get("max_word_count")
    if not all(isinstance(value, (int, float))
               for value in (actual, min_words, max_words)):
        return None
    if actual < min_words:
        return float(min_words - actual)
    if actual > max_words:
        return float(actual - max_words)
    return 0.0


def compact_retained_rejection(
        rejected_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compact a still-retained rejected draft; no-op if only statistics are kept."""
    if not rejected_state:
        return None
    assistant_entry = rejected_state.get("assistant_entry")
    tool_call_id = rejected_state.get("tool_call_id")
    result = rejected_state.get("result")
    if not isinstance(assistant_entry, dict) or not tool_call_id:
        return None
    if not isinstance(result, dict):
        return None
    return compact_rejected_write_in_history(
        assistant_entry, tool_call_id, result
    )


def combined_compaction_record(
        records: List[Optional[Dict[str, Any]]],
        reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Merge the context compactions of one turn into a single trajectory record."""
    present = [record for record in records if record is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return {
        "reason": reason or "multiple_rejected_write_compactions",
        "items": present,
    }


def update_rejected_write_history(
        length_control_mode: str,
        latest_rejected: Optional[Dict[str, Any]],
        assistant_entry: Dict[str, Any],
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]],
           Optional[str]]:
    """Update how a rejected write is kept in the working context.

    ``compact`` and ``tool_target`` compact immediately without feedback;
    ``feedback_only`` compacts immediately and appends explicit length
    feedback; ``recent_failure_context`` always keeps the latest rejected
    draft; ``adaptive_recent_failure`` keeps a new draft only if its length
    distance improves by at least 10% over the best so far, otherwise only
    the length record is kept.

    Returns ``(latest_rejected, compaction, event)``, where event is
    ``rejected_retained``, ``rejected_compacted``, ``accepted``, or
    ``None``.
    """
    if tool_name != "write":
        return latest_rejected, None, None

    length_rejected = not result.get("ok") and "word_count" in result
    feedback_modes = {
        "feedback_only",
        "recent_failure_context",
        "adaptive_recent_failure",
    }
    if length_control_mode not in feedback_modes:
        compaction = compact_rejected_write_in_history(
            assistant_entry, tool_call_id, result
        )
        return latest_rejected, compaction, None

    if length_control_mode == "feedback_only":
        compaction = compact_rejected_write_in_history(
            assistant_entry, tool_call_id, result
        )
        if length_rejected:
            return None, compaction, "rejected_compacted"
        return None, compaction, None

    if length_rejected:
        previous_compaction = compact_retained_rejection(latest_rejected)
        content_sha256 = rejected_write_content_sha256(
            assistant_entry, tool_call_id
        )
        distance = rejected_write_distance(result)

        if length_control_mode == "recent_failure_context":
            return {
                "assistant_entry": assistant_entry,
                "tool_call_id": tool_call_id,
                "result": result,
                "content_sha256": content_sha256,
                "best_distance": distance,
            }, previous_compaction, "rejected_retained"

        # In adaptive mode the first rejected draft is kept in full. Later drafts are
        # kept only if their distance to the allowed range improves by at least 10%
        # over the best so far, so near-identical drafts do not become copy attractors.
        previous_best = (
            latest_rejected.get("best_distance")
            if latest_rejected is not None else None
        )
        previous_sha256 = (
            latest_rejected.get("content_sha256")
            if latest_rejected is not None else None
        )
        first_rejection = latest_rejected is None
        exact_repeat = (
            content_sha256 is not None
            and content_sha256 == previous_sha256
        )
        meaningful_improvement = first_rejection
        if (not first_rejection and distance is not None
                and isinstance(previous_best, (int, float))
                and previous_best > 0):
            meaningful_improvement = (
                distance
                <= previous_best * (1 - ADAPTIVE_LENGTH_MIN_IMPROVEMENT)
            )
        retain_current = meaningful_improvement and not exact_repeat
        best_distance = distance
        if isinstance(previous_best, (int, float)):
            if best_distance is None:
                best_distance = float(previous_best)
            else:
                best_distance = min(float(previous_best), best_distance)

        if retain_current:
            return {
                "assistant_entry": assistant_entry,
                "tool_call_id": tool_call_id,
                "result": result,
                "content_sha256": content_sha256,
                "best_distance": best_distance,
            }, previous_compaction, "rejected_retained"

        current_compaction = compact_rejected_write_in_history(
            assistant_entry, tool_call_id, result
        )
        return {
            "assistant_entry": None,
            "tool_call_id": None,
            "result": result,
            "content_sha256": content_sha256,
            "best_distance": best_distance,
        }, combined_compaction_record(
            [previous_compaction, current_compaction],
            reason="adaptive_rejected_write_not_improving",
        ), "rejected_compacted"

    if result.get("ok"):
        compaction = compact_retained_rejection(latest_rejected)
        return None, compaction, "accepted"

    return latest_rejected, None, None


def recent_failure_feedback(
        result: Dict[str, Any],
        lang: str,
        content_retained: bool = True,
) -> str:
    """State that the last write failed the length check and whether it is in context."""
    actual = result.get("word_count")
    target = result.get("target_word_count")
    min_words = result.get("min_word_count")
    max_words = result.get("max_word_count")
    length_feedback = ""
    if all(isinstance(value, (int, float)) for value in (
            actual, target, min_words, max_words)):
        if actual > max_words:
            keep_percent = max(1, round(100 * target / actual))
            length_feedback = (
                f" 同时，上一稿实际为 {actual} 字，目标 {target} 字，允许范围"
                f" {min_words}–{max_words} 字；请把完整稿压缩到当前长度约"
                f" {keep_percent}%，不要在改格式时扩写。"
                if lang == "zh" else
                f" At the same time, that draft has {actual} words; the target "
                f"is {target} and the accepted range is {min_words}-{max_words}. "
                f"Compress the complete draft to about {keep_percent}% of its "
                "current length; do not expand while fixing the format."
            )
        elif actual < min_words:
            grow_percent = max(1, round(100 * target / max(actual, 1)))
            length_feedback = (
                f" 同时，上一稿实际为 {actual} 字，目标 {target} 字，允许范围"
                f" {min_words}–{max_words} 字；请在保持格式的同时扩写到当前长度"
                f"约 {grow_percent}%。"
                if lang == "zh" else
                f" At the same time, that draft has {actual} words; the target "
                f"is {target} and the accepted range is {min_words}-{max_words}. "
                f"Expand the complete draft to about {grow_percent}% of its "
                "current length while preserving the required format."
            )
    if result.get("format_error"):
        examples = result.get("format_violations") or []
        rendered = " | ".join(str(item)[:100] for item in examples)
        if lang == "zh":
            feedback = (
                "上一次 write 违反强制纯对话格式。每个非空行必须以破折号“—”"
                "开头；推荐采用 —\"完整台词\"，也允许 —完整台词。每行只能有"
                "一条不超过 160 词、无说话者标签的台词；"
                "拆分超长行，并删除标题、旁白、动作说明"
                "和场景描写。"
            )
            if rendered:
                feedback += f"违规行示例：{rendered}。"
            return feedback + length_feedback + (
                "失败稿仍在对话历史中，请直接重写为合规纯对话。"
                if content_retained else
                "失败稿已从工作上下文压缩，请重新提交合规纯对话全文。"
            )
        feedback = (
            "The previous write violated the enforced dialogue-only format. "
            "Every non-empty line must be one utterance beginning with an em "
            "dash. Prefer —\"complete utterance\"; bare —complete utterance is "
            "also valid. Keep it unattributed and at most 160 words; split "
            "overlong lines and remove headings, narration, "
            "action beats, and scene description."
        )
        if rendered:
            feedback += f" Invalid-line examples: {rendered}."
        return feedback + length_feedback + (
            " The rejected draft remains in context; rewrite it directly into "
            "compliant dialogue."
            if content_retained else
            " The rejected draft was compacted out; submit a new complete "
            "dialogue-only draft."
        )
    if result.get("integrity_error"):
        examples = result.get("integrity_violations") or []
        rendered = " | ".join(str(item)[:140] for item in examples)
        feedback = (
            "上一次 write 未通过章节完整性或故事契约检查。"
            if lang == "zh" else
            "The previous write failed the chapter-integrity or Story "
            "Contract check."
        )
        if rendered:
            feedback += (
                f" 原因：{rendered}。" if lang == "zh"
                else f" Reasons: {rendered}."
            )
        feedback += length_feedback
        return feedback + (
            "失败稿仍在对话历史中；请直接重写完整章节，修复断句、重复或契约冲突。"
            if lang == "zh" and content_retained else
            "失败稿已从工作上下文压缩；请重新提交无断句、无大段重复且符合契约的完整章节。"
            if lang == "zh" else
            " The rejected draft remains in context; rewrite the complete "
            "chapter and fix the truncation, repetition, or contract conflict."
            if content_retained else
            " The rejected draft was compacted out; submit a new complete "
            "chapter without truncation, long repetition, or contract conflicts."
        )
    if lang == "zh":
        feedback = (
            f"上一次 write 提交的完整稿字数不合格：实际为 {actual} 字；"
            f"本章目标为 {target} 字，允许范围为 {min_words}–{max_words} 字。"
        )
        if content_retained:
            return feedback + "该失败稿目前仍完整保留在对话历史中。"
        return feedback + "该失败稿正文已从工作上下文压缩，字数记录仍保留。"
    feedback = (
        "The complete draft submitted in the previous write failed "
        f"word-count validation: it contains {actual} words. This chapter "
        f"targets {target} words and accepts {min_words}-{max_words} words."
    )
    if content_retained:
        return feedback + (
            " The rejected draft currently remains fully visible in the "
            "conversation history."
        )
    return feedback + (
        " The rejected draft has been compacted out of the working context; "
        "its word-count record remains visible."
    )


def fresh_prose_retry_messages(
        system_prompt: str,
        user_prompt: str,
        feedback: str,
        lang: str,
) -> List[Dict[str, str]]:
    """Start an independent prose attempt without rejected-draft history.

    Long dialogue drafts are strong copy attractors for small models. A
    placeholder tool argument is unsafe too because the model can submit the
    placeholder itself as prose. Rebuilding the two-message context keeps the
    immutable task and validation feedback while removing both attractors.
    """
    heading = "# 上一次独立尝试的校验反馈" if lang == "zh" else (
        "# Validation Feedback From the Previous Independent Attempt"
    )
    instruction = (
        "不要续写、修补或复述上一稿；从头生成一份独立、完整的新正文。"
        if lang == "zh" else
        "Do not continue, patch, or paraphrase the rejected draft. Generate an "
        "independent new complete chapter from the beginning."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n{heading}\n{feedback}\n{instruction}"
            ),
        },
    ]


def parse_tool_arguments(tool_call: Any) -> Dict[str, Any]:
    """Return {} for invalid JSON arguments so the tool can report a recoverable error."""
    try:
        return json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def is_done_signal(text: str) -> bool:
    """Accept DONE, optionally followed by one common sentence-ending mark."""
    return re.fullmatch(r"\s*DONE[.!。！]?\s*", text or "") is not None


@dataclass
class ChapterRuntime:
    """Mutable state of one chapter's tool loop: budgets, drafts, and trace."""

    chapter_id: int
    chapter_name: str
    read_used: int = 0
    search_used: int = 0
    correct_used: int = 0
    write_attempts: int = 0
    write_succeeded: bool = False
    written_chapter: Optional[Dict[str, Any]] = None
    last_rejected_draft: Optional[Dict[str, Any]] = None
    current_chapter_corrected: bool = False
    update_log: List[Dict[str, Any]] = field(default_factory=list)
    correct_log: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)

    def _draft_from(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": self.chapter_id,
            "name": args.get("title", self.chapter_name),
            "content": args.get("content", ""),
        }

    def record_tool_result(self, turn: int, name: str, args: Dict[str, Any],
                           result: Dict[str, Any]) -> None:
        """Record one tool call and advance the matching budget and draft state."""
        trace = {
            "turn": turn,
            "name": name,
            "args": args,
            "ok": bool(result.get("ok")),
            "info": result.get("info") or result.get("error") or "",
        }
        for key in (
            "action", "state_name", "state_id", "chapter_id",
            "match_score", "word_count", "format_error",
        ):
            if key in result:
                trace[key] = result.get(key)
        self.tool_trace.append(trace)

        if name == "read" and result.get("budget_charged"):
            self.read_used += 1
        if name == "search" and result.get("budget_charged"):
            self.search_used += 1

        if name == "write":
            # An attempt counts only if the arguments are valid and words were counted.
            # A length failure does not use up the write; after ok=true, write is closed.
            if "word_count" in result:
                self.write_attempts += 1
            if result.get("ok"):
                self.write_succeeded = True
                self.written_chapter = self._draft_from(args)
            elif ("word_count" in result
                  and self.written_chapter is None
                  and args.get("content")):
                self.last_rejected_draft = self._draft_from(args)

        if name == "correct" and result.get("ok"):
            self.correct_used += 1
            if args.get("chapter_id") == self.chapter_id:
                self.current_chapter_corrected = True
            self.correct_log.append({
                "chapter_id": args.get("chapter_id"),
                "old_text": args.get("old_text", "")[:80],
            })

        if name == "update":
            # An atomic update changes state only if the whole batch succeeds; operations
            # that passed item validation but were never committed do not count.
            if result.get("ok"):
                for operation_result in result.get("results") or []:
                    if not (isinstance(operation_result, dict)
                            and operation_result.get("ok")):
                        continue
                    if operation_result.get("action") in {
                            "unchanged", "ignored"}:
                        continue
                    self.update_log.append({
                        "state_name": operation_result.get("state_name"),
                        "action": operation_result.get("action"),
                        "state_id": operation_result.get("state_id"),
                    })
        elif name in ITEM_STATE_TOOL_NAMES and result.get("ok"):
            # Each flat tool call is one state change. Idempotent calls return unchanged
            # and are not counted as new narrative changes.
            if result.get("action") not in {"unchanged", "ignored"}:
                self.update_log.append({
                    "state_name": result.get("state_name"),
                    "action": result.get("action"),
                    "state_id": result.get("state_id"),
                })

    def finish(self, logger: logging.Logger) -> Tuple[Dict[str, Any], bool]:
        """Finish the chapter: accepted draft, else rejected draft, else raise."""
        degraded = False
        if self.written_chapter is None and self.last_rejected_draft is not None:
            self.written_chapter = self.last_rejected_draft
            degraded = True
            logger.warning(
                f"  Chapter {self.chapter_id}: no accepted write; falling back "
                f"to the last rejected draft "
                f"({count_words(self.written_chapter['content'])} words)"
            )
        if self.written_chapter is None:
            raise RuntimeError(
                f"Chapter {self.chapter_id}: model never produced any write "
                "call with content"
            )

        self.written_chapter["tool_trace"] = self.tool_trace
        self.written_chapter["degraded"] = degraded
        return self.written_chapter, degraded


@dataclass
class StoryWorkspace:
    """In-memory workspace for one story; also the source of checkpoint data."""

    story_id: Any
    prompt: str
    lang: str
    target_words: int
    fingerprint: str
    artifacts: Dict[str, Any]
    outline: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    state: StoryState
    outline_cache: Optional[Dict[str, Any]] = None

    def checkpoint_data(self) -> Dict[str, Any]:
        """Build checkpoint data that can be written atomically and fully restored."""
        return {
            "version": CHECKPOINT_VERSION,
            "fingerprint": self.fingerprint,
            "id": self.story_id,
            "prompt": self.prompt,
            "lang": self.lang,
            "word_count_target": self.target_words,
            "artifacts": self.artifacts,
            "outline": self.outline,
            "chapters": self.chapters,
            "state": self.state.to_dict(),
            "outline_cache": self.outline_cache,
        }

    def result_record(self, failed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble the per-story output record."""
        record = {
            "id": self.story_id,
            "prompt": self.prompt,
            "outline_artifacts": {
                key: value
                for key, value in self.artifacts.items()
                if key != "outline"
            },
            "outline": self.outline,
            "story": self.chapters,
            "final_state": self.state.to_dict(),
            "complete": failed is None,
        }
        if self.outline_cache is not None:
            record["outline_cache"] = self.outline_cache
        if failed:
            record["failed_chapter"] = failed
        return record


class LongStoryAgent:
    """Run the per-chapter tool loop and maintain the story checkpoint and state."""

    def __init__(self, client: AsyncOpenAI, model: str,
                 max_tokens: int, temperature: float,
                 word_count: int, logger: logging.Logger,
                 length_control_mode: str = DEFAULT_LENGTH_CONTROL_MODE,
                 chapter_token_control: str = DEFAULT_CHAPTER_TOKEN_CONTROL,
                 dynamic_token_ratio_en: float =
                 DEFAULT_DYNAMIC_TOKEN_RATIO_EN,
                 dynamic_token_ratio_zh: float =
                 DEFAULT_DYNAMIC_TOKEN_RATIO_ZH,
                 dynamic_token_overhead: int =
                 DEFAULT_DYNAMIC_TOKEN_OVERHEAD,
                 dynamic_token_minimum: int =
                 DEFAULT_DYNAMIC_TOKEN_MINIMUM,
                 max_turns_per_chapter: int = MAX_TURNS_PER_CHAPTER,
                 request_attempts: int = 3,
                 state_update_mode: str = DEFAULT_STATE_UPDATE_MODE,
                 narrative_ops_guidance_placement: str =
                 DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT,
                 disable_thinking: bool = False,
                 outline_cache_dir: Optional[str] = None,
                 outline_cache_policy: str = DEFAULT_OUTLINE_CACHE_POLICY,
                 outline_cache_model_id: Optional[str] = None,
                 outline_cache_lock_timeout: float =
                 DEFAULT_OUTLINE_CACHE_LOCK_TIMEOUT):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.word_count = word_count
        self.logger = logger
        self.length_control_mode = length_control_mode
        if state_update_mode not in STATE_UPDATE_MODES:
            raise ValueError(f"unknown state_update_mode {state_update_mode!r}")
        self.state_update_mode = state_update_mode
        if narrative_ops_guidance_placement not in \
                NARRATIVE_OPS_GUIDANCE_PLACEMENTS:
            raise ValueError(
                "unknown narrative_ops_guidance_placement "
                f"{narrative_ops_guidance_placement!r}"
            )
        self.narrative_ops_guidance_placement = (
            narrative_ops_guidance_placement
        )
        self.chapter_token_control = chapter_token_control
        self.dynamic_token_ratio_en = dynamic_token_ratio_en
        self.dynamic_token_ratio_zh = dynamic_token_ratio_zh
        self.dynamic_token_overhead = dynamic_token_overhead
        self.dynamic_token_minimum = dynamic_token_minimum
        # API runs may lower these two limits; local runs use the defaults.
        self.max_turns_per_chapter = max_turns_per_chapter
        self.request_attempts = request_attempts
        self.disable_thinking = disable_thinking
        self.outline_cache = OutlineCache(
            cache_dir=outline_cache_dir,
            policy=outline_cache_policy,
            model_id=outline_cache_model_id or model,
            model_request_name=model,
            word_count=word_count,
            max_tokens=max_tokens,
            temperature=temperature,
            disable_thinking=disable_thinking,
            lock_timeout=outline_cache_lock_timeout,
        )
        self.search_index = ChapterSearchIndex()

    def _chapter_max_tokens(self, chapter_target_words: int,
                            lang: str,
                            dialogue_only: bool = False) -> int:
        """Per-turn output budget for the current chapter; fixed mode uses max_tokens."""
        if self.chapter_token_control == "fixed":
            return self.max_tokens
        ratio = (
            self.dynamic_token_ratio_zh
            if lang == "zh" else self.dynamic_token_ratio_en
        )
        estimated = math.ceil(
            chapter_target_words * ratio + self.dynamic_token_overhead
        )
        if dialogue_only:
            # Dialogue-only chapters place the whole chapter inside one tool
            # argument while Qwen also emits hidden reasoning.  Real traces
            # showed the tool JSON being cut off even when ordinary prose at
            # the same word target fit.  Reserve protocol headroom without
            # changing the global cap or the turn/resume budgets.
            estimated = max(
                estimated,
                math.ceil(chapter_target_words * ratio + 2048),
            )
        return min(
            self.max_tokens,
            max(self.dynamic_token_minimum, estimated),
        )

    async def _llm(self, messages: List[Dict], use_tools: bool = False,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   max_tokens: Optional[int] = None):
        requested_max_tokens = (
            max_tokens if max_tokens is not None else self.max_tokens
        )
        kwargs: Dict[str, Any] = dict(
            model=self.model, messages=messages,
            max_tokens=requested_max_tokens, temperature=self.temperature,
        )
        if self.disable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        if use_tools:
            kwargs["tools"] = tools if tools is not None else TOOLS
            kwargs["tool_choice"] = "auto"
        for retry in range(self.request_attempts):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                # Some OpenAI-compatible servers return finish_reason=length at max_tokens;
                # Qwen's auto tool parser may still report tool_calls when the budget is
                # exactly exhausted, so usage is checked as well.
                allowed = (
                    ("stop", "tool_calls", "length")
                    if use_tools else ("stop", "length")
                )
                finish_reason = require_finish_reason(choice, allowed)
                completion_tokens = (
                    getattr(resp.usage, "completion_tokens", None)
                    if getattr(resp, "usage", None) is not None else None
                )
                usage = {
                    "requested_max_tokens": requested_max_tokens,
                    "completion_tokens": completion_tokens,
                    "token_limit_reached": (
                        isinstance(completion_tokens, int)
                        and completion_tokens >= requested_max_tokens
                    ),
                }
                return choice.message, finish_reason, usage
            except Exception as e:
                if retry == self.request_attempts - 1:
                    raise
                self.logger.warning(f"  LLM error (retry {retry+1}): {e}")
                await asyncio.sleep(5 * (retry + 1))
        raise RuntimeError("chapter LLM call exhausted retries")

    def _save_failed_chapter_trace(
            self, trace_dir: Optional[str], story_id: Any,
            chapter_info: Dict[str, Any], user_msg: str,
            trajectory: List[Dict[str, Any]], runtime: ChapterRuntime,
            error: str) -> Optional[str]:
        """Save the full per-turn trace of a failed chapter; errors here never stop generation."""
        if not trace_dir:
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(
            trace_dir,
            f"chapter_{chapter_info['id']}_failed_{stamp}.json",
        )
        tmp_path = path + ".tmp"
        payload = {
            "story_id": story_id,
            "chapter_id": chapter_info["id"],
            "chapter_name": chapter_info.get("name"),
            "chapter_target_words": chapter_info.get("word_count"),
            "length_control_mode": self.length_control_mode,
            "chapter_token_control": self.chapter_token_control,
            "state_update_mode": self.state_update_mode,
            "narrative_ops_guidance_placement": (
                self.narrative_ops_guidance_placement
            ),
            "error": error,
            "initial_user_message": user_msg,
            "trajectory": trajectory,
            "runtime": {
                "read_used": runtime.read_used,
                "search_used": runtime.search_used,
                "correct_used": runtime.correct_used,
                "write_attempts": runtime.write_attempts,
                "write_succeeded": runtime.write_succeeded,
                "has_accepted_write": runtime.written_chapter is not None,
                "tool_trace": runtime.tool_trace,
            },
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            self.logger.warning(f"  Failed chapter trace saved: {path}")
            return path
        except Exception as trace_error:
            self.logger.warning(
                f"  Failed to save chapter trace for story {story_id}: "
                f"{trace_error}"
            )
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return None

    def _build_chapter_prompt(self, prompt: str, outline: List[Dict],
                              chapter_info: Dict, state: StoryState,
                              chapters: List[Dict[str, Any]], lang: str) -> str:
        visible_outline = visible_outline_for_chapter(
            outline, chapter_info, self.state_update_mode
        )
        outline_str = json.dumps(visible_outline, ensure_ascii=False, indent=2)
        state_str = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        completed_words = sum(count_words(c["content"]) for c in chapters)
        return chapter_prompt(
            lang,
            outline=outline_str,
            state=state_str,
            ch_id=chapter_info["id"],
            ch_name=chapter_info.get("name", f"Chapter {chapter_info['id']}"),
            ch_desc=chapter_info.get("description", ""),
            ch_target=chapter_info.get("word_count", 1000),
            completed=len(chapters),
            completed_words=completed_words,
            state_update_mode=self.state_update_mode,
            narrative_ops_guidance_placement=(
                self.narrative_ops_guidance_placement
            ),
            story_contract=prompt,
        )

    async def _run_chapter_loop(self, prompt: str, outline: List[Dict],
                                chapter_info: Dict,
                                chapters: List[Dict[str, Any]],
                                state: StoryState,
                                lang: str,
                                story_id: int,
                                trace_dir: Optional[str] = None) -> Dict[str, Any]:
        ch_id = chapter_info["id"]
        ch_name = chapter_info.get("name", f"Chapter {ch_id}")
        ch_target_wc = chapter_info.get("word_count", 1000)
        self.logger.info(f"Chapter {ch_id}: {ch_name}")

        user_msg = self._build_chapter_prompt(
            prompt, outline, chapter_info, state, chapters, lang
        )
        chapter_system_prompt = get_system_prompt("chapter", lang)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": chapter_system_prompt},
            {"role": "user", "content": user_msg},
        ]

        tools_for_chapter = chapter_tools(
            ch_target_wc, lang, self.length_control_mode,
            self.state_update_mode,
            self.narrative_ops_guidance_placement,
        )
        runtime = ChapterRuntime(chapter_id=ch_id, chapter_name=ch_name)
        trajectory: List[Dict[str, Any]] = []
        latest_rejected: Optional[Dict[str, Any]] = None
        chapter_max_tokens = self._chapter_max_tokens(
            ch_target_wc,
            lang,
            dialogue_only=(
                self.state_update_mode == "semantic"
                and requires_dialogue_only(prompt)
            ),
        )
        self.logger.info(
            f"  Chapter {ch_id}: per-turn max_tokens={chapter_max_tokens} "
            f"({self.chapter_token_control})"
        )

        for turn in range(self.max_turns_per_chapter):
            summary_positions = [
                index for index, item in enumerate(runtime.tool_trace)
                if item.get("name") == "record_chapter_summary"
                and item.get("ok")
                and item.get("action") in ("add", "edit")
            ]
            current_correction_positions = [
                index for index, item in enumerate(runtime.tool_trace)
                if item.get("name") == "correct"
                and item.get("ok")
                and (item.get("args") or {}).get("chapter_id") == ch_id
            ]
            semantic_summary_done = bool(summary_positions) and (
                not current_correction_positions
                or max(summary_positions) > max(current_correction_positions)
            )
            active_tools = chapter_tools_for_phase(
                tools_for_chapter,
                self.state_update_mode,
                runtime.write_succeeded,
                semantic_summary_done,
            )
            try:
                msg, finish_reason, usage = await self._llm(
                    messages,
                    use_tools=True,
                    tools=active_tools,
                    max_tokens=chapter_max_tokens,
                )
            except Exception as e:
                self._save_failed_chapter_trace(
                    trace_dir, story_id, chapter_info, user_msg,
                    trajectory, runtime, str(e),
                )
                raise
            tool_calls, content_text = normalise_model_tool_calls(msg)
            assistant_entry = assistant_history_entry(tool_calls, content_text)
            messages.append(assistant_entry)
            turn_record: Dict[str, Any] = {
                "turn": turn,
                "finish_reason": finish_reason,
                "assistant_content": content_text,
                "reasoning": (
                    getattr(msg, "reasoning", None)
                    or getattr(msg, "reasoning_content", None)
                ),
                "usage": usage,
                "tool_calls": [],
                "tool_results": [],
            }
            for tc in tool_calls:
                turn_record["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": parse_tool_arguments(tc),
                })
            trajectory.append(turn_record)

            if not tool_calls:
                if (runtime.written_chapter is not None
                        and is_done_signal(content_text)
                        and (
                            self.state_update_mode != "semantic"
                            or semantic_summary_done
                        )):
                    break
                if runtime.written_chapter is None:
                    if self.state_update_mode == "semantic":
                        nudge = (
                            f"你尚未为第 {ch_id} 章成功写入正文。当前只开放写作阶段工具；"
                            "请调用 write 写入符合字数要求的完整正文。"
                        ) if lang == "zh" else (
                            f"Chapter {ch_id} has not been written successfully. "
                            "Only content-phase tools are currently available; call "
                            "write with a complete draft within the word-count range."
                        )
                    elif self.state_update_mode == "narrative_ops":
                        nudge = (
                            f"你尚未为第 {ch_id} 章成功调用 write。请先调用 write 写出"
                            "符合字数要求的完整正文，之后只调用一次 update，并在同一"
                            "调用中填写四个叙事操作数组，最后输出 DONE。"
                        ) if lang == "zh" else (
                            f"You have not successfully written chapter {ch_id}. "
                            "Call write with a complete draft that satisfies the "
                            "word-count requirement, then make exactly one update "
                            "call containing all four narrative-operation arrays, "
                            "and finally output DONE."
                        )
                    elif self.state_update_mode == "state_grouped":
                        nudge = (
                            f"你尚未为第 {ch_id} 章成功调用 write。请先调用 write 写出"
                            "符合字数要求的完整正文，之后用一次 update 按 state_name "
                            "分组合并全部状态变化，最后输出 DONE。"
                        ) if lang == "zh" else (
                            f"You have not successfully written chapter {ch_id}. "
                            "Call write with a complete draft that satisfies the "
                            "word-count requirement, then make one update call "
                            "grouping every state change by state_name, and "
                            "finally output DONE."
                        )
                    elif self.state_update_mode == "state_first":
                        nudge = (
                            f"你尚未为第 {ch_id} 章成功调用 write。请先调用 write 写出"
                            "符合字数要求的完整正文，之后按 state_name 分别 update，"
                            "最后输出 DONE。"
                        ) if lang == "zh" else (
                            f"You have not successfully written chapter {ch_id}. "
                            "Call write with a complete draft that satisfies the "
                            "word-count requirement, then call update separately "
                            "for each changed state_name and finally output DONE."
                        )
                    elif self.state_update_mode == "simple":
                        nudge = (
                            f"你尚未为第 {ch_id} 章成功调用 write。请先调用 write 写出"
                            "符合字数要求的完整正文，之后用简单状态工具更新变化，"
                            "最后输出 DONE。"
                        ) if lang == "zh" else (
                            f"You have not successfully written chapter {ch_id}. "
                            "Call write with a complete draft that satisfies the "
                            "word-count requirement, then use the simple state "
                            "tools for its changes and finally output DONE."
                        )
                    else:
                        nudge = (
                            f"你尚未为第 {ch_id} 章成功调用 write。请先调用 write 写出"
                            "符合字数要求的完整正文，之后再批量 update，最后输出 DONE。"
                        ) if lang == "zh" else (
                            f"You have not successfully written chapter {ch_id}. "
                            "Call write with a complete draft that satisfies the "
                            "word-count requirement, then batch update the state "
                            "and finally output DONE."
                        )
                else:
                    if self.state_update_mode == "semantic":
                        if not semantic_summary_done:
                            nudge = (
                                "正文已经写入。现在必须调用 record_chapter_summary 一次，"
                                "写入本章唯一的因果摘要；再按需更新角色或跨章未完成事项。"
                            ) if lang == "zh" else (
                                "The prose is accepted. You must now call "
                                "record_chapter_summary once with this chapter's "
                                "single causal summary; then update characters or "
                                "cross-chapter open requirements only as needed."
                            )
                        else:
                            nudge = (
                                "本章因果摘要已记录。如无其他实质状态变化，现在只输出 DONE。"
                            ) if lang == "zh" else (
                                "The chapter summary is recorded. If no other material "
                                "state change remains, output only DONE now."
                            )
                    elif self.state_update_mode == "narrative_ops":
                        nudge = (
                            "本章只有在你返回内容严格为 DONE、且不调用任何工具时才会结束。"
                            "如状态尚未更新，请只调用一次 update，并同时填写角色 upsert、"
                            "过去事件、新增未来要求和已兑现要求 key 四个数组；否则现在"
                            "只输出 DONE。"
                        ) if lang == "zh" else (
                            "The chapter ends only when you return exactly DONE "
                            "without any tool calls. If state is not updated yet, "
                            "make exactly one update call containing the character "
                            "upserts, past events, new future requirements, and "
                            "resolved requirement keys; otherwise output only DONE now."
                        )
                    elif self.state_update_mode == "state_grouped":
                        nudge = (
                            "本章只有在你返回内容严格为 DONE、且不调用任何工具时才会结束。"
                            "如状态尚未更新，请用一次 update 按 state_name 分组合并所有"
                            "状态变化；否则现在只输出 DONE。"
                        ) if lang == "zh" else (
                            "The chapter ends only when you return exactly DONE "
                            "without any tool calls. If state is not updated yet, "
                            "make one update call grouping every change by "
                            "state_name; otherwise output only DONE now."
                        )
                    elif self.state_update_mode == "state_first":
                        nudge = (
                            "本章只有在你返回内容严格为 DONE、且不调用任何工具时才会结束。"
                            "如状态尚未更新，请按发生变化的 state_name 分别调用 update；"
                            "否则现在只输出 DONE。"
                        ) if lang == "zh" else (
                            "The chapter ends only when you return exactly DONE "
                            "without any tool calls. If state is not updated yet, "
                            "call update separately for each changed state_name; "
                            "otherwise output only DONE now."
                        )
                    elif self.state_update_mode == "simple":
                        nudge = (
                            "本章只有在你返回内容严格为 DONE、且不调用任何工具时才会结束。"
                            "如状态尚未更新，请对每项变化调用对应的简单状态工具；"
                            "否则现在只输出 DONE。"
                        ) if lang == "zh" else (
                            "The chapter ends only when you return exactly DONE "
                            "without any tool calls. If state is not updated yet, "
                            "call the matching simple state tool for each change; "
                            "otherwise output only DONE now."
                        )
                    else:
                        nudge = (
                            "本章只有在你返回内容严格为 DONE、且不调用任何工具时才会结束。"
                            "如状态尚未更新，请先调用一次批量 update；否则现在只输出 DONE。"
                        ) if lang == "zh" else (
                            "The chapter ends only when you return exactly DONE "
                            "without any tool calls. If state is not updated yet, "
                            "call one batched update first; otherwise output only DONE now."
                        )
                messages.append({"role": "user", "content": nudge})
                turn_record["nudge"] = nudge
                continue

            feedback_result: Optional[Dict[str, Any]] = None
            feedback_content_retained = False
            state_phase_transition = False
            for tc in tool_calls:
                fname = tc.function.name
                args = parse_tool_arguments(tc)
                result = self._dispatch_tool(
                    fname, args, ch_id, chapters, state,
                    runtime.read_used, runtime.search_used,
                    runtime.written_chapter,
                    runtime.correct_used, runtime.write_succeeded,
                    ch_target_wc, lang, runtime.current_chapter_corrected,
                    prompt,
                )
                runtime.record_tool_result(turn, fname, args, result)
                turn_record["tool_results"].append({
                    "tool_call_id": tc.id,
                    "name": fname,
                    "result": result,
                })
                latest_rejected, compaction, write_event = (
                    update_rejected_write_history(
                        self.length_control_mode,
                        latest_rejected,
                        assistant_entry,
                        tc.id,
                        fname,
                        result,
                    )
                )
                if compaction is not None:
                    turn_record.setdefault(
                        "working_context_compactions", []
                    ).append(compaction)
                if write_event in {
                    "rejected_retained",
                    "rejected_compacted",
                }:
                    feedback_result = result
                    feedback_content_retained = (
                        write_event == "rejected_retained"
                    )
                elif write_event == "accepted":
                    feedback_result = None
                    feedback_content_retained = False
                    state_phase_transition = (
                        self.state_update_mode == "semantic"
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            if state_phase_transition:
                transition = semantic_state_phase_prompt(lang, ch_id)
                messages.append({"role": "user", "content": transition})
                turn_record["phase_transition"] = transition
            if feedback_result is not None:
                isolate_dialogue_retry = (
                    self.state_update_mode == "semantic"
                    and requires_dialogue_only(prompt)
                )
                feedback = recent_failure_feedback(
                    feedback_result,
                    lang,
                    content_retained=(
                        False if isolate_dialogue_retry
                        else feedback_content_retained
                    ),
                )
                if isolate_dialogue_retry:
                    messages = fresh_prose_retry_messages(
                        chapter_system_prompt, user_msg, feedback, lang
                    )
                    latest_rejected = None
                    turn_record["working_context_reset"] = {
                        "reason": "rejected_dialogue_draft_isolated",
                        "retained_messages": 2,
                    }
                else:
                    messages.append({"role": "user", "content": feedback})
                turn_record["length_feedback"] = feedback
                turn_record["length_feedback_content_retained"] = (
                    feedback_content_retained and not isolate_dialogue_retry
                )
        else:
            error = (
                f"Chapter {ch_id}: model did not output DONE within "
                f"{self.max_turns_per_chapter} turns"
            )
            self._save_failed_chapter_trace(
                trace_dir, story_id, chapter_info, user_msg,
                trajectory, runtime, error,
            )
            raise RuntimeError(error)

        written_chapter, degraded = runtime.finish(self.logger)
        written_chapter["generation_stats"] = {
            "chapter_token_control": self.chapter_token_control,
            "per_turn_max_tokens": chapter_max_tokens,
            "turns": len(trajectory),
            "completion_tokens": sum(
                item.get("usage", {}).get("completion_tokens") or 0
                for item in trajectory
            ),
            "token_limit_hits": sum(
                bool(item.get("usage", {}).get("token_limit_reached"))
                for item in trajectory
            ),
            "write_attempts": runtime.write_attempts,
        }
        wc = count_words(written_chapter["content"])
        self.logger.info(
            f"  Chapter {ch_id}: {wc} words, "
            f"{runtime.read_used} read calls, "
            f"{runtime.search_used} search calls, "
            f"{len(runtime.correct_log)} corrections, "
            f"{len(runtime.update_log)} state updates"
            + (" [DEGRADED]" if degraded else "")
        )

        return written_chapter

    def _dispatch_tool(self, name: str, args: Dict[str, Any],
                       current_ch_id: int,
                       chapters: List[Dict[str, Any]],
                       state: StoryState,
                       read_used: int,
                       search_used: int,
                       current_draft: Optional[Dict[str, Any]] = None,
                       correct_used: int = 0,
                       write_succeeded: bool = False,
                       chapter_target_wc: int = 1000,
                       lang: str = "en",
                       current_chapter_corrected: bool = False,
                       story_contract: str = "") -> Dict[str, Any]:
        """Validate and run one tool call; always returns an ``{ok, ...}`` result."""
        if name == "read":
            return self._tool_read(
                args, current_ch_id, chapters, current_draft,
                read_used,
            )
        if name == "search":
            return self._tool_search(args, chapters, search_used)
        if name == "correct":
            result = self._tool_correct(
                args, current_ch_id, chapters, current_draft, correct_used,
            )
            if result.get("ok"):
                self.search_index.mark_dirty(args["chapter_id"])
            return result
        if name == "write":
            write_args = args
            attempted_chapter_id = args.get("chapter_id")
            if (self.state_update_mode == "semantic"
                    and attempted_chapter_id != current_ch_id):
                # The current chapter is fixed at runtime. Small models often fill in a
                # one-based or next chapter number; in semantic mode normalize this redundant
                # argument so correct prose is not discarded over an off-by-one.
                write_args = dict(args)
                write_args["chapter_id"] = current_ch_id
            result = self._tool_write(
                write_args, current_ch_id, write_succeeded, chapter_target_wc,
                lang, current_chapter_corrected,
                story_contract if self.state_update_mode == "semantic" else "",
                chapters if self.state_update_mode == "semantic" else None,
            )
            if attempted_chapter_id != current_ch_id:
                result["chapter_id_normalized_from"] = attempted_chapter_id
            return result
        if (self.state_update_mode == "semantic"
                and name in SEMANTIC_STATE_TOOL_NAMES):
            return self._tool_semantic_state(
                name, args, state, current_ch_id
            )
        if name in SIMPLE_STATE_TOOL_NAMES:
            return self._tool_simple_state(name, args, state)
        if name == "update" and self.state_update_mode == "narrative_ops":
            return self._tool_narrative_ops_update(
                args, state, current_ch_id
            )
        if name == "update":
            return self._tool_update(args, state)

        return {"ok": False, "error": f"unknown tool {name}"}

    @staticmethod
    def _tool_read(args: Dict[str, Any], current_ch_id: int,
                   chapters: List[Dict[str, Any]],
                   current_draft: Optional[Dict[str, Any]],
                   read_used: int) -> Dict[str, Any]:
        """Read a prior chapter or the current draft; only successful reads use budget."""
        chapter_id = args.get("chapter_id")
        if not isinstance(chapter_id, int) or chapter_id > current_ch_id:
            return {
                "ok": False,
                "budget_charged": False,
                "error": "chapter_id must not be a future chapter",
            }
        if chapter_id == current_ch_id:
            chapter = current_draft
            missing_error = (
                "current chapter is not available until write succeeds"
            )
        else:
            chapter = next(
                (item for item in chapters if item["id"] == chapter_id), None
            )
            missing_error = f"chapter {chapter_id} not found"
        if chapter is None:
            return {
                "ok": False,
                "budget_charged": False,
                "error": missing_error,
            }
        if read_used >= MAX_READ:
            return {
                "ok": False,
                "budget_charged": False,
                "error": "read budget exhausted",
            }
        return {
            "ok": True,
            "budget_charged": True,
            "chapter_id": chapter_id,
            "name": chapter["name"],
            "content": chapter["content"],
        }

    def _tool_search(self, args: Dict[str, Any],
                     chapters: List[Dict[str, Any]],
                     search_used: int) -> Dict[str, Any]:
        """Search the incremental full-text index; every valid executed query uses budget."""
        query = args.get("query", "")
        if not query or not isinstance(query, str):
            return {
                "ok": False,
                "budget_charged": False,
                "error": "query must be a non-empty string",
            }
        if search_used >= MAX_SEARCH:
            return {
                "ok": False,
                "budget_charged": False,
                "error": "search budget exhausted",
            }
        hits = self.search_index.search(chapters, query)
        return {
            "ok": True,
            "budget_charged": True,
            "query": query,
            "hits": hits,
            "hit_count": len(hits),
        }

    @staticmethod
    def _normalised_state_text(value: Any) -> str:
        """Normalize a name or description for upserts and idempotent deduplication."""
        if not isinstance(value, str):
            return ""
        return " ".join(value.split()).casefold()

    @staticmethod
    def _valid_state_key(value: Any) -> bool:
        """Require a short, stable lower-snake-case semantic key."""
        return bool(
            isinstance(value, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value)
        )

    @staticmethod
    def _tool_simple_state(name: str, args: Dict[str, Any],
                           state: StoryState) -> Dict[str, Any]:
        """Apply one flat, semantically explicit state change."""
        if name == "update_character":
            character_name = args.get("name")
            description = args.get("description")
            if not isinstance(character_name, str) or not character_name.strip():
                return {"ok": False, "error": "name must be a non-empty string"}
            if not isinstance(description, str) or not description.strip():
                return {
                    "ok": False,
                    "error": "description must be a non-empty string",
                }
            character_name = " ".join(character_name.split())
            description = description.strip()
            key = LongStoryAgent._normalised_state_text(character_name)
            existing = next(
                (
                    entry for entry in state.character_states
                    if LongStoryAgent._normalised_state_text(entry.get("name"))
                    == key
                ),
                None,
            )
            if existing is None:
                ok, info = state.apply(
                    "character_states", "add", None,
                    {"name": character_name, "description": description},
                )
                state_id = (
                    state.character_states[-1]["id"]
                    if ok else None
                )
                action = "add"
            else:
                state_id = existing.get("id")
                ok, info = state.apply(
                    "character_states", "edit", state_id,
                    {
                        # Keep the casing and spelling already established in the state.
                        "name": existing.get("name") or character_name,
                        "description": description,
                    },
                )
                action = "edit"
            return {
                "ok": ok,
                "state_name": "character_states",
                "action": action,
                "state_id": state_id,
                "info": info,
            }

        if name in ("record_past_event", "add_requirement"):
            description = args.get("description")
            if not isinstance(description, str) or not description.strip():
                return {
                    "ok": False,
                    "error": "description must be a non-empty string",
                }
            description = description.strip()
            state_name = (
                "past_events"
                if name == "record_past_event" else "future_requirements"
            )
            bucket = state._bucket(state_name)
            key = LongStoryAgent._normalised_state_text(description)
            duplicate = next(
                (
                    entry for entry in bucket
                    if LongStoryAgent._normalised_state_text(
                        entry.get("description")
                    ) == key
                ),
                None,
            )
            if duplicate is not None:
                return {
                    "ok": True,
                    "state_name": state_name,
                    "action": "unchanged",
                    "state_id": duplicate.get("id"),
                    "info": "identical state already exists; no duplicate added",
                }
            ok, info = state.apply(
                state_name, "add", None, {"description": description}
            )
            return {
                "ok": ok,
                "state_name": state_name,
                "action": "add",
                "state_id": bucket[-1]["id"] if ok else None,
                "info": info,
            }

        if name == "resolve_requirement":
            state_id = args.get("state_id")
            if isinstance(state_id, bool) or not isinstance(state_id, int):
                return {"ok": False, "error": "state_id must be an integer"}
            ok, info = state.apply(
                "future_requirements", "delete", state_id, None
            )
            return {
                "ok": ok,
                "state_name": "future_requirements",
                "action": "delete",
                "state_id": state_id,
                "info": info,
            }

        return {"ok": False, "error": f"unknown simple state tool {name}"}

    @staticmethod
    def _requirement_similarity(query: str, candidate: str) -> float:
        """Conservative text similarity used to resolve requirements without numeric ids."""
        query_key = LongStoryAgent._normalised_state_text(query)
        candidate_key = LongStoryAgent._normalised_state_text(candidate)
        if not query_key or not candidate_key:
            return 0.0
        if query_key == candidate_key:
            return 1.0
        if (min(len(query_key), len(candidate_key)) >= 12
                and (query_key in candidate_key or candidate_key in query_key)):
            return 0.95
        query_tokens = set(re.findall(r"\w+", query_key, flags=re.UNICODE))
        candidate_tokens = set(
            re.findall(r"\w+", candidate_key, flags=re.UNICODE)
        )
        union = query_tokens | candidate_tokens
        jaccard = (
            len(query_tokens & candidate_tokens) / len(union)
            if union else 0.0
        )
        sequence = difflib.SequenceMatcher(
            None, query_key, candidate_key
        ).ratio()
        return max(jaccard, sequence)

    @staticmethod
    def _requirement_duplicate_similarity(query: str,
                                          candidate: str) -> float:
        """Looser than payoff matching; only blocks re-adding the same open loop."""
        base = LongStoryAgent._requirement_similarity(query, candidate)
        if base >= 0.65:
            return base

        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "before", "by",
            "for", "from", "has", "have", "he", "her", "his", "in", "is",
            "it", "later", "must", "of", "on", "or", "she", "that", "the",
            "their", "this", "to", "whether", "who", "will", "with",
        }

        def tokens(text: str) -> set:
            values = re.findall(r"[A-Za-z0-9]+", text.casefold())
            normalised = set()
            for value in values:
                if value in stopwords:
                    continue
                for suffix in ("ing", "ed", "es", "s", "al"):
                    if len(value) > len(suffix) + 3 and value.endswith(suffix):
                        value = value[:-len(suffix)]
                        break
                normalised.add(value)
            return normalised

        left = tokens(query)
        right = tokens(candidate)
        if not left or not right:
            return base
        overlap = len(left & right) / min(len(left), len(right))
        left_entities = {
            item.casefold()
            for item in re.findall(r"\b[A-Z][A-Za-z0-9]*\b", query)
            if item.casefold() not in stopwords
        }
        right_entities = {
            item.casefold()
            for item in re.findall(r"\b[A-Z][A-Za-z0-9]*\b", candidate)
            if item.casefold() not in stopwords
        }
        if left_entities & right_entities and overlap >= 0.30:
            return max(base, 0.90)
        if len(left & right) >= 4 and overlap >= 0.45:
            return max(base, 0.85)
        return max(base, overlap * 0.8)

    @staticmethod
    def _valid_future_payoff_description(description: str) -> bool:
        """Require an open loop to name an actionable future payoff, not a state or rule."""
        text = " ".join(description.casefold().split())
        action = re.search(
            r"\b(?:reveal|answer|return|confront|fulfill|repay|resolve|"
            r"discover|recover|deliver|choose|decide|meet|face|prevent|"
            r"rescue|explain|find|learn|expose|settle|complete|locate|free|"
            r"identify|determine|uncover|investigate|question|prove|verify|"
            r"save|stop|defeat|escape|reach|obtain|retrieve|repair|open)\w*\b",
            text,
        )
        chinese_action = re.search(
            r"揭示|回答|返回|面对|履行|偿还|解决|发现|找回|交付|选择|"
            r"决定|相遇|阻止|营救|解释|兑现|查明|寻找|识别|调查|证明|"
            r"拯救|击败|逃离|取得|取回|修复|打开",
            text,
        )
        return bool(action or chinese_action)

    @staticmethod
    def _tool_semantic_state(name: str, args: Dict[str, Any],
                             state: StoryState,
                             current_ch_id: int) -> Dict[str, Any]:
        """Apply the compact protocol: causal summaries, character snapshots, text references."""
        if name == "update_character":
            description = args.get("description")
            if (isinstance(description, str)
                    and semantic_state_text_too_long(
                        description, max_words=90, max_cjk_chars=140)):
                return {
                    "ok": False,
                    "error": (
                        "character snapshot exceeds 90 English words or 140 Han "
                        "characters; keep only current "
                        "goals, relationships, knowledge, possessions, and "
                        "physical/emotional condition, with no biography or recap"
                    ),
                }
            return LongStoryAgent._tool_simple_state(name, args, state)

        if name == "add_requirement":
            description = args.get("description")
            if not isinstance(description, str) or not description.strip():
                return {
                    "ok": False,
                    "error": "description must be a non-empty string",
                }
            description = " ".join(description.split())
            if semantic_state_text_too_long(
                    description, max_words=50, max_cjk_chars=80):
                return {
                    "ok": False,
                    "error": (
                        "future payoff exceeds 50 English words or 80 Han "
                        "characters; state only the single "
                        "specific later action"
                    ),
                }
            if not LongStoryAgent._valid_future_payoff_description(description):
                return {
                    "ok": True,
                    "state_name": "future_requirements",
                    "action": "ignored",
                    "state_id": None,
                    "info": (
                        "ignored because this is not a specific later payoff "
                        "action; ongoing rules, behaviors, threats, background "
                        "facts, and general character goals are not stored"
                    ),
                }
            same_chapter = next(
                (
                    entry for entry in state.future_requirements
                    if entry.get("introduced_chapter_id") == current_ch_id
                ),
                None,
            )
            if same_chapter is not None:
                return {
                    "ok": True,
                    "state_name": "future_requirements",
                    "action": "unchanged",
                    "state_id": same_chapter.get("id"),
                    "info": (
                        "this chapter already introduced one future payoff; "
                        "the per-chapter cap prevented another"
                    ),
                }
            if state.future_requirements:
                score, match = max(
                    (
                        (
                            LongStoryAgent._requirement_duplicate_similarity(
                                description, entry.get("description", "")
                            ),
                            entry,
                        )
                        for entry in state.future_requirements
                    ),
                    key=lambda item: item[0],
                )
                if score >= 0.65:
                    return {
                        "ok": True,
                        "state_name": "future_requirements",
                        "action": "unchanged",
                        "state_id": match.get("id"),
                        "match_score": round(score, 3),
                        "info": (
                            "a semantically equivalent open requirement already "
                            "exists; no duplicate added"
                        ),
                    }
            result = LongStoryAgent._tool_simple_state(
                name, {"description": description}, state
            )
            if result.get("ok") and result.get("action") == "add":
                state.future_requirements[-1][
                    "introduced_chapter_id"
                ] = current_ch_id
            return result

        if name == "record_chapter_summary":
            summary = args.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                return {
                    "ok": False,
                    "error": "summary must be a non-empty string",
                }
            summary = " ".join(summary.split())
            if semantic_state_text_too_long(
                    summary, max_words=60, max_cjk_chars=100):
                return {
                    "ok": False,
                    "error": (
                        "chapter summary exceeds 60 English words or 100 Han "
                        "characters; compress it to the "
                        "causal actions, discoveries, and state changes future "
                        "chapters need"
                    ),
                }
            existing = next(
                (
                    entry for entry in state.past_events
                    if entry.get("chapter_id") == current_ch_id
                ),
                None,
            )
            if existing is None:
                ok, info = state.apply(
                    "past_events", "add", None,
                    {"description": summary},
                )
                state_id = (
                    state.past_events[-1]["id"] if ok else None
                )
                if ok:
                    state.past_events[-1]["chapter_id"] = current_ch_id
                action = "add"
            else:
                state_id = existing.get("id")
                if (LongStoryAgent._normalised_state_text(
                        existing.get("description"))
                        == LongStoryAgent._normalised_state_text(summary)):
                    ok = True
                    info = "chapter summary already recorded"
                    action = "unchanged"
                else:
                    ok, info = state.apply(
                        "past_events", "edit", state_id,
                        {"description": summary},
                    )
                    action = "edit"
            return {
                "ok": ok,
                "state_name": "past_events",
                "action": action,
                "state_id": state_id,
                "chapter_id": current_ch_id,
                "word_count": count_words(summary),
                "info": info,
            }

        if name == "resolve_requirement_by_text":
            description = args.get("description")
            if not state.future_requirements:
                return {
                    "ok": True,
                    "state_name": "future_requirements",
                    "action": "unchanged",
                    "state_id": None,
                    "info": "no open requirement remains; treated as idempotent",
                }
            if not isinstance(description, str) or not description.strip():
                return {
                    "ok": False,
                    "error": "description must be a non-empty string",
                }
            ranked = sorted(
                (
                    (
                        LongStoryAgent._requirement_similarity(
                            description, entry.get("description", "")
                        ),
                        entry,
                    )
                    for entry in state.future_requirements
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            score, match = ranked[0]
            if score < 0.50:
                return {
                    "ok": True,
                    "state_name": "future_requirements",
                    "action": "unchanged",
                    "state_id": None,
                    "match_score": round(score, 3),
                    "info": (
                        "no sufficiently similar open requirement; nothing "
                        "deleted (safe idempotent result)"
                    ),
                }
            state_id = match.get("id")
            ok, info = state.apply(
                "future_requirements", "delete", state_id, None
            )
            return {
                "ok": ok,
                "state_name": "future_requirements",
                "action": "delete",
                "state_id": state_id,
                "matched_description": match.get("description"),
                "match_score": round(score, 3),
                "info": info,
            }

        return {"ok": False, "error": f"unknown semantic state tool {name}"}

    @staticmethod
    def _tool_narrative_ops_update(
            args: Dict[str, Any], state: StoryState,
            current_ch_id: int) -> Dict[str, Any]:
        """Apply the keyed four-array narrative update atomically."""
        field_names = (
            "upsert_character_state",
            "add_past_event",
            "add_future_requirement",
            "resolve_future_requirement",
        )
        # Some small-model tool parsers quote an otherwise valid JSON array.
        # Repair only strings that strict json.loads can unambiguously recover;
        # malformed/ambiguous strings still fail below and must be regenerated.
        normalised_args = dict(args)
        normalised_fields: List[str] = []
        for field_name in field_names:
            raw_value = normalised_args.get(field_name)
            parsed_value = _maybe_json(raw_value)
            if isinstance(raw_value, str) and isinstance(parsed_value, list):
                normalised_args[field_name] = parsed_value
                normalised_fields.append(field_name)
        args = normalised_args
        for field_name in field_names:
            if not isinstance(args.get(field_name), list):
                return {
                    "ok": False,
                    "error": (
                        f"{field_name} must be a native JSON array, not a "
                        "quoted/serialized array string; resend the update with "
                        "literal [...] or []"
                    ),
                    "applied_count": 0,
                    "operation_count": 0,
                    "results": [],
                }

        operation_count = sum(len(args[name]) for name in field_names)
        if operation_count == 0:
            return {
                "ok": True,
                "info": "No narrative-state changes requested",
                "applied_count": 0,
                "valid_count": 0,
                "operation_count": 0,
                "results": [],
            }

        trial = StoryState.from_dict(copy.deepcopy(state.to_dict()))
        trial._next_ids = dict(state._next_ids)
        results: List[Dict[str, Any]] = []

        for item in args["upsert_character_state"]:
            if not isinstance(item, dict):
                result = {
                    "ok": False,
                    "state_name": "character_states",
                    "action": "upsert",
                    "state_id": None,
                    "info": "character upsert must be an object",
                }
            else:
                result = LongStoryAgent._tool_simple_state(
                    "update_character", item, trial
                )
            results.append(result)

        for item in args["add_past_event"]:
            if not isinstance(item, dict):
                results.append({
                    "ok": False,
                    "state_name": "past_events",
                    "action": "add",
                    "state_id": None,
                    "info": "past event must be an object",
                })
                continue
            key = item.get("key")
            description = item.get("description")
            if not LongStoryAgent._valid_state_key(key):
                results.append({
                    "ok": False,
                    "state_name": "past_events",
                    "action": "add",
                    "state_id": None,
                    "info": "past-event key must be lower_snake_case",
                })
                continue
            if not isinstance(description, str) or not description.strip():
                results.append({
                    "ok": False,
                    "state_name": "past_events",
                    "action": "add",
                    "state_id": None,
                    "info": "past-event description must be non-empty",
                })
                continue
            description = " ".join(description.split())
            existing = next(
                (entry for entry in trial.past_events
                 if entry.get("key") == key),
                None,
            )
            if existing is not None:
                unchanged = (
                    LongStoryAgent._normalised_state_text(
                        existing.get("description")
                    ) == LongStoryAgent._normalised_state_text(description)
                )
                results.append({
                    "ok": unchanged,
                    "state_name": "past_events",
                    "action": "unchanged" if unchanged else "add",
                    "state_id": existing.get("id"),
                    "key": key,
                    "info": (
                        "identical key already exists; no duplicate added"
                        if unchanged else
                        "past-event key already exists with different content"
                    ),
                })
                continue
            ok, info = trial.apply(
                "past_events", "add", None,
                {"key": key, "description": description},
            )
            if ok:
                trial.past_events[-1]["chapter_id"] = current_ch_id
            results.append({
                "ok": ok,
                "state_name": "past_events",
                "action": "add",
                "state_id": trial.past_events[-1]["id"] if ok else None,
                "key": key,
                "info": info,
            })

        for item in args["add_future_requirement"]:
            if not isinstance(item, dict):
                results.append({
                    "ok": False,
                    "state_name": "future_requirements",
                    "action": "add",
                    "state_id": None,
                    "info": "future requirement must be an object",
                })
                continue
            key = item.get("key")
            description = item.get("description")
            if not LongStoryAgent._valid_state_key(key):
                results.append({
                    "ok": False,
                    "state_name": "future_requirements",
                    "action": "add",
                    "state_id": None,
                    "info": "future-requirement key must be lower_snake_case",
                })
                continue
            if not isinstance(description, str) or not description.strip():
                results.append({
                    "ok": False,
                    "state_name": "future_requirements",
                    "action": "add",
                    "state_id": None,
                    "info": "future-requirement description must be non-empty",
                })
                continue
            description = " ".join(description.split())
            existing = next(
                (entry for entry in trial.future_requirements
                 if entry.get("key") == key),
                None,
            )
            if existing is not None:
                unchanged = (
                    LongStoryAgent._normalised_state_text(
                        existing.get("description")
                    ) == LongStoryAgent._normalised_state_text(description)
                )
                results.append({
                    "ok": unchanged,
                    "state_name": "future_requirements",
                    "action": "unchanged" if unchanged else "add",
                    "state_id": existing.get("id"),
                    "key": key,
                    "info": (
                        "identical key already exists; no duplicate added"
                        if unchanged else
                        "future-requirement key already exists with different content"
                    ),
                })
                continue
            ok, info = trial.apply(
                "future_requirements", "add", None,
                {"key": key, "description": description},
            )
            if ok:
                trial.future_requirements[-1][
                    "introduced_chapter_id"
                ] = current_ch_id
            results.append({
                "ok": ok,
                "state_name": "future_requirements",
                "action": "add",
                "state_id": (
                    trial.future_requirements[-1]["id"] if ok else None
                ),
                "key": key,
                "info": info,
            })

        for key in args["resolve_future_requirement"]:
            if not LongStoryAgent._valid_state_key(key):
                results.append({
                    "ok": False,
                    "state_name": "future_requirements",
                    "action": "resolve",
                    "state_id": None,
                    "key": key,
                    "info": "resolved key must be lower_snake_case",
                })
                continue
            existing = next(
                (entry for entry in trial.future_requirements
                 if entry.get("key") == key),
                None,
            )
            if existing is None:
                results.append({
                    "ok": True,
                    "state_name": "future_requirements",
                    "action": "unchanged",
                    "state_id": None,
                    "key": key,
                    "info": "key is not open; treated as idempotent",
                })
                continue
            state_id = existing.get("id")
            ok, info = trial.apply(
                "future_requirements", "delete", state_id, None
            )
            results.append({
                "ok": ok,
                "state_name": "future_requirements",
                "action": "resolve",
                "state_id": state_id,
                "key": key,
                "info": info,
            })

        all_ok = len(results) == operation_count and all(
            result.get("ok") for result in results
        )
        applied_count = (
            sum(
                result.get("action") not in {"unchanged", "ignored"}
                for result in results
            )
            if all_ok else 0
        )
        if all_ok:
            state.character_states = trial.character_states
            state.past_events = trial.past_events
            state.future_requirements = trial.future_requirements
            state._next_ids = trial._next_ids
        return {
            "ok": all_ok,
            "applied_count": applied_count,
            "valid_count": sum(bool(result.get("ok")) for result in results),
            "operation_count": operation_count,
            "normalised_fields": normalised_fields,
            "results": results,
            "info": (
                f"Applied {applied_count}/{operation_count} narrative operations"
                + (
                    "; normalized serialized arrays: "
                    + ", ".join(normalised_fields)
                    if normalised_fields else ""
                )
                if all_ok else
                "Applied 0 because the narrative update is atomic and at least "
                "one operation was invalid"
            ),
        }

    @staticmethod
    def _normalise_update_operations(
            args: Dict[str, Any]
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Convert flat update arguments to internal operations, tolerating common misplacements."""
        operations: List[Dict[str, Any]] = []
        if "state_updates" in args:
            state_updates = args.get("state_updates")
            if not isinstance(state_updates, list):
                return None, "state_updates must be an array"
            seen_state_names = set()
            for state_update in state_updates:
                if not isinstance(state_update, dict):
                    operations.append({"_invalid_item": state_update})
                    continue
                state_name = state_update.get("state_name", "")
                if state_name in seen_state_names:
                    return None, (
                        f"duplicate state_name in state_updates: {state_name}"
                    )
                seen_state_names.add(state_name)
                for group_name, action in (
                    ("adds", "add"), ("deletes", "delete"),
                    ("edits", "edit"),
                ):
                    items = state_update.get(group_name)
                    if not isinstance(items, list):
                        return None, (
                            f"state_updates.{group_name} must be an array"
                        )
                    for item in items:
                        if not isinstance(item, dict):
                            operations.append({"_invalid_item": item})
                            continue
                        operations.append({
                            "state_name": state_name,
                            "state_id": item.get("state_id"),
                            "action": action,
                            "content": {
                                "name": item.get("name"),
                                "description": item.get("description", ""),
                            },
                        })
            return operations, None

        has_flat_groups = any(
            key in args for key in ("adds", "edits", "deletes")
        )

        if has_flat_groups:
            top_level_state_name = args.get("state_name")
            groups = (
                (("adds", "add"), ("deletes", "delete"),
                 ("edits", "edit"))
                if top_level_state_name is not None else
                (("adds", "add"), ("edits", "edit"),
                 ("deletes", "delete"))
            )
            for group_name, action in groups:
                items = args.get(group_name)
                if not isinstance(items, list):
                    return None, f"{group_name} must be an array"
                for item in items:
                    if not isinstance(item, dict):
                        operations.append({"_invalid_item": item})
                        continue
                    nested = _maybe_json(item.get("content"))
                    nested = nested if isinstance(nested, dict) else {}
                    state_name = (
                        top_level_state_name or item.get("state_name")
                        or nested.get("state_name") or ""
                    )
                    state_id = item.get("state_id")
                    if state_id is None:
                        state_id = nested.get("state_id", nested.get("id"))
                    name = item.get("name", nested.get("name"))
                    description = item.get(
                        "description", nested.get("description", "")
                    )
                    content = {
                        "name": name,
                        "description": description,
                    }
                    operations.append({
                        "state_name": state_name,
                        "state_id": state_id,
                        "action": action,
                        "content": content,
                    })
            return operations, None

        # Also accept the flat ``operations`` list (e.g. from resumed checkpoints or
        # non-strict servers).
        legacy = args.get("operations")
        if not isinstance(legacy, list):
            return None, "adds, edits, and deletes must be arrays"
        for item in legacy:
            if not isinstance(item, dict):
                operations.append({"_invalid_item": item})
                continue
            content = _maybe_json(item.get("content"))
            nested = content if isinstance(content, dict) else {}
            state_id = item.get("state_id", item.get("id"))
            if state_id is None:
                state_id = nested.get("state_id", nested.get("id"))
            operations.append({
                "state_name": (
                    item.get("state_name") or nested.get("state_name") or ""
                ),
                "state_id": state_id,
                "action": item.get("action") or nested.get("action") or "",
                "content": content,
            })
        return operations, None

    @staticmethod
    def _tool_update(args: Dict[str, Any],
                     state: StoryState) -> Dict[str, Any]:
        """Validate the whole batch on a state copy; commit atomically only if all succeed."""
        operations, normalise_error = (
            LongStoryAgent._normalise_update_operations(args)
        )
        if normalise_error:
            return {
                "ok": False,
                "error": normalise_error,
                "applied_count": 0,
                "results": [],
            }
        if not operations:
            return {
                "ok": True,
                "applied_count": 0,
                "valid_count": 0,
                "operation_count": 0,
                "results": [],
                "info": "No state changes requested",
            }

        # Deep-copy so validation cannot modify the live state; keep the real
        # auto-increment ids so from_dict never reuses ids after the max id is deleted.
        trial = StoryState.from_dict(copy.deepcopy(state.to_dict()))
        trial._next_ids = dict(state._next_ids)

        results: List[Dict[str, Any]] = []
        valid_count = 0
        for index, operation in enumerate(operations):
            if "_invalid_item" in operation:
                operation_result = {
                    "index": index,
                    "ok": False,
                    "info": "operation must be an object",
                }
            else:
                ok, info = trial.apply(
                    operation.get("state_name", ""),
                    operation.get("action", ""),
                    operation.get("state_id"),
                    operation.get("content"),
                )
                operation_result = {
                    "index": index,
                    "ok": ok,
                    "info": info,
                    "state_name": operation.get("state_name"),
                    "action": operation.get("action"),
                    "state_id": operation.get("state_id"),
                }
                if ok:
                    valid_count += 1
            results.append(operation_result)

        all_ok = valid_count == len(operations)
        if all_ok:
            state.character_states = trial.character_states
            state.past_events = trial.past_events
            state.future_requirements = trial.future_requirements
            state._next_ids = trial._next_ids
        applied_count = len(operations) if all_ok else 0
        return {
            "ok": all_ok,
            "applied_count": applied_count,
            "valid_count": valid_count,
            "operation_count": len(operations),
            "results": results,
            "info": (
                f"Applied {applied_count}/{len(operations)} state operations"
                if all_ok else
                f"Validated {valid_count}/{len(operations)} state operations; "
                "applied 0 because the batch is atomic"
            ),
        }

    @staticmethod
    def _tool_correct(args: Dict[str, Any], current_ch_id: int,
                      chapters: List[Dict[str, Any]],
                      current_draft: Optional[Dict[str, Any]],
                      correct_used: int) -> Dict[str, Any]:
        """Locate old text exactly or fuzzily and replace it; length is checked only by write."""
        if correct_used >= MAX_CORRECT_PER_CHAPTER:
            return {
                "ok": False,
                "error": (
                    f"correct budget exhausted "
                    f"({MAX_CORRECT_PER_CHAPTER}/chapter); proceed to the state update + DONE"
                ),
            }
        chapter_id = args.get("chapter_id")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        if not isinstance(chapter_id, int):
            return {"ok": False, "error": "chapter_id must be an integer"}
        if chapter_id > current_ch_id:
            return {"ok": False, "error": "chapter_id cannot be a future chapter"}

        if chapter_id == current_ch_id:
            if current_draft is None:
                return {
                    "ok": False,
                    "error": (
                        "current chapter has not been written yet; call write first"
                    ),
                }
            target = current_draft
        else:
            target = next(
                (item for item in chapters if item["id"] == chapter_id), None
            )
            if target is None:
                return {
                    "ok": False,
                    "error": f"chapter {chapter_id} not found",
                }

        content = target["content"]
        if content.count(old_text) > 1:
            return {
                "ok": False,
                "error": "old_text matches multiple times; make it more specific",
            }

        index = content.find(old_text)
        span: Optional[Tuple[int, int]] = None
        fuzzy = False
        if index != -1:
            span = (index, index + len(old_text))
        else:
            span = fuzzy_locate(content, old_text)
            fuzzy = span is not None
        if span is None:
            return {
                "ok": False,
                "error": (
                    "old_text not found in chapter content; do NOT retry the "
                    "same old_text — move on or use a shorter exact match"
                ),
            }

        corrected_content = (
            content[:span[0]] + new_text + content[span[1]:]
        )
        target["content"] = corrected_content
        return {
            "ok": True,
            "chapter_id": chapter_id,
            "fuzzy": fuzzy,
            "word_count": count_words(corrected_content),
        }

    @staticmethod
    def _tool_write(args: Dict[str, Any], current_ch_id: int,
                    write_succeeded: bool, chapter_target_wc: int, lang: str,
                    current_chapter_corrected: bool,
                    story_contract: str = "",
                    prior_chapters: Optional[
                        List[Dict[str, Any]]
                    ] = None) -> Dict[str, Any]:
        """Accept a full chapter draft and return actionable length feedback."""
        chapter_id = args.get("chapter_id")
        content = args.get("content", "")
        if chapter_id != current_ch_id:
            return {
                "ok": False,
                "error": f"write must target current chapter id={current_ch_id}",
            }
        if write_succeeded:
            return {
                "ok": False,
                "error": (
                    "write has already succeeded once for this chapter; "
                    "further whole-chapter writes are not allowed. Use correct "
                    "for any changes, then update the state + DONE"
                ),
            }
        if current_chapter_corrected:
            return {
                "ok": False,
                "error": (
                    "this chapter has already been corrected; a rewrite would "
                    "discard that correction. Keep the current text and use "
                    "correct for any further fixes, then update the state + DONE"
                ),
            }
        if not content or not isinstance(content, str):
            return {"ok": False, "error": "content must be a non-empty string"}

        if requires_dialogue_only(story_contract):
            # Qwen-4B sometimes emits valid em-dash utterances in one physical
            # JSON string line.  Canonicalize only terminal-punctuation
            # boundaries; this changes serialization, not prose content.
            content = re.sub(r"(?<=[.!?。！？…])\s+(?=—)", "\n", content)
            args["content"] = content

        actual_words = count_words(content)
        min_words = math.ceil(
            chapter_target_wc * (1 - WORD_COUNT_TOLERANCE)
        )
        max_words = math.floor(
            chapter_target_wc * (1 + WORD_COUNT_TOLERANCE)
        )
        deviation = (
            100 * (actual_words - chapter_target_wc) / chapter_target_wc
            if chapter_target_wc else 0
        )
        feedback = LongStoryAgent._word_count_feedback(
            actual_words, chapter_target_wc, deviation, lang
        )
        rounded_deviation = round(deviation, 1)
        within_tolerance = min_words <= actual_words <= max_words
        format_violations: List[str] = []
        if requires_dialogue_only(story_contract):
            dialogue_lines: List[str] = []
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                quoted = (
                    (stripped.startswith('—"') and stripped.endswith('"'))
                    or (stripped.startswith("—“") and stripped.endswith("”"))
                )
                dash_utterance = stripped.startswith("—")
                if dash_utterance:
                    dialogue_lines.append(stripped[1:].strip().strip('"“”'))
                narration = re.search(
                    r"\b(?:the stranger|the (?:first|second|third) guide|"
                    r"one of them|three (?:figures|voices)|the silence)\s+"
                    r"(?:had|felt|saw|looked|stood|opened|closed|moved|stepped|"
                    r"turned|nodded|asked|said|spoke|speaks|laughed|pressed|"
                    r"stretched|returned|deepened|became|cut|echoed)\b",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if not dash_utterance:
                    format_violations.append(
                        f"line does not begin as dialogue: {stripped[:120]}"
                    )
                elif not quoted and narration:
                    format_violations.append(
                        f"unquoted line contains narration: {stripped[:120]}"
                    )
                elif count_words(stripped) > 160:
                    format_violations.append(
                        f"utterance exceeds 160 words: {stripped[:120]}"
                    )
                if len(format_violations) >= 5:
                    break
            if not format_violations and len(dialogue_lines) >= 20:
                normalized_lines = [
                    " ".join(re.findall(
                        r"[a-z0-9]+(?:['’-][a-z0-9]+)*|[\u3400-\u9fff]",
                        line.casefold(),
                    ))
                    for line in dialogue_lines
                ]
                adjacent_duplicate = next(
                    (
                        line for left, right, line in zip(
                            normalized_lines,
                            normalized_lines[1:],
                            dialogue_lines[1:],
                        )
                        if left and left == right
                    ),
                    None,
                )
                unique_ratio = (
                    len(set(normalized_lines)) / len(normalized_lines)
                )
                mean_utterance_words = (
                    sum(count_words(line) for line in dialogue_lines)
                    / len(dialogue_lines)
                )
                if adjacent_duplicate is not None:
                    format_violations.append(
                        "adjacent utterance is repeated verbatim: "
                        + adjacent_duplicate[:120]
                    )
                if unique_ratio < 0.90:
                    format_violations.append(
                        f"only {unique_ratio:.1%} of dialogue lines are unique; "
                        "at least 90% must be unique"
                    )
                if mean_utterance_words < 6.0:
                    format_violations.append(
                        f"dialogue averages only {mean_utterance_words:.1f} "
                        "words per line; use substantive utterances averaging "
                        "at least 6 words"
                    )
                if lang == "en" and actual_words >= 500:
                    dialogue_words = re.findall(
                        r"[a-z0-9]+(?:['’-][a-z0-9]+)*",
                        " ".join(dialogue_lines).casefold(),
                    )
                    bigrams = list(zip(dialogue_words, dialogue_words[1:]))
                    distinct_bigram_ratio = (
                        len(set(bigrams)) / len(bigrams) if bigrams else 1.0
                    )
                    if distinct_bigram_ratio < 0.55:
                        format_violations.append(
                            f"only {distinct_bigram_ratio:.1%} of dialogue "
                            "bigrams are distinct; stop padding lines with "
                            "near-paraphrases and make each exchange advance "
                            "the scene"
                        )
        format_ok = not format_violations
        if not format_ok:
            feedback = (
                "强制纯对话格式不合格；每个非空行必须以破折号“—”开头，"
                "；推荐使用 —\"完整台词\"，也允许 —完整台词。每行只能是一条"
                "不超过 160 词、无说话者标签的台词。"
                if lang == "zh" else
                "Enforced dialogue-only format failed; every non-empty line "
                "must be one unattributed utterance beginning with an em dash. "
                "Prefer —\"complete utterance\"; bare —complete utterance is "
                "also valid. Keep each utterance at most 160 words. Specific "
                "violations: " + "; ".join(format_violations)
            )
        integrity_violations: List[str] = []
        # A provider can return syntactically valid tool-call JSON even when
        # the long prose argument itself was cut off at its token ceiling.
        # Reject that unmistakable failure in every state-update mode.
        ending = content.rstrip()
        while ending and ending[-1] in '\"\'”’)]}*_':
            ending = ending[:-1].rstrip()
        if not ending or ending[-1] not in ".!?…。！？":
            integrity_violations.append(
                "chapter ends mid-sentence or without terminal punctuation"
            )

        if story_contract:

            contract_text = " ".join(story_contract.casefold().split())
            opening = " ".join(content[:5000].casefold().split())
            if (
                current_ch_id == 0
                and ("teenage" in contract_text or "teenager" in contract_text)
                and not re.search(r"\b(?:12|twelve)\b", contract_text)
                and re.search(
                    r"\b(?:at|aged?)\s+(?:12|twelve)\b|"
                    r"\b(?:12|twelve)[-\s]year[-\s]old\b",
                    opening,
                )
            ):
                integrity_violations.append(
                    "opening contradicts the teenage-age contract by making "
                    "the protagonist twelve"
                )

            def substantial_sentences(value: str) -> List[str]:
                sentences = re.split(r"(?<=[.!?…])\s+|\n+", value)
                normalized = []
                for sentence in sentences:
                    item = " ".join(
                        re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", sentence.casefold())
                    )
                    if len(item.split()) >= 18:
                        normalized.append(item)
                return normalized

            current_sentences = substantial_sentences(content)
            repeated_here = {
                sentence for sentence in current_sentences
                if current_sentences.count(sentence) > 1
            }
            previous_sentences = {
                sentence
                for chapter in (prior_chapters or [])
                for sentence in substantial_sentences(
                    str(chapter.get("content", ""))
                )
            }
            repeated_before = set(current_sentences) & previous_sentences
            if repeated_here:
                integrity_violations.append(
                    "chapter repeats a long sentence verbatim within itself: "
                    + sorted(repeated_here)[0][:180]
                )
            if repeated_before:
                integrity_violations.append(
                    "chapter repeats a long sentence verbatim from an earlier "
                    "chapter: " + sorted(repeated_before)[0][:180]
                )
        integrity_ok = not integrity_violations
        if not integrity_ok:
            detail = "; ".join(integrity_violations)
            feedback = (
                f"章节完整性检查失败：{detail}。请重写为完整收束、无大段重复且"
                "严格遵守故事契约的章节。"
                if lang == "zh" else
                f"Chapter integrity check failed: {detail}. Rewrite the "
                "chapter with a complete ending, no long verbatim repetition, "
                "and literal compliance with the Story Contract."
            )
        return {
            "ok": within_tolerance and format_ok and integrity_ok,
            "chapter_id": chapter_id,
            "word_count": actual_words,
            "target_word_count": chapter_target_wc,
            "min_word_count": min_words,
            "max_word_count": max_words,
            "deviation_percent": rounded_deviation,
            "within_tolerance": within_tolerance,
            "format_error": not format_ok,
            "format_violations": format_violations,
            "integrity_error": not integrity_ok,
            "integrity_violations": integrity_violations,
            "info": (
                f"Wrote {actual_words} words (target: {chapter_target_wc}, "
                f"deviation: {rounded_deviation:+.1f}%). {feedback}"
            ),
        }

    @staticmethod
    def _word_count_feedback(actual_words: int, target_words: int,
                             deviation: float, lang: str) -> str:
        """Report the allowed total range so the model does not mistake a delta for the target."""
        if abs(deviation) <= WORD_COUNT_TOLERANCE_PERCENT:
            return "字数合格。" if lang == "zh" else "Word count acceptable."
        min_words = math.ceil(target_words * (1 - WORD_COUNT_TOLERANCE))
        max_words = math.floor(target_words * (1 + WORD_COUNT_TOLERANCE))
        if lang == "zh":
            return (
                "字数不合格；完整章节允许的总字数范围为 "
                f"{min_words}–{max_words} 字。"
            )
        return (
            "Word count rejected; the acceptable total length for the "
            f"complete chapter is {min_words}-{max_words} words."
        )

    async def _open_story_workspace(
            self, story_id: Any, prompt: str, lang: str,
            ckpt_dir: Optional[str], resume: bool,
    ) -> Tuple[StoryWorkspace, bool]:
        """Resume from a checkpoint if present; otherwise build a new outline and empty state."""
        fingerprint = checkpoint_fingerprint(
            prompt, self.word_count, self.state_update_mode,
            self.narrative_ops_guidance_placement,
        )
        checkpoint = (
            load_checkpoint(ckpt_dir, story_id, fingerprint, self.logger)
            if ckpt_dir and resume else None
        )

        if checkpoint:
            workspace = StoryWorkspace(
                story_id=story_id,
                prompt=prompt,
                lang=checkpoint.get("lang", lang),
                target_words=self.word_count,
                fingerprint=fingerprint,
                artifacts=checkpoint["artifacts"],
                outline=checkpoint["outline"],
                chapters=checkpoint["chapters"],
                state=StoryState.from_dict(checkpoint["state"]),
                outline_cache=checkpoint.get("outline_cache"),
            )
            self.logger.info(
                f"  resuming from checkpoint: {len(workspace.chapters)}/"
                f"{len(workspace.outline)} chapters already written"
            )
            return workspace, True

        outliner = OutlineAgent(
            client=self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            word_count=self.word_count,
            logger=self.logger,
            lang=lang,
            disable_thinking=self.disable_thinking,
        )

        def validate_cached_artifacts(value: Dict[str, Any]) -> None:
            if not isinstance(value, dict):
                raise ValueError("outline artifacts must be a JSON object")
            expected_stages = stage_count_for_words(self.word_count)
            if value.get("stages") != expected_stages:
                raise ValueError(
                    "outline cache stage mismatch: "
                    f"expected {expected_stages}, got {value.get('stages')!r}"
                )
            outline = value.get("outline")
            outliner._validate_chapter_outline(outline, prompt)

        artifacts, outline_cache_metadata = (
            await self.outline_cache.get_or_create(
                story_id=story_id,
                prompt=prompt,
                lang=lang,
                generator=lambda: outliner.generate(prompt),
                validator=validate_cached_artifacts,
            )
        )
        self.logger.info(
            "  outline cache: policy=%s status=%s key=%s",
            outline_cache_metadata.get("policy"),
            outline_cache_metadata.get("status"),
            (outline_cache_metadata.get("cache_key") or "-")[:16],
        )
        workspace = StoryWorkspace(
            story_id=story_id,
            prompt=prompt,
            lang=lang,
            target_words=self.word_count,
            fingerprint=fingerprint,
            artifacts=artifacts,
            outline=artifacts["outline"],
            chapters=[],
            state=StoryState(),
            outline_cache=outline_cache_metadata,
        )
        return workspace, False

    async def generate_story(self, story_id: int, prompt: str,
                             trace_base_dir: Optional[str] = None,
                             ckpt_dir: Optional[str] = None,
                             resume: bool = True) -> Dict[str, Any]:
        self.logger.info(f"{'='*40} Story id={story_id}")
        lang = detect_language(prompt)
        self.logger.info(f"  detected language: {lang}")

        # On auto-resume with the same writer, drop the failed attempt's in-memory
        # index; completed chapters come from the checkpoint and are reindexed lazily.
        self.search_index.reset()

        # Keep the per-story trace directory layout; detailed traces are not written.
        trace_dir = None
        if trace_base_dir:
            trace_dir = os.path.join(trace_base_dir, f"story_{story_id}")
            os.makedirs(trace_dir, exist_ok=True)

        workspace, resumed = await self._open_story_workspace(
            story_id, prompt, lang, ckpt_dir, resume
        )

        # Outline generation takes several LLM calls; persist once before chapter 0.
        if ckpt_dir and not resumed:
            save_checkpoint(
                ckpt_dir, story_id, workspace.checkpoint_data(), self.logger
            )

        # A checkpoint may stop at any chapter; skip by completed id, not list length.
        done_ids = {chapter["id"] for chapter in workspace.chapters}
        failed: Optional[Dict[str, Any]] = None

        for ch_info in workspace.outline:
            if ch_info["id"] in done_ids:
                continue
            if self.state_update_mode == "semantic":
                target_changes = rebalance_remaining_outline_word_targets(
                    workspace.outline, workspace.chapters,
                    workspace.target_words,
                )
                if ch_info["id"] in target_changes:
                    old_target, new_target = target_changes[ch_info["id"]]
                    self.logger.info(
                        f"  Chapter {ch_info['id']}: rebalanced target "
                        f"{old_target} -> {new_target} after accepted length "
                        "drift"
                    )
            try:
                written = await self._run_chapter_loop(
                    prompt, workspace.outline, ch_info, workspace.chapters,
                    workspace.state, workspace.lang,
                    story_id, trace_dir
                )
            except Exception as e:
                # Keep earlier chapters on failure; auto-resume reloads the clean checkpoint.
                self.logger.warning(
                    f"  Chapter {ch_info['id']} failed, pausing story "
                    f"{story_id} here ({len(workspace.chapters)}/"
                    f"{len(workspace.outline)} chapters kept): {e}"
                )
                failed = {"chapter_id": ch_info["id"], "error": str(e)}
                break
            workspace.chapters.append(written)
            if ckpt_dir:
                save_checkpoint(
                    ckpt_dir, story_id, workspace.checkpoint_data(), self.logger
                )

        if failed is None and ckpt_dir:
            # The story is about to be written to JSONL; the checkpoint is no longer needed.
            discard_checkpoint(ckpt_dir, story_id)
        return workspace.result_record(failed)


# =============================================================================
# 7. Batch processing, auto-resume, and CLI entry point
# =============================================================================


async def generate_story_with_auto_resume(
        writer: LongStoryAgent,
        story_id: Any,
        prompt: str,
        trace_base_dir: Optional[str],
        ckpt_dir: Optional[str],
        initial_resume: bool,
        max_auto_resumes: int,
        logger: logging.Logger,
        backoff_seconds: float = AUTO_RESUME_BACKOFF_SECONDS,
) -> Dict[str, Any]:
    """Generate one story; on failure, reload the checkpoint and resume."""
    last_record: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None

    for attempt in range(max_auto_resumes + 1):
        try:
            # --no-resume ignores only state from before startup; checkpoints created by
            # this process are used for auto-resume so finished work is not regenerated.
            resume = initial_resume or attempt > 0
            record = await writer.generate_story(
                story_id,
                prompt,
                trace_base_dir=trace_base_dir,
                ckpt_dir=ckpt_dir,
                resume=resume,
            )
            last_record = record
            last_error = None
            if record.get("complete"):
                return record

            failed = record.get("failed_chapter") or {}
            reason = failed.get("error", "unknown chapter failure")
        except Exception as e:
            last_error = e
            reason = str(e)

        if attempt >= max_auto_resumes:
            break

        delay = min(60.0, backoff_seconds * (2 ** attempt))
        logger.warning(
            f"Story {story_id} will automatically resume from checkpoint "
            f"(resume {attempt+1}/{max_auto_resumes}, in {delay:g}s): {reason}"
        )
        if delay > 0:
            await asyncio.sleep(delay)

    if last_record is not None:
        logger.error(
            f"Story {story_id} exhausted {max_auto_resumes} automatic "
            "checkpoint resumes; saving the latest incomplete result"
        )
        return last_record
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Story {story_id} produced no result")


def load_prompt_items(path: str, start: int = 0,
                      end: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read JSONL input and apply the same slicing semantics as the CLI."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line.strip()))
    stop = end if end is not None else len(items)
    return items[start:stop]


def completed_story_ids(output_path: str) -> Tuple[set, int]:
    """Read append-only output; for each id, the last record's complete flag wins."""
    if not os.path.exists(output_path):
        return set(), 0

    status: Dict[Any, bool] = {}
    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line.strip())
                status[record["id"]] = record.get("complete", True)
            except (json.JSONDecodeError, KeyError):
                continue
    completed = {story_id for story_id, ok in status.items() if ok}
    return completed, len(status) - len(completed)


def prepare_output_dirs(output_path: str) -> Tuple[str, str]:
    """Create output, trace, and checkpoint dirs; return the latter two paths."""
    output_dir = os.path.dirname(output_path) or "."
    trace_dir = os.path.join(output_dir, "traces")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return trace_dir, checkpoint_dir


async def main_async(args):
    """Story-level concurrency; chapters within a story run strictly in order."""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger("longstoryagent", f"logs/longstoryagent_{ts}.log")

    all_prompts = load_prompt_items(args.input)
    logger.info(f"Loaded {len(all_prompts)} prompts")
    end = args.end if args.end is not None else len(all_prompts)
    prompts = all_prompts[args.start:end]

    processed_ids = set()
    if not args.no_resume and os.path.exists(args.output):
        processed_ids, incomplete = completed_story_ids(args.output)
        logger.info(
            f"Resuming: {len(processed_ids)} already completed"
            + (f", {incomplete} incomplete will be retried" if incomplete else "")
        )

    to_process = [p for p in prompts if p["id"] not in processed_ids]
    logger.info(f"Generating {len(to_process)} stories")
    if not to_process:
        print("Nothing to generate.")
        return

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        max_retries=args.client_max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrent)
    file_lock = asyncio.Lock()
    trace_base_dir, ckpt_dir = prepare_output_dirs(args.output)

    pbar = atqdm(total=len(to_process), desc="Generating", unit="story")

    async def process_one(item):
        async with semaphore:
            try:
                writer = LongStoryAgent(
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
                    outline_cache_lock_timeout=
                    args.outline_cache_lock_timeout,
                )
                record = await generate_story_with_auto_resume(
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
                }
                if args.state_update_mode == "narrative_ops":
                    record["method"] = "NstAgent"
                    record["method_id"] = "nstagent"
                    record["protocol_version"] = "narrative_ops_prompt_v1"
                else:
                    record["method"] = "StructuredState"
                    record["method_id"] = args.state_update_mode
                record["language"] = item.get("language", "")
                total_words = sum(count_words(ch["content"]) for ch in record["story"])
                record["target_word_count"] = args.word_count
                record["word_count"] = total_words
                whole_min_words = math.ceil(
                    args.word_count * (1 - WORD_COUNT_TOLERANCE)
                )
                whole_max_words = math.floor(
                    args.word_count * (1 + WORD_COUNT_TOLERANCE)
                )
                whole_length_ok = (
                    whole_min_words <= total_words <= whole_max_words
                )
                record["whole_story_length_ok"] = whole_length_ok
                record["accepted_word_count_range"] = [
                    whole_min_words,
                    whole_max_words,
                ]
                # Whole-story word count is diagnostic only. Each chapter still enforces ±20%,
                # but accumulated chapter drift does not invalidate a completed story.
                record["whole_story_length_enforced"] = False
                if record.get("complete") and not whole_length_ok:
                    logger.warning(
                        "Story %s finished all chapters but whole length "
                        "%d is outside diagnostic range [%d, %d]; "
                        "the whole-story limit is not enforced",
                        item["id"],
                        total_words,
                        whole_min_words,
                        whole_max_words,
                    )
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
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="LongStoryAgent: structured-state + tools")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--api-base", default="https://www.autodl.art/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument("--word-count", type=int, default=DEFAULT_WORD_COUNT)
    parser.add_argument(
        "--chapter-token-control",
        choices=CHAPTER_TOKEN_CONTROL_MODES,
        default=DEFAULT_CHAPTER_TOKEN_CONTROL,
        help=(
            "fixed uses --max-tokens for every chapter turn; dynamic derives "
            "a smaller per-chapter budget from the chapter target"
        ),
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
    parser.add_argument(
        "--length-control-mode",
        choices=LENGTH_CONTROL_MODES,
        default=DEFAULT_LENGTH_CONTROL_MODE,
        help=(
            "chapter length-control mode: compact (default), "
            "tool_target (put the exact target in write description), or "
            "recent_failure_context (retain the latest rejected draft and "
            "append factual length feedback), feedback_only (compact the "
            "draft but append explicit failure feedback), or "
            "adaptive_recent_failure (retain only meaningfully improving "
            "rejected drafts)"
        ),
    )
    parser.add_argument(
        "--state-update-mode",
        choices=STATE_UPDATE_MODES,
        default=DEFAULT_STATE_UPDATE_MODE,
        help=(
            "batch uses the atomic adds/edits/deletes update schema "
            "(default); narrative_ops uses one atomic call with character "
            "upserts, keyed past events, keyed future requirements, and "
            "resolved requirement keys; "
            "state_grouped keeps one atomic call but nests "
            "adds/deletes/edits under each state_name; state_first promotes "
            "state_name before the "
            "adds/deletes/edits groups and updates one collection per call; "
            "simple and semantic use flat per-change state tools"
        ),
    )
    parser.add_argument(
        "--narrative-ops-guidance-placement",
        choices=NARRATIVE_OPS_GUIDANCE_PLACEMENTS,
        default=DEFAULT_NARRATIVE_OPS_GUIDANCE_PLACEMENT,
        help=(
            "narrative_ops guidance location: place detailed update rules "
            "only in the chapter prompt (default), only in the tool "
            "description, or in both surfaces"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--auto-resumes", type=int, default=DEFAULT_AUTO_RESUMES,
        help=("additional checkpoint resume attempts after a story pauses "
              f"(default: {DEFAULT_AUTO_RESUMES})"),
    )
    parser.add_argument(
        "--max-turns-per-chapter",
        type=int,
        default=MAX_TURNS_PER_CHAPTER,
        help="maximum model-tool turns per chapter",
    )
    parser.add_argument(
        "--request-attempts",
        type=int,
        default=3,
        help="total attempts per API request on errors (including the first)",
    )
    parser.add_argument(
        "--client-max-retries",
        type=int,
        default=2,
        help="extra connection/API retries inside the OpenAI client",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "send Qwen chat_template_kwargs.enable_thinking=false "
            "(off by default)"
        ),
    )
    parser.add_argument(
        "--outline-cache-dir",
        default=None,
        help=(
            "shared outline-artifact cache directory; required for every "
            "outline cache policy except off"
        ),
    )
    parser.add_argument(
        "--outline-cache-policy",
        choices=OUTLINE_CACHE_POLICIES,
        default=DEFAULT_OUTLINE_CACHE_POLICY,
        help=(
            "off samples normally; read-write loads or atomically populates "
            "missing entries; require refuses to sample when an entry is "
            "missing; refresh deliberately replaces the matching entry"
        ),
    )
    parser.add_argument(
        "--outline-cache-model-id",
        default=None,
        help=(
            "canonical model identity used in cache keys (for example "
            "qwen3.5-4b), independent of the endpoint's request model path"
        ),
    )
    parser.add_argument(
        "--outline-cache-lock-timeout",
        type=float,
        default=DEFAULT_OUTLINE_CACHE_LOCK_TIMEOUT,
        help="seconds before a per-entry generation lock is considered stale",
    )
    return parser


def print_run_config(args) -> None:
    """Print key settings before starting (never the API key)."""
    print("=" * 60)
    print("LongStoryAgent")
    print(f"  Model:        {args.model}")
    print(f"  Input:        {args.input}")
    print(f"  Output:       {args.output}")
    print(f"  Word count:   {args.word_count}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Concurrent:   {args.concurrent}")
    print(f"  Auto resumes: {args.auto_resumes}")
    print(f"  Chapter turns:{args.max_turns_per_chapter}")
    print(f"  API attempts: {args.request_attempts}")
    print(f"  Client retry: {args.client_max_retries}")
    print(f"  Length mode:  {args.length_control_mode}")
    print(f"  State update: {args.state_update_mode}")
    print(f"  Ops guidance: {args.narrative_ops_guidance_placement}")
    print(f"  Thinking:     {'disabled' if args.disable_thinking else 'default'}")
    print(f"  Outline cache:{args.outline_cache_policy}")
    if args.outline_cache_policy != "off":
        print(f"  Cache dir:    {os.path.abspath(args.outline_cache_dir)}")
        print(
            "  Cache model:  "
            f"{args.outline_cache_model_id or args.model}"
        )
    print(f"  Token control:{args.chapter_token_control}")
    if args.chapter_token_control == "dynamic":
        print(
            "  Dynamic ratio:"
            f" en={args.dynamic_token_ratio_en:g},"
            f" zh={args.dynamic_token_ratio_zh:g},"
            f" overhead={args.dynamic_token_overhead},"
            f" minimum={args.dynamic_token_minimum}"
        )
    print("=" * 60)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.auto_resumes < 0:
        parser.error("--auto-resumes must be >= 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.dynamic_token_ratio_en <= 0 or args.dynamic_token_ratio_zh <= 0:
        parser.error("dynamic token ratios must be > 0")
    if args.dynamic_token_overhead < 0:
        parser.error("--dynamic-token-overhead must be >= 0")
    if args.dynamic_token_minimum <= 0:
        parser.error("--dynamic-token-minimum must be > 0")
    if args.max_turns_per_chapter <= 0:
        parser.error("--max-turns-per-chapter must be > 0")
    if args.request_attempts <= 0:
        parser.error("--request-attempts must be > 0")
    if args.client_max_retries < 0:
        parser.error("--client-max-retries must be >= 0")
    if args.outline_cache_policy != "off" and not args.outline_cache_dir:
        parser.error(
            "--outline-cache-dir is required when outline cache policy is not off"
        )
    if args.outline_cache_lock_timeout <= 0:
        parser.error("--outline-cache-lock-timeout must be > 0")
    print_run_config(args)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
