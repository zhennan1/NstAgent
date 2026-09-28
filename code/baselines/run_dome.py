#!/usr/bin/env python
# coding: utf-8
"""
Thin wrapper to run the DOME pipeline (Dynamic Hierarchical Outlining with
Memory-Enhancement) using the original repo's prompt templates, against an
OpenAI-compatible chat endpoint.

What is reused VERBATIM from the original DOME repository:
  - `pipline/promptwritting.py::PROMPT_TEMPLATE_WRITE`
    (storyline / detail_storyline / chapter_outline / write_en)
  - The 5-stage narrative theory from `pipline/1storyline.py`
  - The output-parser schema (`stage` + `storyline`) and the ChatPromptTemplate
    rendering mechanics from the original DHO.py / 1storyline.py.

What is replaced:
  - The original hard-coded remote Neo4j credentials are replaced with the
    local Neo4j 5 service, while preserving the same KG extraction, entity
    retrieval, path and temporal-pattern flow.
  - `chat()` calls the configured model through an OpenAI-compatible
    endpoint (e.g. vLLM) instead of the upstream DashScope API.
  - The DOME repo's `DHO.py` has several structural bugs (undefined
    `last_chapter_story` before use, wrong indexing into the chapter-outline
    regex, missing input modules).  The orchestrator below runs the paper's
    described flow using the verbatim repo prompts.
  - Original takes `(setting, character, plot_requirement)` from a CSV.  We
    have free-text prompts, so a tiny preprocessing step derives the three
    fields with an LLM call (prompt in English).
"""

import argparse, asyncio, json, logging, os, re, sys, threading
from datetime import datetime

from openai import AsyncOpenAI, OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DOME_DIR = os.path.join(
    HERE, "Generating-Long-form-Story-Using-Dynamic-Hierarchical-Outlining-with-Memory-Enhancement",
    "pipline")
sys.path.insert(0, DOME_DIR)

# Reused verbatim from original repo
from promptwritting import PROMPT_TEMPLATE_WRITE  # noqa: E402

# Neo4j-backed memory module that mirrors the original MEM.py API
import dome_memory  # noqa: E402
from baseline_common import load_completed_ids, strict_length  # noqa: E402
from generation_common import (  # noqa: E402
    chat_completion,
    count_words,
    detect_language,
    dynamic_max_tokens,
    retry_messages,
    update_failure_state,
    validate_generation,
    validation_dict,
)

STAGE_THEORY = (
    "1. Exposition: The story begins in a set setting, introducing the main "
    "characters and the setting of the story.\n"
    "2. Rising Action: An event or conflict is introduced; the main character "
    "leaves their comfort zone and faces challenges.\n"
    "3. Climax: The most tense and exciting moment; the main character faces "
    "their main conflict or challenge.\n"
    "4. Falling Action: After the climax, conflicts are resolved and tension "
    "decreases.\n"
    "5. Denouement or Resolution: Final outcome; fate of the characters is "
    "clarified and inner growth is revealed.\n"
)

# Structured output schema used by the original `1storyline.py`.
FORMAT_INSTRUCTIONS = (
    'Output a JSON list whose elements are objects with these two fields:\n'
    '  "stage": one of "Exposition", "Rising Action", "Climax", '
    '"Falling Action", "Resolution".\n'
    '  "storyline": a string describing how that stage unfolds at chapter '
    'granularity, enumerating chapters e.g. '
    '"Chapter 1: ...\\nChapter 2: ...".\n'
    "Return only the JSON list, no prose."
)

SYS_WRITER = ("You are a novelist who specializes in writing captivating, "
              "logically-rigorous long-form fiction.")
DOME_MAX_TOKENS = int(os.environ.get("DOME_MAX_TOKENS", "32768"))
DOME_REQUEST_ATTEMPTS = int(os.environ.get("DOME_REQUEST_ATTEMPTS", "3"))
DOME_DISABLE_THINKING = False


# -------- LLM chat ----------------------------------------------------------
async def chat(
    client,
    model,
    prompt,
    max_tokens=DOME_MAX_TOKENS,
    temperature=1.0,
    disable_thinking=False,
):
    for retry in range(DOME_REQUEST_ATTEMPTS):
        try:
            extra_body = {"top_k": 20}
            if disable_thinking and "deepseek" in model.lower():
                extra_body["thinking"] = {"type": "disabled"}
            if DOME_DISABLE_THINKING:
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": False,
                }
            request = dict(
                model=model,
                messages=[{"role": "system", "content": SYS_WRITER},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body=extra_body,
            )
            try:
                resp = await client.chat.completions.create(**request)
            except Exception as error:
                # Some unified-gateway Luna routes accept vLLM's optional
                # ``top_k`` extension while others reject that one field with
                # HTTP 400.  Retry the identical request once without the
                # unsupported optional field; prompts, sampling parameters,
                # token limits, and the DOME algorithm remain unchanged.
                if "Unknown parameter: 'top_k'" not in str(error):
                    raise
                request["extra_body"] = dict(extra_body)
                request["extra_body"].pop("top_k", None)
                resp = await client.chat.completions.create(**request)
            choice = resp.choices[0]
            completion_tokens = (
                getattr(resp.usage, "completion_tokens", None)
                if getattr(resp, "usage", None) is not None
                else None
            )
            if (
                choice.finish_reason != "stop"
                or (
                    isinstance(completion_tokens, int)
                    and completion_tokens >= max_tokens
                )
            ):
                raise RuntimeError(
                    "incomplete DOME planning/retrieval response: "
                    f"finish_reason={choice.finish_reason!r}, "
                    f"tokens={completion_tokens}/{max_tokens}"
                )
            m = choice.message
            c = m.content or ""
            if not c:
                d = m.model_dump()
                r = d.get("reasoning", "") or ""
                c = r.split("</think>", 1)[-1] if "</think>" in r else r
            return c.strip()
        except Exception:
            if retry == DOME_REQUEST_ATTEMPTS - 1:
                raise
            await asyncio.sleep(5 * (retry + 1))


def strip_code_fence(t):
    t = t.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return t


def parse_json_list(t):
    """Robust to Qwen 4B's character-level failures: unescaped quotes inside
    storyline strings often produce `...behind by strangers.}, {"stage": ...`
    (missing closing `"`). Try strict json first, then patch missing quotes
    before `}, {` / `}\\s*]` boundaries and retry."""
    t = strip_code_fence(t)
    m = re.search(r"\[[\s\S]*\]", t)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Heuristic 1: insert a missing `"` before `}, {` / `}]` when preceded by
    # a non-quote, non-whitespace character (i.e., the string wasn't closed).
    patched = re.sub(r'([^"\s])(\s*\},\s*\{)', r'\1"\2', raw)
    patched = re.sub(r'([^"\s])(\s*\}\s*\])', r'\1"\2', patched)
    try:
        return json.loads(patched)
    except Exception:
        pass
    # Heuristic 2: per-object salvage — extract each {...} block and try
    # individually, dropping ones that still fail.
    objs = []
    for chunk in re.findall(r"\{[^{}]*\}", patched):
        try:
            objs.append(json.loads(chunk))
        except Exception:
            try:
                # Last-ditch: pull stage + storyline by regex
                s_m = re.search(r'"stage"\s*:\s*"([^"]+)"', chunk)
                t_m = re.search(r'"storyline"\s*:\s*"(.+?)(?:"\s*\}|$)',
                                chunk, re.DOTALL)
                if s_m and t_m:
                    objs.append({"stage": s_m.group(1),
                                 "storyline": t_m.group(1)})
            except Exception:
                pass
    return objs or None


# -------- Memory (Neo4j-backed, faithful to original MEM.py) --------------
class MemoryStore:
    """Neo4j-backed analogue of the paper's MEM module. Uses
    `dome_memory.set_initial / set_history / find_relevant_info`, which run
    KG extraction, entity embedding, neighbor/path retrieval, relevance
    scoring and schema-grouped temporal pattern mining -- the same flow as
    the original `MEM.py`. Per-title isolation (every Neo4j node carries a
    `title` property) lets multiple samples run concurrently."""

    def __init__(self, sync_client, model, title):
        self.client = sync_client
        self.model = model
        self.title = title
        self.initial = {}
        self.history = []  # cache for prompt rendering, doesn't replace KG
        self.memory_llm_calls = 0
        self.memory_llm_failures = 0
        self.memory_cache_hits = 0
        self._memory_stats_lock = threading.Lock()
        self._retrieval_cache = {}

    def _llm_chat_sync(self, prompt):
        # KG construction, graph-to-text, relevance scoring and temporal
        # patterns are memory tool calls with strict output formats. DeepSeek's
        # default reasoning can burn thousands of tokens on these short tasks,
        # so thinking is disabled for this endpoint and only the tool result is
        # returned. Creative planning and prose generation still use the
        # regular chat above with reasoning enabled.
        # When a ~1k-word chapter is extracted sentence by sentence and verb by
        # verb with the original template, the triples alone can exceed 2k
        # tokens; these calls therefore share the 16k budget used by the other
        # 10k-setting calls, and finish_reason / completion-token truncation is
        # still checked.
        memory_max_tokens = DOME_MAX_TOKENS
        for retry in range(DOME_REQUEST_ATTEMPTS):
            try:
                with self._memory_stats_lock:
                    self.memory_llm_calls += 1
                request_options = {}
                if "deepseek" in self.model.lower():
                    request_options["extra_body"] = {
                        "thinking": {"type": "disabled"}
                    }
                elif DOME_DISABLE_THINKING:
                    request_options["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        }
                    }
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system",
                               "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}],
                    max_tokens=memory_max_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    **request_options,
                )
                choice = resp.choices[0]
                completion_tokens = (
                    getattr(resp.usage, "completion_tokens", None)
                    if getattr(resp, "usage", None) is not None
                    else None
                )
                if (
                    choice.finish_reason != "stop"
                    or (
                        isinstance(completion_tokens, int)
                        and completion_tokens >= memory_max_tokens
                    )
                ):
                    raise RuntimeError(
                        "incomplete DOME memory response: "
                        f"finish_reason={choice.finish_reason!r}, "
                        f"tokens={completion_tokens}/{memory_max_tokens}"
                    )
                m = choice.message
                c = m.content or ""
                if not c:
                    d = m.model_dump()
                    r = d.get("reasoning", "") or ""
                    c = r.split("</think>", 1)[-1] if "</think>" in r else r
                return c.strip()
            except Exception as error:
                with self._memory_stats_lock:
                    self.memory_llm_failures += 1
                print(
                    f"[dome-memory] title={self.title} "
                    f"retry={retry + 1}/{DOME_REQUEST_ATTEMPTS} error={error}",
                    flush=True,
                )
                if retry == DOME_REQUEST_ATTEMPTS - 1:
                    return ""
        return ""

    def set_initial(self, triple, title):
        setting, character, outline = triple
        self.initial = {"setting": setting, "character": character,
                        "outline": outline, "title": title}
        self.history = []
        dome_memory.reset_title(title)
        await_sync = lambda fn: fn()
        # Run synchronously in current thread (called once, before async loop)
        dome_memory.set_initial([setting, character, outline], title,
                                self._llm_chat_sync)

    def set_history(self, text, title, step):
        self.history.append((step, text))
        # Offload KG extraction + Neo4j writes to a worker thread later
        dome_memory.set_history(text, title, step, self._llm_chat_sync)

    async def find_relevant_info(self, current_outline, step, title):
        # Retrieval for the same KG version, outline and step is an idempotent
        # tool query. The pipeline queries three times in a row before writing
        # back a new chapter; the first complete result is cached, and the key
        # changes automatically with the KG/step so stale memory is never reused.
        cache_key = (current_outline, step, title, len(self.history))
        if cache_key in self._retrieval_cache:
            self.memory_cache_hits += 1
            return self._retrieval_cache[cache_key]
        # Heavy work (Neo4j + LLM scoring) -- run in thread to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, dome_memory.find_relevant_info,
            current_outline, step, title, self._llm_chat_sync)
        self._retrieval_cache[cache_key] = result
        return result


# -------- Derive (setting, character, plot_requirement) --------------------
async def derive_sco(client, model, premise, word_count):
    prompt = (
        "Given the following story prompt, derive THREE inputs needed by a "
        "novel-planner: a Setting description, a Character introduction list, "
        "and a Plot Requirement summary. Return strict JSON with keys "
        '"setting", "character", "outline". '
        f"The novel target length is about {word_count} words; make the "
        "plot requirement proportional to that scope.\n\n"
        f"Prompt: {premise}"
    )
    t = await chat(
        client,
        model,
        prompt,
        max_tokens=DOME_MAX_TOKENS,
        temperature=0.7,
        disable_thinking=True,
    )
    t = strip_code_fence(t)
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            d = json.loads(m.group(0))
            return d.get("setting", ""), d.get("character", ""), d.get("outline", "")
        except Exception:
            pass
    return "", "", premise


# -------- Original prompts (minimal render without langchain) --------------
def render(template, **kw):
    """Lightweight alternative to langchain ChatPromptTemplate.format_messages
    used by the original code. The templates contain `{...}` placeholders."""
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# -------- Pipeline ---------------------------------------------------------
async def gen_storyline(client, model, setting, character, outline,
                        word_count):
    tmpl = PROMPT_TEMPLATE_WRITE["storyline"]
    p = render(tmpl, theory=STAGE_THEORY, setting=setting, character=character,
               outline=outline, format=FORMAT_INSTRUCTIONS)
    p += (f"\nThe whole novel should target about {word_count} words; "
          "distribute roughly across the five stages so the totals add up.")
    raw = await chat(
        client,
        model,
        p,
        max_tokens=DOME_MAX_TOKENS,
        disable_thinking=True,
    )
    return parse_json_list(raw), raw


async def gen_detail_outline(client, model, rough_outline, history,
                             previous_detail, num):
    tmpl = PROMPT_TEMPLATE_WRITE["detail_storyline"]
    fmt = ("- Outline of chapter N:\n  <plot beats>\n"
           "- Outline of chapter N+1:\n  <plot beats>\n")
    p = render(tmpl, rough_outline=rough_outline, history=history,
               detail_outline=previous_detail, format=fmt, num=num)
    return await chat(
        client,
        model,
        p,
        max_tokens=DOME_MAX_TOKENS,
        disable_thinking=True,
    )


async def gen_chapter_outline(client, model, volume_outline, last_chapter,
                              history):
    tmpl = PROMPT_TEMPLATE_WRITE["chapter_outline"]
    p = render(tmpl, volume_outline=volume_outline,
               last_chapter=last_chapter or "(none, this is the first chapter)",
               history=history or "(none yet)")
    return await chat(
        client,
        model,
        p,
        max_tokens=DOME_MAX_TOKENS,
        disable_thinking=True,
    )


async def write_chapter(client, model, volume_outline, chapter_outline,
                        last_chapter, history, chapter_words, control):
    tmpl = PROMPT_TEMPLATE_WRITE["write_en"]
    p = render(tmpl, volume_outline=volume_outline,
               chapter_outline=chapter_outline,
               last_chapter=last_chapter or "(none)",
               history=history or "(none yet)")
    minimum = int(chapter_words * 0.8 + 0.999999)
    maximum = int(chapter_words * 1.2)
    p += (
        f"\n\nThe complete chapter must contain {minimum}-{maximum} words "
        f"(target {chapter_words}). Return only the complete chapter prose."
    )
    base_messages = [
        {"role": "system", "content": SYS_WRITER},
        {"role": "user", "content": p},
    ]
    messages = base_messages
    failure = None
    language = control["language"]
    if control["token_control"] == "dynamic":
        budget = dynamic_max_tokens(
            chapter_words,
            language,
            control["max_tokens"],
            overhead=control["dynamic_token_overhead"],
        )
    else:
        budget = control["max_tokens"]
    attempts = []
    for attempt in range(1, control["max_length_attempts"] + 1):
        response = await chat_completion(
            client,
            model,
            messages,
            max_tokens=budget,
            temperature=1.0,
            logger=control["logger"],
            disable_thinking=control["disable_thinking"],
        )
        validation = validate_generation(
            response.content,
            response,
            chapter_words,
        )
        attempts.append({
            "attempt": attempt,
            "requested_max_tokens": budget,
            "completion_tokens": response.completion_tokens,
            "validation": validation_dict(validation),
        })
        print(
            "[dome-write] "
            f"target={chapter_words} attempt={attempt} "
            f"words={validation.actual_words} "
            f"finish={response.finish_reason!r} "
            f"accepted={validation.ok}",
            flush=True,
        )
        if validation.ok:
            return response.content, attempts
        failure = update_failure_state(
            control["length_control_mode"],
            failure,
            response.content,
            validation,
        )
        messages = retry_messages(
            base_messages,
            failure,
            language,
        )
        if (
            control["token_control"] == "dynamic"
            and response.token_limit_reached
            and validation.actual_words <= validation.max_words
        ):
            budget = min(control["max_tokens"], budget + 128)
    raise RuntimeError(
        f"chapter failed ±20%/finish validation after "
        f"{control['max_length_attempts']} attempts"
    )


def extract_chapter_outlines(text):
    """Mirrors `re_chapter_outline` in DHO.py, but robust.
    The template asks for two sub-outlines labeled 'Chapter Outline 1/2'."""
    m = re.findall(
        r"(?:Chapter Outline\s*\d+|Outline of chapter\s*\d+)\s*[:\-]\s*(.+?)"
        r"(?=(?:Chapter Outline\s*\d+|Outline of chapter\s*\d+)\s*[:\-]|\Z)",
        text, re.DOTALL | re.IGNORECASE)
    outs = [x.strip() for x in m if x.strip()]
    return outs or [text.strip()]


async def dome_one(client, sync_client, model, item, word_count, inter_dir,
                   control):
    sid = item["id"]
    run_namespace = os.environ.get("DOME_RUN_NAMESPACE", "default")
    title = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        f"dome_{run_namespace}_{model}_{word_count}_{sid}",
    )[-180:]
    premise = item["prompt"]

    mem = MemoryStore(sync_client, model, title)

    setting, character, outline = await derive_sco(client, model, premise,
                                                   word_count)
    print(f"[dome-progress] id={sid} derived setting/characters", flush=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, mem.set_initial,
                               [setting, character, outline], title)
    print(f"[dome-progress] id={sid} initialized KG", flush=True)

    storyline_json, storyline_raw = await gen_storyline(
        client, model, setting, character, outline, word_count)
    if not storyline_json:
        return {"id": sid, "prompt": premise, "story": "",
                "error": "storyline parse failed",
                "storyline_raw": storyline_raw}
    print(
        f"[dome-progress] id={sid} planned {len(storyline_json)} stages",
        flush=True,
    )

    texts, detail_outs, near_info = [], [], ""
    chapter_stats = []
    step = 1
    for stage_index, stage_obj in enumerate(storyline_json):
        volume_outline = stage_obj.get("storyline", "")
        print(
            f"[dome-progress] id={sid} stage={stage_index + 1}/"
            f"{len(storyline_json)} retrieval/detail",
            flush=True,
        )
        # Refine the rough stage outline into chapter-level detail (num=1)
        history = await mem.find_relevant_info(volume_outline, step, title)
        detail = await gen_detail_outline(
            client, model, volume_outline, history, "(none)", 1)
        detail_outs.append(detail)

        # Chapter outlines (two at a time, per original 'chapter_outline')
        history = await mem.find_relevant_info(volume_outline, step, title)
        co_raw = await gen_chapter_outline(
            client, model, volume_outline, near_info, history)
        chapter_outlines = extract_chapter_outlines(co_raw)
        print(
            f"[dome-progress] id={sid} stage={stage_index + 1} "
            f"chapter_units={len(chapter_outlines)}",
            flush=True,
        )

        for outline_index, co in enumerate(chapter_outlines):
            # The original template is expected to yield two chapter outlines
            # per remaining stage. Distribute the remaining budget by the words
            # written so far, so a mismatch between the rough-outline chapter
            # count and the parsed count does not skew lengths.
            remaining_current = len(chapter_outlines) - outline_index
            remaining_stages = len(storyline_json) - stage_index - 1
            remaining_units = max(
                1,
                remaining_current + 2 * remaining_stages,
            )
            remaining_words = max(500, word_count - count_words(
                "\n\n".join(texts)
            ))
            chapter_words = max(500, round(remaining_words / remaining_units))
            history = await mem.find_relevant_info(volume_outline, step, title)
            text, attempts = await write_chapter(
                client, model, volume_outline, co, near_info, history,
                chapter_words, control)
            texts.append(text)
            chapter_stats.append({
                "id": step - 1,
                "target_word_count": chapter_words,
                "word_count": count_words(text),
                "attempts": attempts,
            })
            print(
                f"[dome-progress] id={sid} chapter={step} "
                f"words={count_words(text)}/{chapter_words}",
                flush=True,
            )
            await loop.run_in_executor(None, mem.set_history, text, title, step)
            print(
                f"[dome-progress] id={sid} chapter={step} KG updated",
                flush=True,
            )
            near_info = text[-1800:]
            step += 1

    story = "\n\n".join(texts)

    if inter_dir:
        os.makedirs(inter_dir, exist_ok=True)
        with open(os.path.join(inter_dir, f"{sid}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"id": sid, "prompt": premise,
                       "setting": setting, "character": character,
                       "outline": outline,
                       "storyline": storyline_json,
                       "detail_outlines": detail_outs,
                       "num_chapters": len(texts),
                       "memory_llm_calls": mem.memory_llm_calls,
                       "memory_llm_failures": mem.memory_llm_failures,
                       "memory_cache_hits": mem.memory_cache_hits},
                      f, ensure_ascii=False, indent=2)

    complete, actual_words, minimum, maximum = strict_length(story, word_count)
    return {
        "id": sid, "prompt": premise,
        "language": item.get("language", ""),
        "story": story,
        "method": "DOME",
        "method_id": "dome",
        "target_word_count": word_count,
        "word_count": actual_words,
        "complete": complete,
        "num_chapters": len(texts),
        "chapters": chapter_stats,
        "memory_llm_calls": mem.memory_llm_calls,
        "memory_llm_failures": mem.memory_llm_failures,
        "memory_cache_hits": mem.memory_cache_hits,
        "accepted_range": [minimum, maximum],
        "timestamp": datetime.now().isoformat(),
    }


async def main_async(args):
    import concurrent.futures
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=64))
    prompts = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    end = args.end if args.end is not None else len(prompts)
    prompts = prompts[args.start:end]
    done = (
        set()
        if args.no_resume
        else load_completed_ids(args.output, args.word_count)
    )
    todo = [p for p in prompts if p["id"] not in done]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # DOME's KG pipeline issues many short sequential requests. Give each
    # request a finite timeout and let the stage-level retries above take over,
    # so a single dropped connection cannot stall the batch on one triple.
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=180.0,
        max_retries=0,
    )
    sync_client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=180.0,
        max_retries=0,
    )
    sem = asyncio.Semaphore(args.concurrent)
    lock = asyncio.Lock()
    logger = logging.getLogger("dome")
    control = {
        "language": "en",
        "token_control": args.token_control,
        "max_tokens": args.max_tokens,
        "dynamic_token_overhead": args.dynamic_token_overhead,
        "length_control_mode": args.length_control_mode,
        "max_length_attempts": args.max_length_attempts,
        "logger": logger,
        "disable_thinking": args.disable_thinking,
    }

    async def worker(item):
        async with sem:
            rec = None
            error = None
            item_control = dict(control)
            item_control["language"] = (
                item.get("language") or detect_language(item["prompt"])
            )
            for story_attempt in range(1, args.story_attempts + 1):
                try:
                    candidate = await dome_one(
                        client,
                        sync_client,
                        args.model,
                        item,
                        args.word_count,
                        args.intermediate_dir,
                        item_control,
                    )
                    candidate["story_attempt"] = story_attempt
                    if candidate["complete"]:
                        rec = candidate
                        error = None
                        break
                    error = RuntimeError(
                        f"whole story length {candidate['word_count']} outside "
                        f"{candidate['accepted_range']}"
                    )
                except Exception as current_error:
                    error = current_error
            if rec is None:
                rec = {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "story": "",
                    "method": "DOME",
                    "method_id": "dome",
                    "target_word_count": args.word_count,
                    "complete": False,
                    "error": str(error),
                }
            async with lock:
                destination = (
                    args.output
                    if rec.get("complete")
                    else args.output + ".failures.jsonl"
                )
                with open(destination, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[dome] id={rec['id']} words={rec.get('word_count',0)} "
                  f"err={rec.get('error','')}")

    await asyncio.gather(*[worker(p) for p in todo])
    print(f"Done -> {args.output}")


def main():
    global DOME_DISABLE_THINKING
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen3.5-4B")
    ap.add_argument("--api-base", default="http://127.0.0.1:8004/v1")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    ap.add_argument("--word-count", type=int, default=10000)
    ap.add_argument("--concurrent", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument(
        "--token-control",
        choices=("fixed", "dynamic"),
        default="dynamic",
    )
    ap.add_argument("--dynamic-token-overhead", type=int, default=1024)
    ap.add_argument(
        "--length-control-mode",
        choices=(
            "feedback_only",
            "recent_failure_context",
            "adaptive_recent_failure",
        ),
        default="adaptive_recent_failure",
    )
    ap.add_argument("--max-length-attempts", type=int, default=20)
    ap.add_argument("--story-attempts", type=int, default=3)
    ap.add_argument("--intermediate-dir", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Send Qwen chat_template_kwargs.enable_thinking=false on every "
            "planning, memory, and chapter-writing request."
        ),
    )
    args = ap.parse_args()
    DOME_DISABLE_THINKING = args.disable_thinking
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
