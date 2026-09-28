#!/usr/bin/env python
# coding: utf-8
"""
Thin wrapper to run the original StoryWriter multi-agent pipeline
(`baselines/StoryWriter/agent_try.py`) on our prompts file.

The original entry-point `load_premise.py` reads a hard-coded
/HDD_DATA/.../moderate.jsonl and writes per-id JSONs into ./output/<run>/
sub-folders. This wrapper iterates the prompts JSONL given by --input and
calls the
SAME three pipeline functions of agent_try.py:
    event_generate -> event_extract -> story_generate
then collates per-prompt outputs into a single stories JSONL.

Patches applied directly inside `agent_try.py`:
  - `url`, `MODEL`, `API_KEY` taken from CLI/env vars and point to an available
    OpenAI-compatible generation endpoint.
  - `MessageRedact` uses the same endpoint instead of the upstream
    hard-coded gpt-4o-mini endpoint.
"""

import argparse, json, os, re, sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "StoryWriter"))
sys.path.insert(0, HERE)

from baseline_common import load_completed_ids, strict_length


def ensure_dirs(prefix):
    for sub in ("final_events", "final_process", "final_outlines", "final_story"):
        os.makedirs(os.path.join(prefix, sub), exist_ok=True)


def collate_story(story_json_path):
    """Rebuild the final story from the actual writer/critic turns.

    The upstream collator keeps only messages tagged ``StoryN.M{...}``; in
    long conversations the model occasionally omits the tag, and finished
    text would be silently dropped. The conversation starts with a seed and
    then alternates between writer text and critic verdicts. When the critic
    returns ``RE-WRITE``, its revision is the final version of that segment;
    otherwise the writer's draft is kept.
    """
    if not os.path.exists(story_json_path):
        return ""
    with open(story_json_path, encoding="utf-8") as stream:
        msgs = json.load(stream)
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ""

    def unwrap(text):
        text = (text or "").strip()
        text = re.sub(
            r"^\s*Story\s*\d+\.\d+\s*\{\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if text.endswith("}"):
            text = text[:-1].rstrip()
        text = re.sub(
            r"\n?\s*TERMINATE\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).rstrip()
        return text

    def tagged_segments(text):
        """Extract every StoryN.M segment from a response that may contain
        interleaved reasoning text."""
        text = text or ""
        markers = list(re.finditer(
            r"Story\s*\d+\.\d+\s*\{\s*",
            text,
            flags=re.IGNORECASE,
        ))
        segments = []
        for position, marker in enumerate(markers):
            end = (
                markers[position + 1].start()
                if position + 1 < len(markers)
                else len(text)
            )
            candidate = text[marker.end():end].strip()
            # Each segment ends at its own closing brace; ignore list/quote
            # wrappers added by the model.
            closing = candidate.rfind("}")
            if closing >= 0:
                candidate = candidate[:closing]
            candidate = candidate.strip(" \t\r\n'\"`,[]()")
            if candidate:
                segments.append(candidate)
        return segments

    parts = []
    # msgs[0] is the seed the critic sends to the writer; every two
    # messages after that form one round.
    for index in range(1, len(msgs), 2):
        writer_text = msgs[index] if isinstance(msgs[index], str) else ""
        critic_text = (
            msgs[index + 1]
            if index + 1 < len(msgs)
            and isinstance(msgs[index + 1], str)
            else ""
        )
        if writer_text.strip() == "TERMINATE":
            continue

        rewrite = re.search(
            r"^\s*\*\*RE-WRITE\*\*\s*:?\s*(.*?)"
            r"(?=\n?\s*Next sub-event\s*:|\Z)",
            critic_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        final_text = rewrite.group(1) if rewrite else writer_text
        extracted = tagged_segments(final_text)
        if extracted:
            parts.extend(extracted)
            continue
        final_text = unwrap(final_text)
        # Without tags, filter out only pure control messages; an
        # instruction echoed inside story text must not drop the whole story.
        if (
            final_text
            and not final_text.lower().startswith(
                "generate one complete sub-story"
            )
        ):
            parts.append(final_text)
    return "\n\n".join(parts)


def process(item, prefix, word_count, story_attempts):
    import agent_try  # imported lazily so env vars take effect
    sid = str(item["id"])
    # The outline agents only plan events. Putting the "write 10k words"
    # requirement into the premise conflicts with their system prompt: Qwen
    # tends to write the full story directly and keep continuing it between
    # the two outline agents, until the context exceeds the vLLM limit and
    # the request fails with HTTP 400.
    outline_premise = (
        f"{item['prompt']}\n\nPlan 5 to 10 major events, matching the original "
        "StoryWriter event range. Each major event will later be divided "
        "into three sub-events."
    )
    msg = agent_try.event_generate(outline_premise, sid, prefix)
    if msg == "None":
        return {
            "id": item["id"],
            "prompt": item["prompt"],
            "story": "",
            "method": "StoryWriter",
            "method_id": "storywriter",
            "target_word_count": word_count,
            "complete": False,
            "disable_thinking": os.environ.get(
                "STORYW_DISABLE_THINKING", "0"
            ) == "1",
            "error": "event_generate failed",
        }
    events = agent_try.event_extract(msg, sid, prefix)
    event_count = max(
        1,
        len(re.findall(r"Event\s*\d+\s*:", events, re.IGNORECASE)),
    )
    base_segment_words = max(250, round(word_count / (event_count * 3)))
    segment_words = base_segment_words
    attempts = []
    story = ""
    res = None
    complete = False
    actual = 0
    feedback = ""
    for attempt in range(1, story_attempts + 1):
        res = agent_try.story_generate(
            events,
            sid,
            prefix,
            target_words=word_count,
            segment_words=segment_words,
            length_feedback=feedback,
        )
        story = collate_story(
            os.path.join(prefix, "final_story", f"story{sid}.json")
        )
        length_ok, actual, minimum, maximum = strict_length(story, word_count)
        terminated = bool(isinstance(res, dict) and res.get("terminated"))
        complete = length_ok and terminated
        attempts.append({
            "attempt": attempt,
            "word_count": actual,
            "segment_target": segment_words,
            "terminated": terminated,
        })
        if complete:
            break
        ratio = word_count / max(actual, 1)
        segment_words = max(
            150,
            min(3000, round(segment_words * ratio)),
        )
        if actual < minimum:
            feedback = (
                f"The previous assembled draft was rejected because it "
                f"contained only {actual} words; the accepted whole-story "
                f"range is {minimum}-{maximum}, so it is still "
                f"{minimum - actual} words below the minimum. Regenerate "
                f"every complete sub-story using the revised per-segment "
                f"target. Preserve the full event sequence and ending, and "
                f"expand each event into concrete scenes with dialogue, "
                f"actions, reactions, and transitions; do not summarize or "
                f"compress events. "
            )
        else:
            feedback = (
                f"The previous assembled draft was rejected because it "
                f"contained {actual} words; the accepted whole-story range "
                f"is {minimum}-{maximum}, so it exceeds the maximum by "
                f"{actual - maximum} words. Regenerate every complete "
                f"sub-story using the revised per-segment target. Remove "
                f"repetition and non-causal digressions while preserving the "
                f"complete event sequence and ending. "
            )
    return {
        "id": item["id"],
        "prompt": item["prompt"],
        "language": item.get("language", ""),
        "story": story,
        "method": "StoryWriter",
        "method_id": "storywriter",
        "target_word_count": word_count,
        "word_count": actual,
        "complete": complete,
        "stage": res,
        "generation_attempts": attempts,
        "disable_thinking": os.environ.get(
            "STORYW_DISABLE_THINKING", "0"
        ) == "1",
        "error": None if complete else (
            f"story did not terminate within ±20% after "
            f"{story_attempts} attempts"
        ),
        "timestamp": datetime.now().isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--word-count", type=int, default=10000)
    ap.add_argument("--workdir", default="stories/storyw_5_10000_inter")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--story-attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument(
        "--api-base",
        default="http://127.0.0.1:8004/v1",
    )
    ap.add_argument(
        "--model",
        default="Qwen3.5-4B",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    ap.add_argument(
        "--client-max-retries",
        type=int,
        default=1,
        help="Extra OpenAI/AutoGen client retries after connection failures.",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Send Qwen chat_template_kwargs.enable_thinking=false on every "
            "AutoGen and helper request."
        ),
    )
    args = ap.parse_args()

    os.environ["STORYW_API_BASE"] = args.api_base
    os.environ["STORYW_MODEL"] = args.model
    os.environ["STORYW_API_KEY"] = args.api_key
    os.environ["STORYW_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["STORYW_MAX_RETRIES"] = str(args.client_max_retries)
    os.environ["STORYW_DISABLE_THINKING"] = (
        "1" if args.disable_thinking else "0"
    )
    print(
        f"STORYWRITER_RUN_START {datetime.now().isoformat()} "
        f"api_base={args.api_base} workers={args.workers} "
        f"max_tokens={args.max_tokens} "
        f"disable_thinking={args.disable_thinking}",
        flush=True,
    )

    items = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    end = args.end if args.end is not None else len(items)
    items = items[args.start:end]
    completed = (
        set()
        if args.no_resume
        else load_completed_ids(args.output, args.word_count)
    )
    items = [item for item in items if item["id"] not in completed]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ensure_dirs(args.workdir)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(
                    process,
                    it,
                    args.workdir,
                    args.word_count,
                    args.story_attempts,
                ): it
                for it in items}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                it = futs[fut]
                rec = {"id": it["id"], "prompt": it["prompt"], "story": "",
                       "error": str(e)}
            destination = (
                args.output
                if rec.get("complete")
                else args.output + ".failures.jsonl"
            )
            with open(destination, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[storyw] id={rec['id']} words={rec.get('word_count', 0)} err={rec.get('error','')}")
    print(f"[wrapper] final stories -> {args.output}")


if __name__ == "__main__":
    main()
