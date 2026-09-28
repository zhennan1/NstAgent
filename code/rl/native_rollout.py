"""Native chapter execution with immutable per-call training contexts.

This module does not flatten non-append-only histories into a fake PPO sequence.
The caller must train every returned turn against its own recorded input.
"""
import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable

from longstoryagent_dynamic_tokens import LongStoryAgent, StoryState
from narrative_config import PROFILE, MAX_TOKENS, verify_snapshot


@dataclass(frozen=True)
class TurnRecord:
    prompt_ids: tuple
    response_ids: tuple
    response_logprobs: tuple
    requested_max_tokens: int

    def as_dict(self):
        return dict(prompt_ids=list(self.prompt_ids), response_ids=list(self.response_ids),
                    response_logprobs=list(self.response_logprobs),
                    requested_max_tokens=self.requested_max_tokens)


class NativeChapterRollout(LongStoryAgent):
    """Reuse the actual production loop, including compaction, feedback and DONE."""
    def __init__(self, snapshot: dict, generate: Callable):
        verify_snapshot(snapshot)
        super().__init__(None, 'Qwen3.5-4B', MAX_TOKENS, .7, snapshot['length'],
                         logging.getLogger(__name__), **PROFILE)
        self.snapshot = copy.deepcopy(snapshot)
        self.generate = generate
        self.turns = []
        self.failed_runtime = None

    async def _llm(self, messages, use_tools=False, tools=None, max_tokens=None):
        budget = self.max_tokens if max_tokens is None else max_tokens
        # The native loop intentionally mutates previous assistant messages.
        # Do not leak references to that mutable working history into training.
        result = await self.generate(copy.deepcopy(messages), copy.deepcopy(tools), budget)
        prompt_ids = tuple(result['prompt_ids'])
        response_ids = tuple(result['response_ids'])
        logprobs = tuple(result.get('response_logprobs') or ())
        if not prompt_ids or not response_ids:
            raise ValueError('Empty native rollout input/output')
        if logprobs and len(logprobs) != len(response_ids):
            raise ValueError('Per-turn rollout logprobs are misaligned')
        if len(response_ids) > budget:
            raise ValueError('Generation exceeded the native per-turn budget')
        self.turns.append(TurnRecord(prompt_ids, response_ids, logprobs, budget))
        return result['message'], result['finish_reason'], dict(
            requested_max_tokens=budget, completion_tokens=len(response_ids),
            token_limit_reached=len(response_ids) >= budget)

    def _save_failed_chapter_trace(self, trace_dir, story_id, chapter_info,
                                   user_msg, trajectory, runtime, error):
        self.failed_runtime = copy.deepcopy(runtime)
        return super()._save_failed_chapter_trace(trace_dir, story_id, chapter_info,
                                                  user_msg, trajectory, runtime, error)

    async def run_chapter(self, trace_dir=None):
        s = self.snapshot
        chapters = copy.deepcopy(s['chapters'])
        state = StoryState.from_dict(copy.deepcopy(s['state']))
        complete = True
        try:
            chapter = await self._run_chapter_loop(s['prompt'], s['outline'], s['outline'][s['k']],
                chapters, state, 'en', s['prompt_id'], trace_dir=trace_dir)
            trace = chapter.get('tool_trace', [])
            complete = not chapter.get('degraded', False)
        except RuntimeError as error:
            # Infrastructure and parsing failures must propagate, not become reward zero.
            expected = f"Chapter {s['k']}: model did not output DONE within {self.max_turns_per_chapter} turns"
            if str(error) != expected or self.failed_runtime is None:
                raise
            complete = False
            runtime = self.failed_runtime
            chapter = runtime.written_chapter or runtime.last_rejected_draft
            trace = runtime.tool_trace
        updates = [dict(arguments=t['args'], result=dict(ok=t['ok'], info=t.get('info')))
                   for t in trace if t['name'] == 'update']
        complete = complete and any(u['result']['ok'] for u in updates)
        return dict(chapter=chapter, complete=complete, updates=updates, trace=trace,
                    post_state=state.to_dict(), turns=[t.as_dict() for t in self.turns])
