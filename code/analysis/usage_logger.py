#!/usr/bin/env python3
"""Run a generation script unchanged while logging the usage of every chat.completions call.

Usage: USAGE_LOG=/path/usage.jsonl python usage_logger.py /path/to/script.py [script args...]

The OpenAI SDK's sync and async ``chat.completions.create`` are wrapped at class level, so direct clients,
AutoGen's OpenAIWrapper and every adapter are covered without editing method code. One JSON line per call:
wall time, model, max_tokens, whether thinking was explicitly disabled, finish_reason and the provider's
usage object (prompt tokens, cache hit/miss, completion tokens, reasoning tokens). Logging failures never
affect the call itself.
"""
import functools
import json
import os
import runpy
import sys
import threading
import time

from openai.resources.chat.completions import completions as _chat

LOG = os.environ["USAGE_LOG"]
_lock = threading.Lock()


def _record(kwargs, response, started, error=None):
    try:
        extra = kwargs.get("extra_body") or {}
        thinking = extra.get("thinking") if isinstance(extra, dict) else None
        template = extra.get("chat_template_kwargs") if isinstance(extra, dict) else None
        entry = {
            "t": round(time.time(), 3),
            "seconds": round(time.time() - started, 3),
            "model": kwargs.get("model"),
            "max_tokens": kwargs.get("max_tokens"),
            "stream": bool(kwargs.get("stream")),
            "tools": bool(kwargs.get("tools")),
            "thinking_disabled": bool(
                (isinstance(thinking, dict) and thinking.get("type") == "disabled")
                or (isinstance(template, dict) and template.get("enable_thinking") is False)
            ),
        }
        if error is not None:
            entry["error"] = type(error).__name__
        else:
            usage = getattr(response, "usage", None)
            entry["usage"] = usage.model_dump() if usage is not None else None
            try:
                entry["finish_reason"] = response.choices[0].finish_reason
            except Exception:
                entry["finish_reason"] = None
        line = json.dumps(entry, ensure_ascii=False)
        with _lock, open(LOG, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except Exception:
        pass


_sync_create = _chat.Completions.create
_async_create = _chat.AsyncCompletions.create


@functools.wraps(_sync_create)
def _logged_sync_create(self, *args, **kwargs):
    started = time.time()
    try:
        response = _sync_create(self, *args, **kwargs)
    except Exception as error:
        _record(kwargs, None, started, error)
        raise
    _record(kwargs, response, started)
    return response


@functools.wraps(_async_create)
async def _logged_async_create(self, *args, **kwargs):
    started = time.time()
    try:
        response = await _async_create(self, *args, **kwargs)
    except Exception as error:
        _record(kwargs, None, started, error)
        raise
    _record(kwargs, response, started)
    return response


_chat.Completions.create = _logged_sync_create
_chat.AsyncCompletions.create = _logged_async_create

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: usage_logger.py SCRIPT [ARGS...]")
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
