#!/usr/bin/env python
"""Prepare WritingBench input without evaluator-injected format violations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
from agent_core import requires_dialogue_only  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        story = row.get("story")
        if requires_dialogue_only(row.get("prompt", "")) and isinstance(story, list):
            # WritingBench otherwise inserts Markdown chapter headings.  Those
            # headings are not model output and must not count against a prompt
            # that explicitly requires dialogue-only prose.
            row = dict(row)
            row["story_chapters"] = story
            row["story"] = "\n\n".join(
                str(chapter.get("content", ""))
                if isinstance(chapter, dict) else str(chapter)
                for chapter in story
            )
        rows.append(row)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
