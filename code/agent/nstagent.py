#!/usr/bin/env python
# coding: utf-8
"""Frozen public entry point for NstAgent (Narrative State Tracking Agent).

NstAgent uses the atomic four-array NarrativeOps update and keeps its
detailed state-maintenance guidance only in the chapter user prompt.  The
underlying implementation is shared with the ablations, but this entry
point refuses protocol overrides.
"""

import asyncio
import hashlib
from pathlib import Path

import agent_core as core


METHOD_NAME = "NstAgent"
PROTOCOL_VERSION = "narrative_ops_prompt_v1"
STATE_UPDATE_MODE = "narrative_ops"
GUIDANCE_PLACEMENT = "prompt"


def current_core_sha256() -> str:
    return hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()


def build_arg_parser():
    parser = core.build_arg_parser()
    parser.prog = "nstagent.py"
    parser.description = (
        "NstAgent: frozen NarrativeOps structured-state story generation"
    )
    parser.set_defaults(
        state_update_mode=STATE_UPDATE_MODE,
        narrative_ops_guidance_placement=GUIDANCE_PLACEMENT,
    )
    return parser


def validate_args(parser, args) -> None:
    if args.state_update_mode != STATE_UPDATE_MODE:
        parser.error(
            f"NstAgent fixes --state-update-mode={STATE_UPDATE_MODE}"
        )
    if args.narrative_ops_guidance_placement != GUIDANCE_PLACEMENT:
        parser.error(
            "NstAgent fixes --narrative-ops-guidance-placement="
            f"{GUIDANCE_PLACEMENT}"
        )
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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    print(f"{METHOD_NAME} protocol: {PROTOCOL_VERSION}")
    print(f"Core SHA-256: {current_core_sha256()}")
    core.print_run_config(args)
    asyncio.run(core.main_async(args))


if __name__ == "__main__":
    main()
