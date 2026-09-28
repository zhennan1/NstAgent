"""Pinned generation profile for the Qwen3.5-4B narrative_ops agent.

Settings are listed explicitly because they differ from the agent's CLI defaults.
"""
PROFILE = dict(disable_thinking=False, length_control_mode='adaptive_recent_failure',
               chapter_token_control='dynamic', dynamic_token_ratio_en=1.35,
               dynamic_token_ratio_zh=.85, dynamic_token_overhead=1536,
               dynamic_token_minimum=1024, max_turns_per_chapter=30,
               request_attempts=3, state_update_mode='narrative_ops')
MAX_TOKENS = 16384
PROFILE_ID = 'current-narrative-4b-thinking-dynamic-adaptive-v2'

def verify_snapshot(snapshot):
    if snapshot.get('generation_profile') != PROFILE_ID:
        raise ValueError('Snapshot generation settings do not match the pinned thinking profile')
