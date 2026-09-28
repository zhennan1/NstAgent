"""GRPO normalization before turn expansion, never weighted by a chapter's turn count."""
import math
import statistics
from collections import defaultdict


def chapter_advantages(episodes, expected_group_size=8, epsilon=1e-6):
    groups=defaultdict(list)
    seen=set()
    for episode in episodes:
        uid=episode['trajectory_id']
        if uid in seen:
            raise ValueError('Duplicate trajectory: normalize chapter rewards before expanding turns')
        seen.add(uid)
        reward=episode['reward']
        if isinstance(reward,bool) or not math.isfinite(reward):
            raise ValueError('Reward must be finite')
        groups[episode['prompt_uid']].append(episode)
    result={}
    for group in groups.values():
        if len(group)!=expected_group_size:
            raise ValueError('Incomplete GRPO candidate group')
        scores=[e['reward'] for e in group]
        mean=statistics.mean(scores)
        std=statistics.stdev(scores) if len(scores)>1 else 0.
        for episode in group:
            result[episode['trajectory_id']]=(episode['reward']-mean)/(std+epsilon)
    return result
