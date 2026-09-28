"""Node-shared conservative admission control, no secrets or payloads on disk."""
import asyncio
import fcntl
import json
import os
import time
from pathlib import Path

def admit(state, now, cost, budget=4_000_000):
    events = [e for e in state.get('events', []) if e[0] > now - 65]
    assert 0 < cost <= budget, 'Request exceeds conservative admission budget'
    wait = max(0., state.get('cooldown_until', 0.)-now)
    if events:
        wait = max(wait, events[-1][0]+2.-now)
    total = sum(e[1] for e in events)
    for stamp, amount in events:
        if total + cost <= budget:
            break
        wait = max(wait, stamp+65.-now)
        total -= amount
    state['events'] = events
    if wait <= 0:
        events.append([now, cost])
    return state, max(0., wait)

def transaction(cost=None, cooldown=False):
    path = os.environ.get('NARRATIVE_JUDGE_RATE_FILE')
    if not path:
        return 0.
    p = Path(path)
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        raw = f.read()
        state = json.loads(raw) if raw else {}
        now = time.time()
        if cooldown:
            state['cooldown_until'] = max(state.get('cooldown_until', 0), now+65)
            wait = 65.
        else:
            state, wait = admit(state, now, cost)
        f.seek(0)
        json.dump(state, f)
        f.truncate()
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return wait

async def acquire(cost):
    while True:
        wait = transaction(cost)
        if wait <= 0:
            return
        await asyncio.sleep(min(wait+.05, 30))
