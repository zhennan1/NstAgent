"""Reuse completed base-model candidates for a disposable smoke test ONLY.

Never replay these into real training runs: failed-candidate replacement can affect
sampling, so this cache is limited to plumbing/learning-signal diagnostics.
"""
import json
import os
from functools import lru_cache
from pathlib import Path


def context_key(sample):
    return ':'.join(str(int(sample[k])) for k in ('prompt_id','length','k'))


@lru_cache(maxsize=1)
def verified_manifest(path):
    m=json.loads(Path(path).read_text())
    if m['scope']!='disposable_smoke_only' or m['optimizer_updates']!=0:
        raise ValueError('Replay is not verified as a pre-update smoke cache')
    if os.environ.get('MODEL_PATH')!=m['model_path']:
        raise ValueError('Smoke replay model differs from the original base model')
    return m


def take_smoke_replay(sample):
    path=os.environ.get('NARRATIVE_SMOKE_REPLAY_MANIFEST')
    if not path:return None
    if os.environ.get('EXPERIMENT_NAME')!='narrative-ops-base-smoke' or int(sample['step'])!=0:
        raise ValueError('Smoke replay is forbidden in formal or later-step training')
    m=verified_manifest(path)
    key=context_key(sample)
    if key not in m['contexts']:return None
    context=m['contexts'][key]
    claims=Path(os.environ['NARRATIVE_SMOKE_REPLAY_CLAIMS'])
    claims.mkdir(parents=True,exist_ok=True)
    for candidate in context['candidates']:
        target=Path(candidate['path'])
        claim=claims/(target.stem+'.claim')
        try:
            fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:
            continue
        os.close(fd)
        result=json.loads(target.read_text())
        if context_key(result['sample'])!=key:raise ValueError('Cached rollout context mismatch')
        result['replayed_from']=str(target)
        return result
    return None
