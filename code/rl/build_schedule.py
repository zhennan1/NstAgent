"""32 distinct premises / batch, eight per length, eight distinct k per length."""
import argparse
import json
import random
from pathlib import Path

LENGTHS = [10000, 20000, 50000, 100000]

def schedule(seed=20260908, epochs=3):
    rows=[]
    rng=random.Random(seed)
    for epoch in range(epochs):
        ids=list(range(256)); rng.shuffle(ids)
        # Four passes, rotating each disjoint group of eight through every length.
        for cycle in range(4):
            for block in range(8):
                batch=[]
                for length_slot,length in enumerate(LENGTHS):
                    group=(length_slot-cycle)%4
                    ks=rng.sample(range(10),8)
                    for offset,k in enumerate(ks):
                        pid=ids[block*32+group*8+offset]
                        batch.append(dict(epoch=epoch,step=epoch*32+cycle*8+block,
                                          prompt_id=pid,length=length,k=k))
                assert len({r['prompt_id'] for r in batch})==32
                for length in LENGTHS:
                    assert len({r['k'] for r in batch if r['length']==length})==8
                rows.extend(batch)
        epoch_rows=rows[-1024:]
        assert len({(r['prompt_id'],r['length']) for r in epoch_rows})==1024
    return rows

def materialize(root, output, epochs, smoke_only=False, snapshot_manifest=None):
    from datasets import Dataset
    from longstoryagent_dynamic_tokens import LongStoryAgent, StoryState, get_system_prompt
    import logging
    from narrative_config import PROFILE, MAX_TOKENS, verify_snapshot
    agent=LongStoryAgent(None,'Qwen3.5-4B',MAX_TOKENS,.7,10000,logging.getLogger(),**PROFILE)
    records=[]
    selected=schedule(epochs=epochs)
    if smoke_only:selected=selected[:32]
    frozen=json.loads(snapshot_manifest.read_text())['contexts'] if snapshot_manifest else {}
    assert not frozen or smoke_only, 'Replay snapshots are diagnostic-only, never formal training data'
    for r in selected:
        path=root/f"{r['prompt_id']:03d}_{r['length']}"/f"prefix_{r['k']}.json"
        if frozen:
            key=f"{r['prompt_id']}:{r['length']}:{r['k']}"
            path=Path(frozen[key]['snapshot_path'])
        s=json.loads(path.read_text())
        verify_snapshot(s)
        if smoke_only:
            from generation_common import count_words
            assert s['k']==r['k'] and len(s['chapters'])==r['k']
            for index,chapter in enumerate(s['chapters']):
                words=count_words(chapter['content']);target=s['outline'][index]['word_count']
                assert .8*target<=words<=1.2*target and not chapter.get('degraded'), f'Invalid smoke context: {path}'
        content=agent._build_chapter_prompt(s['prompt'],s['outline'],s['outline'][s['k']],StoryState.from_dict(s['state']),s['chapters'],'en')
        records.append(dict(data_source='narrative_ops',agent_name='narrative_ops',
                            prompt=[dict(role='system',content=get_system_prompt('chapter','en')),dict(role='user',content=content)],
                            reward_model=dict(style='rule',ground_truth='chapter'),
                            extra_info=dict(r,snapshot_path=str(path.resolve()))))
    output.mkdir(parents=True,exist_ok=True)
    # Pre-shuffled schedule is one physical pass containing 2–3 logical epochs.
    if not smoke_only:Dataset.from_list(records).to_parquet(output/'train.parquet')
    Dataset.from_list(records[:32]).to_parquet(output/'smoke.parquet')
    (output/('smoke_schedule.json' if smoke_only else 'schedule.json')).write_text(json.dumps(selected,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--snapshot-manifest',type=Path)
    a=p.parse_args();materialize(a.root,a.output,a.epochs,a.smoke_only,a.snapshot_manifest)
