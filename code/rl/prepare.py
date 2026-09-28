"""Generate first-nine-chapter trajectories with unmodified NstAgent and freeze every prefix."""
import argparse
import asyncio
import copy
import json
import logging
import hashlib
import time
import httpx
from pathlib import Path
from openai import AsyncOpenAI
from longstoryagent_dynamic_tokens import LongStoryAgent, OutlineAgent, StoryState, count_words
from narrative_config import PROFILE, MAX_TOKENS, PROFILE_ID, verify_snapshot

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False))
    temporary.replace(path)

async def generate(row, length, args, semaphore):
    async with semaphore:
        root = args.output / f"{row['id']:03d}_{length}"
        digest=hashlib.sha256(row['prompt'].encode()).hexdigest()
        manifest=root/'generation_config.json'
        if manifest.exists():
            assert {k:v for k,v in json.loads(manifest.read_text()).items() if k!='prompt_sha256'} == dict(PROFILE,max_tokens=MAX_TOKENS,profile_id=PROFILE_ID)
        elif root.exists() and any(root.iterdir()):
            raise RuntimeError('Unversioned old output directory; archive it before generation')
        else:
            save(manifest,dict(PROFILE,max_tokens=MAX_TOKENS,profile_id=PROFILE_ID,prompt_sha256=digest))
        if (root/'prefix_0.json').exists():
            existing=json.loads((root/'prefix_0.json').read_text())
            verify_snapshot(existing)
            if existing['prompt']!=row['prompt']:
                raise RuntimeError('Prompt changed: refusing to reuse an old prefix')
        if (root / 'complete.json').exists() and json.loads((root/'complete.json').read_text()).get('chapters',0)>=args.chapters:
            return
        urls = args.base_url.split(',')
        url = urls[(row['id']*4 + [10000,20000,50000,100000].index(length)) % len(urls)]
        async def audit_response(response):
            await response.aread()
            # Local inference only: never serialize request headers or API keys.
            try: payload=response.json()
            except ValueError: payload={'status':response.status_code,'body':response.text[:2000]}
            save(root/'responses'/f'{time.time_ns()}.json',payload)
        client = AsyncOpenAI(api_key='local', base_url=url, max_retries=2, timeout=600,
                            http_client=httpx.AsyncClient(timeout=600,event_hooks={'response':[audit_response]}))
        logger = logging.getLogger(f"{row['id']}_{length}")
        agent = LongStoryAgent(client, args.model, MAX_TOKENS, .7, length, logger, **PROFILE)
        started = time.monotonic()
        try:
            if (root / 'outline.json').exists():
                artifacts = json.loads((root/'outline.json').read_text())
            else:
                planner = OutlineAgent(client, args.model, MAX_TOKENS, .7, length, logger, 'en', False)
                artifacts = await planner.generate(row['prompt'])
                if len(artifacts['outline']) < 10:
                    raise RuntimeError('Need at least 10 outline chapters to support k=9')
                save(root/'outline.json', artifacts)
            outline = artifacts['outline']
            state, chapters = StoryState(), []
            for k in range(args.chapters+1):
                path = root/f'prefix_{k}.json'
                if path.exists():
                    snapshot = json.loads(path.read_text())
                    verify_snapshot(snapshot)
                    state = StoryState.from_dict(copy.deepcopy(snapshot['state']))
                    chapters = copy.deepcopy(snapshot['chapters'])
                else:
                    snapshot = dict(prompt_id=row['id'], prompt=row['prompt'], prompt_sha256=digest, generation_profile=PROFILE_ID, length=length, k=k,
                                    outline=outline, chapters=copy.deepcopy(chapters), state=copy.deepcopy(state.to_dict()))
                    save(path, snapshot)
                if k == args.chapters:
                    break
                if (root/f'prefix_{k+1}.json').exists():
                    continue
                (root/'traces').mkdir(parents=True,exist_ok=True)
                chapter = await agent._run_chapter_loop(row['prompt'], outline, outline[k], chapters, state,
                                                         'en', row['id'], str(root/'traces'))
                updates = [t for t in chapter['tool_trace'] if t['name'] == 'update' and t['ok']]
                target = outline[k]['word_count']
                actual = count_words(chapter['content'])
                if chapter.get('degraded') or not updates or not .8*target <= actual <= 1.2*target:
                    raise RuntimeError(f'Invalid prefix chapter {k}: words={actual}, updates={len(updates)}')
                chapters.append(chapter)
            save(root/'complete.json', dict(prompt_id=row['id'], length=length, chapters=args.chapters,
                                           elapsed_seconds=time.monotonic()-started))
            print(f'COMPLETE prompt={row["id"]} length={length}', flush=True)
        except Exception as exc:
            save(root/'failure.json', dict(error_type=type(exc).__name__, error=str(exc)[:1000]))
            print(f'FAILED prompt={row["id"]} length={length} {type(exc).__name__}', flush=True)
        finally:
            await client.close()

async def main(args):
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()][:args.limit]
    rows = [r for r in rows if r['id']%args.num_shards in args.shard_indices]
    semaphore = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(generate(row, length, args, semaphore) for row in rows for length in args.lengths))
    expected = len(rows)*len(args.lengths)
    completed = sum((p:=args.output/f'{r["id"]:03d}_{length}'/'complete.json').exists() and json.loads(p.read_text()).get('chapters',0)>=args.chapters for r in rows for length in args.lengths)
    print(json.dumps(dict(expected=expected, completed=completed)), flush=True)
    if completed != expected:
        raise SystemExit(2)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8018/v1')
    p.add_argument('--model', default='Qwen3.5-4B')
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-indices',type=int,nargs='+',default=[0])
    p.add_argument('--lengths', nargs='+', type=int, default=[10000,20000,50000,100000])
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--chapters', type=int, default=9)
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(p.parse_args()))
