"""Author short, varied premises; benchmark data is used only for subsequent exclusion."""
import argparse
import asyncio
import json
import re
from pathlib import Path
from openai import AsyncOpenAI

GENRES = [
 'domestic literary fiction about unconventional households',
 'historical maritime fiction without piracy or shipwrecks',
 'near-future science fiction about scientific fieldwork',
 'romantic comedy about adults with incompatible practical responsibilities',
 'rural social drama centered on changing livelihoods',
 'fantasy about crafts and unusual material laws, without chosen heroes',
 'nonviolent mysteries about ordinary objects and disputed provenance',
 'sports fiction about amateur teams and collective ambition',
 'musical fiction about ensembles, rehearsals, and artistic disagreement',
 'workplace comedy in unusual small businesses',
 'intergenerational fiction about adult relatives learning to cooperate',
 'ecological fiction about unfamiliar habitats and competing forms of care',
 'historical fiction about civilian trades before the industrial era',
 'travel fiction about planned journeys gradually changing purpose',
 'culinary fiction about labor, hospitality, and cultural misunderstanding',
 'speculative fiction about bureaucracies regulating impossible phenomena',
 'theatrical fiction about backstage relationships and touring performances',
 'quiet romantic drama between older adults beginning new work',
 'adventure about collaborative mapping without treasure hunts',
 'magical realism rooted in ordinary neighborhood routines',
 'legal social drama about property, custody of objects, and communal agreements',
 'epistolary fiction involving practical collaboration across distance',
 'archaeological fiction about interpretation and ownership, without curses',
 'space fiction about civilian logistics rather than warfare',
 'urban ensemble fiction about shared public spaces',
 'psychological suspense about voluntary professional commitments, without murder',
 'comic animal fiction told from a working animal viewpoint, without pet illness',
 'alternative history about infrastructure and civic institutions',
 'diaspora fiction about language, belonging, and mutual obligations',
 'science fiction about communication with nonhuman natural systems',
 'winter adventure about communities organizing a complex shared undertaking',
 'philosophical fantasy about exchanges with precise unexpected consequences',
]

def build():
    path=Path(__file__).parent/'data/prompts.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

async def author(index, genre, client, args, semaphore):
    path=args.output/f'group_{index:02d}.json'
    if path.exists(): return
    instruction=(f'Invent EIGHT entirely different original English story-writing prompts in this area: {genre}. '
      'Return a JSON object with a prompts array containing eight strings. Each string must be 50–70 English words. '
      'Each should read like a concise natural user request: a distinctive protagonist, specific situation, central conflict, '
      'and one or two narrative preferences. Do not give the answer or complete plot. Do not specify word counts or chapter counts. '
      'Avoid generic instruction lists about consistency, tools, state tracking, character knowledge, or novel structure. '
      'Vary places, occupations, ages, relationships, stakes, tone, and viewpoint within the eight prompts. '
      'Do not recycle one plot with changed names. Avoid conventional lost-parent secrets, dying pets, locked-room murders, '
      'chosen teenagers saving the world, forbidden noble-servant romance, memory archives, returning estranged mentors, '
      'and time travelers questioning a lifelong quest. Invent specifics rather than adapting a known story or benchmark. '
      'Use varied sentence patterns and do not put all stories in coastal towns. Write the prompts, not the stories.')
    instruction += (' Use fluent English throughout, with no stray foreign words. Keep realistic genres realistic; '
                    'only introduce impossible events in explicitly speculative genres. Every premise must be coherent '
                    'and concrete, not a pile of whimsical contradictions. Do not imitate any published story.')
    async with semaphore:
        accepted=[]
        for attempt in range(6):
            response=await client.chat.completions.create(model=args.model,temperature=.8,max_tokens=1800,
                messages=[dict(role='user',content=instruction)],response_format={'type':'json_object'},
                extra_body=({'thinking':{'type':'disabled'}} if args.provider=='judge' else {'chat_template_kwargs':{'enable_thinking':False}}))
            try:
                prompts=json.loads(response.choices[0].message.content)['prompts']
                counts=[len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*",s)) for s in prompts]
                accepted.extend(s for s,n in zip(prompts,counts) if 50<=n<=70 and s not in accepted and not any(c.isalpha() and ord(c)>127 for c in s))
                if len(accepted)>=8:
                    prompts=accepted[:8]
                    path.write_text(json.dumps(dict(genre=genre,prompts=prompts,usage=response.usage.model_dump()),ensure_ascii=False,indent=2))
                    print(index,[len(s.split()) for s in prompts],flush=True)
                    return
                instruction += ' Give eight new plots, each approximately 60 words. Avoid repeating these accepted premises: '+json.dumps(accepted)
            except (ValueError,KeyError,AssertionError) as error:
                instruction += f' Previous batch failed validation ({str(error)[:120]}). Keep EACH prompt within 50–70 words.'
        raise RuntimeError(f'Could not author valid group {index}')

async def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if args.provider=='judge':
        from reward import credentials
        key,base=credentials()
    else:key,base='local',args.base_url
    async with AsyncOpenAI(api_key=key,base_url=base,max_retries=0,timeout=300) as client:
        semaphore=asyncio.Semaphore(8)
        results=await asyncio.gather(*(author(i,g,client,args,semaphore) for i,g in enumerate(GENRES)),return_exceptions=True)
        failures=[str(r) for r in results if isinstance(r,Exception)]
        if failures:raise RuntimeError(failures)
    rows=[]
    for i in range(32):
        group=json.loads((args.output/f'group_{i:02d}.json').read_text())
        start=len(rows)
        rows.extend(dict(id=start+j,language='en',genre=group['genre'],prompt=s,prompt_version='v2_short') for j,s in enumerate(group['prompts']))
    assert len(rows)==256 and len({r['prompt'] for r in rows})==256
    path=args.output.parent/'prompts_v2_candidates.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://gpu67:8000/v1')
    p.add_argument('--provider',choices=['local','judge'],default='local')
    p.add_argument('--model',default='Qwen3.5-4B')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'data/authoring_v2')
    asyncio.run(main(p.parse_args()))
