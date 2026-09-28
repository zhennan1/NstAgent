"""Read-only semantic/structural checks of completed records; no model or judge calls."""
import json
import argparse
from pathlib import Path
from generation_common import count_words
from narrative_config import PROFILE_ID

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix-root',type=Path,default=root/'data/prefixes')
    args=parser.parse_args()
    prompts={r['id']:r['prompt'] for r in map(json.loads,(root/'data/prompts.jsonl').read_text().splitlines())}
    problems=[];complete=0;checked_chapters=0
    for d in sorted(args.prefix_root.iterdir()):
        marker=d/'complete.json'
        if not marker.exists() or json.loads(marker.read_text()).get('chapters',0)<9:continue
        complete+=1;issues=[]
        try:
            s=json.loads((d/'prefix_9.json').read_text())
            assert s['k']==9 and len(s['chapters'])==9
            assert s['generation_profile']==PROFILE_ID
            assert s['prompt']==prompts[s['prompt_id']]
            assert d.name==f"{s['prompt_id']:03d}_{s['length']}"
            assert s['length'] in (10000,20000,50000,100000)
            assert len(s['outline'])>=10
            assert all((d/f'prefix_{k}.json').exists() for k in range(10))
            # GRPO can sample any k: final-history validity alone is insufficient.
            for k in range(9):
                prefix=json.loads((d/f'prefix_{k}.json').read_text())
                assert prefix['k']==k and len(prefix['chapters'])==k
                assert prefix['generation_profile']==PROFILE_ID and prefix['prompt']==s['prompt']
                for i,c in enumerate(prefix['chapters']):
                    target=prefix['outline'][i]['word_count'];words=count_words(c['content'])
                    if not .8*target<=words<=1.2*target:
                        issues.append(dict(prefix=k,chapter=i,error='intermediate_word_count',actual=words,target=target))
            for k,c in enumerate(s['chapters']):
                checked_chapters+=1;actual=count_words(c['content']);target=s['outline'][k]['word_count']
                if c['id']!=k or c.get('degraded'):issues.append(dict(chapter=k,error='id_or_degraded'))
                if not .8*target<=actual<=1.2*target:issues.append(dict(chapter=k,error='final_word_count',actual=actual,target=target))
                if not any(t['name']=='update' and t['ok'] for t in c['tool_trace']):issues.append(dict(chapter=k,error='missing_successful_update'))
        except Exception as error:issues.append(dict(error=type(error).__name__,detail=str(error)))
        if issues:problems.append(dict(record=d.name,issues=issues))
    result=dict(profile=PROFILE_ID,prefix_root=str(args.prefix_root.resolve()),completed_markers=complete,checked_chapters=checked_chapters,
                valid_records=complete-len(problems),invalid_records=problems,
                all_1024_valid=complete==1024 and not problems)
    (root/'results/prefix_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
