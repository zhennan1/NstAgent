"""CPU/API-only standard benchmark scoring after generation nodes release."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from reward import credentials
root=Path(os.environ.get('NARRATIVE_ROOT', Path(__file__).resolve().parents[1]))
bundle=root/'narrative_grpo/benchmark16_assets'
out=root/'narrative_grpo/results/step16_constory20_10k_judge_v1'
assert (out/'generation_done.json').is_file()
import fcntl
lock=(out/'judge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'judges_launched.json').exists(),'Inspect partial judging before bounded recovery'
key,base=credentials();env=dict(os.environ,OPENAI_API_KEY=key)
(out/'judges_launched.json').write_text(json.dumps(dict(pid=os.getpid())))
for sub in ('writingbench','constory_improved','writing_inputs'):(out/sub).mkdir(exist_ok=True)
def run(args,log):
    with (out/log).open('a') as f:subprocess.run([sys.executable,*map(str,args)],env=env,cwd=bundle,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=14400)
def judge(name):
    inp=out/'validated'/f'{name}_10k.jsonl';wi=out/'writing_inputs'/f'{name}.jsonl'
    run([bundle/'prepare_writingbench_input.py','--input',inp,'--output',wi],f'{name}_prepare_writing.log')
    common=['--judge-model','DeepSeek-V4-Pro','--api-base',base,'--concurrent','4','--max-tokens','65536','--request-timeout','3600']
    jobs=[([bundle/'WritingBench/evaluate_v4pro.py','--input',wi,'--output',out/'writingbench'/f'{name}.jsonl','--criteria-cache',bundle/'criteria_cache.jsonl',*common],f'{name}_writingbench.log'),([bundle/'ConStory-Bench/evaluate.py','--input',inp,'--output',out/'constory_improved'/f'{name}.csv','--prompts-dir',bundle/'ConStory-Bench/prompts_terminal_scope','--target-ending-chapters','0',*common],f'{name}_constory.log')]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(run,*j) for j in jobs]:f.result()
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for f in [pool.submit(judge,n) for n in ('base','step16')]:f.result()
sys.path.insert(0,str(bundle))
from analyze_paired_results import writingbench_by_id,ced_by_id,paired_summary
metrics={}
for label,idx in [('writingbench',None),('ced_official',0),('ced_instance',1)]:
    scores=[writingbench_by_id(out/'writingbench'/f'{n}.jsonl') if idx is None else ced_by_id(out/'constory_improved'/f'{n}.csv')[idx] for n in ('base','step16')]
    result=paired_summary(*scores,samples=200000,seed=20260910)
    metrics[label]={k.replace('delta_narrative_ops_minus_batch','delta_step16_minus_base').replace('batch_mean','base_mean').replace('narrative_ops_mean','step16_mean'):v for k,v in result.items()}
    if idx is not None:metrics[label]['wins'],metrics[label]['losses']=metrics[label]['losses'],metrics[label]['wins']
(out/'summary.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics),flush=True)
