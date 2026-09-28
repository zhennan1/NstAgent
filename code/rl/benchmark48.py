"""Paired standard 20-prompt 10K full-story benchmark; no training mutations."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from staged_checkpoint import digest

root=Path(os.environ.get('NARRATIVE_ROOT', Path(__file__).resolve().parents[1]))
bundle=root/'narrative_grpo/benchmark16_assets'
out=root/'narrative_grpo/results/step48_constory20_10k_v1'
out.mkdir(parents=True,exist_ok=True)
py=sys.executable
children=[]
env=dict(os.environ,OPENAI_API_KEY='EMPTY')
def run(args,log,environment=None):
    with (out/log).open('a') as f:
        subprocess.run([py,*map(str,args)],cwd=root,env=environment or env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=14400)
def stop(*args):raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
try:
    run([root/'narrative_grpo/export_step48.py'],'export.log')
    assert (root/'narrative_grpo/results/step48_export.json').is_file()
    previous=json.loads((root/'narrative_grpo/results/step16_constory20_10k_v1/manifest.json').read_text())
    manifest=dict(prompts_sha256=digest(bundle/'prompts_20.jsonl'),agent_sha256=digest(root/'longstoryagent_dynamic_tokens.py'),
        models=['step48'],length=10000,temperature=.7,thinking=True,outline_policy='require',
        judge='DeepSeek-V4-Pro',scope='Standard ConStoryBench 20 prompts, full 10K stories, paired frozen base outlines')
    path=out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:path.write_text(json.dumps(manifest,indent=2))
    for name,port,cards,model in [('step48',8041,'0,1,2,3',str(root/'narrative_grpo/models/rl-step48-hf'))]:
        with (out/f'{name}_server.log').open('a') as f:
            children.append(subprocess.Popen([py,'-m','vllm.entrypoints.openai.api_server','--model',model,'--served-model-name','qwen3.5-4b','--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','4','--max-model-len','262144','--max-num-seqs','8','--gpu-memory-utilization','.85','--language-model-only','--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_coder'],env=dict(env,CUDA_VISIBLE_DEVICES=cards),stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
    deadline=time.monotonic()+1200
    for port in (8041,):
        while True:
            assert all(p.poll() is None for p in children),'Server exited'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as r:
                    if r.status==200:break
            except OSError:pass
            assert time.monotonic()<deadline,'Server startup timeout'
            time.sleep(3)
    def generate(name,port):
        folder=out/name;folder.mkdir(exist_ok=True)
        run([root/'longstoryagent_dynamic_tokens.py','--input',bundle/'prompts_20.jsonl','--output',folder/'results.jsonl','--model','qwen3.5-4b','--api-base',f'http://127.0.0.1:{port}/v1','--max-tokens','16384','--temperature','.7','--concurrent','4','--word-count','10000','--outline-cache-dir',bundle/'plans','--outline-cache-policy','require','--outline-cache-model-id','qwen3.5-4b','--chapter-token-control','dynamic','--dynamic-token-overhead','1536','--length-control-mode','adaptive_recent_failure','--state-update-mode','narrative_ops','--narrative-ops-guidance-placement','prompt','--max-turns-per-chapter','30','--auto-resumes','1','--request-attempts','3','--client-max-retries','2','--start','0','--end','20'],f'{name}_generation.log')
        run([bundle/'prepare_final_evaluation.py','--dataset',f'{name}_10k={folder}/results.jsonl','--output-dir',out/'validated','--expected-samples','20','--expected-prompts',bundle/'prompts_20.jsonl','--reject-duplicates'],f'{name}_validation.log')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(generate,n,p) for n,p in [('step48',8041)]]
        for f in futures:f.result()
    (out/'generation_done.json').write_text(json.dumps(dict(time=time.time())))
finally:
    for p in children:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for p in children:
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
    (out/'servers_released.json').write_text(json.dumps(dict(time=time.time())))


