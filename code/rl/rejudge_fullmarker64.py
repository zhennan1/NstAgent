"""Full-marker ConStory re-judging for base 10K and step64 10K-100K (CPU/API only).

The baseline judge runs use prompts_terminal_scope with --target-ending-chapters 0
(no marker, empty target IDs).  This re-judges the same frozen validated inputs with
all chapters marked as target.  WritingBench is unaffected and not re-run.
"""
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

from reward import credentials

csv.field_size_limit(sys.maxsize)
root = Path(os.environ.get('NARRATIVE_ROOT', Path(__file__).resolve().parents[1]))
bundle = root/'narrative_grpo/benchmark16_assets'
results = root/'narrative_grpo/results'
out = results/'fullmarker_constory20_v1'
DATASETS = {
    # name: (validated input, previous no-marker CSV, first-pass concurrency)
    'base_10k': ('step16_constory20_10k_judge_v1/validated/base_10k.jsonl', 'step16_constory20_10k_judge_v1/constory_improved/base.csv', 10),
    'step64_10k': ('step64_constory20_10k_judge_v1/validated/step64_10k.jsonl', 'step64_constory20_10k_judge_v1/constory_improved/step64.csv', 10),
    'step64_20k': ('step64_constory20_20k_judge_v1/validated/step64_20k.jsonl', 'step64_constory20_20k_judge_v1/constory_improved/step64.csv', 10),
    'step64_50k': ('step64_constory20_50k_judge_v1/validated/step64_50k.jsonl', 'step64_constory20_50k_judge_v1/constory_improved/step64.csv', 8),
    'step64_100k': ('step64_constory20_100k_judge_v1/validated/step64_100k.jsonl', 'step64_constory20_100k_judge_v1/constory_improved/step64.csv', 6),
}
RECOVERY_CONCURRENT = 2
OLD_100K_PID = 1486761  # previous no-marker 100K judge; wait so both do not share quota

for sub in ('inputs', 'constory_improved', 'logs', 'stages'):
    (out/sub).mkdir(parents=True, exist_ok=True)
lock = (out/'rejudge.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
expected = {str(json.loads(x)['id']) for x in (bundle/'prompts_20.jsonl').read_text().splitlines() if x.strip()}
assert len(expected) == 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest_path = out/'input_manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for name, (src, old_csv, _) in DATASETS.items():
    src, frozen = results/src, out/'inputs'/f'{name}.jsonl'
    if name in manifest:
        continue
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    assert {str(r['id']) for r in rows} == expected and len(rows) == 20, name
    assert all(r['complete'] and isinstance(r['story'], list) and r['story'] for r in rows), name
    shutil.copyfile(src, frozen)
    manifest[name] = dict(source=str(src), source_sha256=sha(src), frozen_sha256=sha(frozen),
                          previous_csv=str(results/old_csv))
manifest_path.write_text(json.dumps(manifest, indent=2))

key, base_url = credentials()
env = dict(os.environ, OPENAI_API_KEY=key)
env.pop('NARRATIVE_JUDGE_API_KEY', None)


def event(msg):
    with (out/'events.tsv').open('a') as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{msg}\n")


def complete_ids(path):
    if not path.exists():
        return set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = {str(r['id']): r for r in csv.DictReader(f) if r.get('id')}
    return {k for k, r in rows.items() if r.get('evaluation_status') == 'completed'
            and r.get('criteria_completed') in ('5', '5.0') and r.get('scope_mode') == 'target_ending_chapters'}


def evaluate(name, concurrent, tag):
    event(f'{name}\tstart\t{tag}\tconcurrent={concurrent}')
    with (out/'logs'/f'constory_{name}_{tag}.log').open('a') as f:
        subprocess.run([sys.executable, str(bundle/'ConStory-Bench/evaluate.py'),
                        '--input', str(out/'inputs'/f'{name}.jsonl'),
                        '--output', str(out/'constory_improved'/f'{name}.csv'),
                        '--prompts-dir', str(bundle/'ConStory-Bench/prompts_terminal_scope'),
                        '--target-ending-chapters', '999',
                        '--judge-model', 'DeepSeek-V4-Pro', '--api-base', base_url,
                        '--concurrent', str(concurrent), '--max-tokens', '65536', '--request-timeout', '3600'],
                       cwd=bundle, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=43200)


def run_dataset(name):
    stages, csv_path = out/'stages', out/'constory_improved'/f'{name}.csv'
    if (stages/f'{name}.done').exists():
        return name, 'done'
    if (stages/f'{name}.stopped').exists():
        return name, 'stopped'
    if name == 'step64_100k':
        while Path(f'/proc/{OLD_100K_PID}').exists():
            time.sleep(60)
        event(f'{name}\tprevious 100K judge exited; starting')
    evaluate(name, DATASETS[name][2], 'pass1')
    if complete_ids(csv_path) != expected:
        if (stages/f'{name}.recovery1').exists():
            (stages/f'{name}.stopped').write_text('recovery already used\n')
            return name, 'stopped'
        (stages/f'{name}.recovery1').write_text(time.strftime('%FT%T%z') + '\n')
        evaluate(name, RECOVERY_CONCURRENT, 'recovery1')
        if complete_ids(csv_path) != expected:
            (stages/f'{name}.stopped').write_text('incomplete after one bounded recovery; diagnose before any paid call\n')
            event(f'{name}\tstopped\t{len(complete_ids(csv_path))}/20')
            return name, 'stopped'
    (stages/f'{name}.done').write_text(time.strftime('%FT%T%z') + '\n')
    event(f'{name}\tdone')
    return name, 'done'


with concurrent.futures.ThreadPoolExecutor(len(DATASETS)) as pool:
    status = dict(f.result() for f in [pool.submit(run_dataset, n) for n in DATASETS])
event(f'all\tfinished\t{json.dumps(status)}')

sys.path.insert(0, str(bundle))
from analyze_paired_results import ced_by_id  # noqa: E402


def paired(new, old):
    ids = sorted(set(new) & set(old), key=int)
    a, b = [new[i] for i in ids], [old[i] for i in ids]
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(20260914)
    boot = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(20000))
    try:
        from scipy import stats
        t_p, w_p = float(stats.ttest_rel(a, b).pvalue), float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        t_p = w_p = None
    return dict(n=len(ids), new_mean=statistics.fmean(a), old_mean=statistics.fmean(b), diff=statistics.fmean(d),
                ci95=[boot[500], boot[19499]], t_p=t_p, wilcoxon_p=w_p,
                higher_equal_lower=[sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)])


summary = dict(status=status, config='prompts_terminal_scope + target-ending-chapters 999 (all chapters marked)', datasets={})
scores = {}
for name, (_, old_csv, _) in DATASETS.items():
    if status.get(name) != 'done':
        continue
    new_off, new_inst = ced_by_id(out/'constory_improved'/f'{name}.csv')
    old_off, old_inst = ced_by_id(results/old_csv)
    assert set(new_off) == expected, name
    scores[name] = dict(new_official=new_off, new_instance=new_inst, old_official=old_off, old_instance=old_inst)
    summary['datasets'][name] = dict(official_new_vs_old=paired(new_off, old_off), instance_new_vs_old=paired(new_inst, old_inst))
if 'base_10k' in scores and 'step64_10k' in scores:
    s, b = scores['step64_10k'], scores['base_10k']
    summary['step64_minus_base_10k'] = {f'{era}_{m}': paired(s[f'{era}_{m}'], b[f'{era}_{m}'])
                                        for era in ('old', 'new') for m in ('official', 'instance')}
(out/'summary.json').write_text(json.dumps(summary, indent=2))
print('FULLMARKER_REJUDGE_FINISHED', json.dumps(status), flush=True)
