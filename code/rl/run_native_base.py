"""Reuse the verified base shell settings with only an experiment-local entry point."""
import subprocess
import sys
from pathlib import Path

source=Path(__file__).with_name('run_grpo_base.sh').read_text()
anchor='python -m verl.trainer.main_ppo '
if source.count(anchor)!=1:
    raise RuntimeError('Base training launcher changed; inspect before adapting')
# No installed launcher or VeRL source file is changed. All positional overrides
# are forwarded as arguments, never interpolated into the shell source.
source=source.replace(anchor,'python -m native_train_entry ')
temp_anchor='export RAY_TMPDIR="$project_root/ray"'
if source.count(temp_anchor)!=1:
    raise RuntimeError('Inspect changed Ray temporary-directory setup')
# Ray appends session/socket suffixes; the experiment's shared path exceeds
# Linux's 107-byte Unix-socket limit. Only ephemeral IPC files use this short path.
source=source.replace(temp_anchor,'export RAY_TMPDIR="$(mktemp -d /tmp/nst-ray.XXXXXXXX)"')
raise SystemExit(subprocess.run(['bash','-s','--',*sys.argv[1:]],input=source,text=True).returncode)
