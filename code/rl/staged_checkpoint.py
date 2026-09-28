"""Single-node transactional checkpoint publication, outside installed VeRL."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate(folder, world=8):
    paths = [folder/'actor'/f'{kind}_world_size_{world}_rank_{rank}.pt'
             for kind in ('model', 'optim', 'extra_state') for rank in range(world)]
    paths.append(folder/'data.pt')
    for path in paths:
        assert path.stat().st_size > 0, path
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, path
    assert (folder/'actor/fsdp_config.json').is_file()
    assert (folder/'actor/huggingface/config.json').is_file()


def publish(source, root, step):
    """Keep both the prior durable checkpoint and RAM source until commit."""
    root.mkdir(parents=True, exist_ok=True)
    target = root/f'global_step_{step}'
    if target.exists():
        validate(target)
        manifest = json.loads((target/'durable_manifest.json').read_text())
        assert manifest['step'] == step
        pointer = root/'latest_checkpointed_iteration.txt.tmp'
        with pointer.open('w') as out:
            out.write(str(step))
            out.flush()
            os.fsync(out.fileno())
        pointer.replace(root/'latest_checkpointed_iteration.txt')
        return target
    pending = root/f'.pending_step_{step}'
    pending.mkdir(exist_ok=True)
    records = {}
    for src in sorted(source.rglob('*')):
        if not src.is_file():
            continue
        rel = str(src.relative_to(source))
        expected = digest(src)
        dst = pending/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.is_file() and dst.stat().st_size == src.stat().st_size):
            part = dst.with_name(dst.name+'.partial')
            with src.open('rb') as inp, part.open('wb') as out:
                shutil.copyfileobj(inp, out, 8*1024**2)
                out.flush()
                os.fsync(out.fileno())
            part.replace(dst)
        records[rel] = dict(bytes=src.stat().st_size, sha256=expected)
    validate(pending)
    manifest = pending/'durable_manifest.json'
    with manifest.open('w') as out:
        json.dump(dict(step=step, files=records), out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    pending.rename(target)
    pointer = root/'latest_checkpointed_iteration.txt.tmp'
    with pointer.open('w') as out:
        out.write(str(step))
        out.flush()
        os.fsync(out.fileno())
    pointer.replace(root/'latest_checkpointed_iteration.txt')
    fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def install():
    from native_trainer import NativePPOTrainer
    original = NativePPOTrainer._save_checkpoint

    def save(self):
        assert self.config.trainer.nnodes == 1, 'RAM staging is single-node only'
        assert not self.use_critic
        root = Path(self.config.trainer.default_local_dir).resolve()
        assert root == Path(os.environ.get('NARRATIVE_ROOT', Path(__file__).resolve().parents[1])) / 'narrative_grpo/checkpoints/narrative-ops-base-2epoch'
        stage = Path(tempfile.mkdtemp(prefix='nst-checkpoint-', dir='/dev/shm'))
        assert shutil.disk_usage(stage).free > 65*1024**3
        old_dir = self.config.trainer.default_local_dir
        old_keep = self.config.trainer.max_actor_ckpt_to_keep
        try:
            self.config.trainer.default_local_dir = str(stage)
            self.config.trainer.max_actor_ckpt_to_keep = None
            original(self)
        finally:
            self.config.trainer.default_local_dir = old_dir
            self.config.trainer.max_actor_ckpt_to_keep = old_keep
        source = stage/f'global_step_{self.global_steps}'
        validate(source)
        # An I/O error must not immediately discard the completed optimizer state.
        # Retry publication only, never repeat generation or the optimizer update.
        for attempt in range(120):
            try:
                target = publish(source, root, self.global_steps)
                break
            except OSError as error:
                print(f'CHECKPOINT_PUBLISH_RETRY step={self.global_steps} attempt={attempt+1} '
                      f'ram_source={source} error={error!r}', flush=True)
                if attempt == 119:
                    raise
                time.sleep(30)
        print(f'CHECKPOINT_DURABLE step={self.global_steps} path={target}', flush=True)
        # Only successful earlier checkpoints in this exact run root are rotated.
        for previous in root.glob('global_step_*'):
            suffix = previous.name.removeprefix('global_step_')
            if suffix.isdigit() and int(suffix) < self.global_steps:
                shutil.rmtree(previous)
        shutil.rmtree(stage)

    NativePPOTrainer._save_checkpoint = save
