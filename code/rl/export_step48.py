"""Export pinned step48 model, verify loading and persist hashes before evaluation."""
import os
import json
from pathlib import Path
import torch
from staged_checkpoint import digest
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

root=Path(os.environ.get('NARRATIVE_ROOT', Path(__file__).resolve().parents[1]))/'narrative_grpo'
src=root/'models/rl-step48-sharded'
dst=root/'models/rl-step48-hf'
report=root/'results/step48_export.json'
if report.exists():
    saved=json.loads(report.read_text())
    assert saved['status']=='verified'
    for rel,meta in saved['files'].items():
        assert (dst/rel).stat().st_size==meta['bytes']
else:
    assert not dst.exists(), 'Inspect partial export before retrying'
    manifest=json.loads((src/'evaluation_manifest.json').read_text())
    assert manifest['step']==48 and manifest['model_only']
    for rel,meta in manifest['files'].items():
        assert (src/rel).stat().st_size==meta['bytes']
    config=ModelMergerConfig(operation='merge',backend='fsdp',local_dir=str(src/'actor'),
        target_dir=str(dst),hf_model_config_path=str(src/'actor/huggingface'),use_cpu_initialization=True)
    merger=FSDPModelMerger(config)
    merger.merge_and_save()
    cls=merger.get_transformers_auto_model_class()
    model,info=cls.from_pretrained(str(dst),torch_dtype=torch.bfloat16,device_map='cpu',output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys'),info
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    del model
    files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in dst.rglob('*') if p.is_file()}
    report.write_text(json.dumps(dict(status='verified',step=48,source_manifest_sha256=digest(src/'evaluation_manifest.json'),files=files),indent=2))
print('STEP48_EXPORT_VERIFIED',flush=True)


