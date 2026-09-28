"""Reuse exact base settings; replace only the entry point's storage hook."""
from pathlib import Path
source = Path(__file__).with_name('run_native_base.py').read_text()
assert source.count("'python -m native_train_entry '") == 1
source = source.replace("'python -m native_train_entry '", "'python -m staged_train_entry '")
exec(compile(source, __file__, 'exec'))
