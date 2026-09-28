"""Ray worker hook: release unused PyTorch cache before vLLM wakes its weights.

No tensors, gradients or optimizer state are modified. Imported before worker
modules so their allocator-toggle reference includes this boundary cleanup.
"""
import importlib.abc
import importlib.machinery
import sys

def patch(device):
    if getattr(device.set_expandable_segments, '_narrative_cache_hook', False):
        return
    original = device.set_expandable_segments
    def clear_before_wake(enable):
        import torch
        if not enable and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            print(f'ROLLOUT_CACHE_RELEASE reserved_before={before} '
                  f'reserved_after={torch.cuda.memory_reserved()} '
                  f'allocated={torch.cuda.memory_allocated()} free={free}', flush=True)
        return original(enable)
    clear_before_wake._narrative_cache_hook = True
    device.set_expandable_segments = clear_before_wake

class DeferredDevicePatch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != 'verl.utils.device':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        loader = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return loader.create_module(spec)
            def exec_module(self, module):
                loader.exec_module(module)
                patch(module)
        spec.loader = Loader()
        return spec

def install():
    # Ray runs setup before per-task CUDA_VISIBLE_DEVICES is established.
    # Never import torch or device here: defer to their normal import time.
    if 'verl.utils.device' in sys.modules:
        patch(sys.modules['verl.utils.device'])
    elif not any(isinstance(f, DeferredDevicePatch) for f in sys.meta_path):
        sys.meta_path.insert(0, DeferredDevicePatch())
