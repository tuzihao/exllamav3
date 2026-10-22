"""Explicit, inference-only compatibility patch for the pinned EXL3 0.0.37.

GDN uses upstream's deterministic reduction. Fused MoE uses upstream's
one-output-slot-per-assignment design, retaining the installed quantized CUDA
kernel byte for byte: gathered duplicate input rows let its existing scatter
write unique slots, then a separate kernel sums them in fixed expert order.
No quantizer, weight decoder, expert GEMM, routing or precision is replaced.
"""
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os

PACKAGE = Path(__file__).resolve().parent
HERE = PACKAGE / 'determinism_ext'


def build(build_directory=None):
    import torch
    from torch.utils.cpp_extension import load, _get_build_directory
    from .ext import exllamav3_ext as native
    manifest = json.loads((HERE / 'base-manifest.json').read_text())
    if version('exllamav3') != manifest['base_version']:
        raise RuntimeError('This patch is only for the pinned EXL3 0.0.37 wheel')
    for relative, expected in manifest['native_files'].items():
        with (PACKAGE / relative).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                raise RuntimeError(f'Native source/binary changed: {relative}')
    if hashlib.sha256((HERE / 'gdn_128.cuh').read_bytes()).hexdigest() != manifest['gdn_generated_sha256']:
        raise RuntimeError('Generated GDN source changed')
    with Path(native.__file__).open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != manifest['reference_native_binary_sha256']:
            raise RuntimeError('Untested native binary: use the documented 0.0.37 wheel')
    build_dir = Path(build_directory or _get_build_directory('escha_exl3_issue353_v1', verbose=False))
    build_dir.mkdir(exist_ok=True)
    capability = torch.cuda.get_device_capability()
    previous = {key: os.environ.get(key) for key in ('MAX_JOBS', 'TORCH_CUDA_ARCH_LIST')}
    try:
        os.environ['MAX_JOBS'] = '2'
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{capability[0]}.{capability[1]}'
        return load(name='escha_exl3_issue353_v1', sources=[str(HERE / 'extension.cu')],
                    build_directory=str(build_dir), extra_cflags=['-Ofast'],
                    extra_cuda_cflags=manifest['cuda_flags'], verbose=True)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def enabled(extension=None):
    """Enable only around inference in one process; restore original entry points."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    if getattr(ext.cuda_recurrent_gated_delta_rule, '_issue353', False):
        raise RuntimeError('Nested patch activation is unsupported')
    extension = extension or build()
    original_gdn, original_moe = ext.cuda_recurrent_gated_delta_rule, ext.exl3_moe
    stats = {'gdn_calls': 0, 'gdn_other_shape_fallbacks': 0, 'moe_calls': 0, 'max_moe_assignments': 0}

    def recurrent(*args):
        if torch.is_grad_enabled():
            raise RuntimeError('Issue353 patch is inference-only')
        if args[7:9] != (128, 128):
            stats['gdn_other_shape_fallbacks'] += 1
            return original_gdn(*args)
        stats['gdn_calls'] += 1
        return extension.recurrent(*args)

    recurrent._issue353 = True

    def moe(hidden, output, counts, token_sorted, weights, *args):
        if torch.is_grad_enabled():
            raise RuntimeError('Issue353 patch is inference-only')
        rows = hidden.shape[0]
        assignments = token_sorted.numel()
        if rows == 0 or assignments == 0 or assignments % rows:
            raise ValueError('Invalid fused-MoE assignment layout')
        # Preserve expert groups, their row order, routing weights and every
        # quantized compute operation. Only destinations become non-overlapping.
        expanded = hidden.index_select(0, token_sorted)
        slots = torch.zeros((assignments, hidden.shape[1]), dtype=output.dtype, device=output.device)
        identity = torch.arange(assignments, dtype=token_sorted.dtype, device=token_sorted.device)
        original_moe(expanded, slots, counts, identity, weights, *args)
        order = token_sorted.argsort(stable=True).view(rows, assignments // rows)
        extension.gather_slots(slots, order, output)
        stats['moe_calls'] += 1
        stats['max_moe_assignments'] = max(stats['max_moe_assignments'], assignments)

    ext.cuda_recurrent_gated_delta_rule, ext.exl3_moe = recurrent, moe
    try:
        yield stats
    finally:
        ext.cuda_recurrent_gated_delta_rule, ext.exl3_moe = original_gdn, original_moe
