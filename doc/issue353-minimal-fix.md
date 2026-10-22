# Minimal inference determinism patch for EXL3 0.0.37

This development fork adds an **explicit, inference-only** patch for
the floating-point accumulation nondeterminism discussed in upstream
[issue #353](https://github.com/turboderp-org/exllamav3/issues/353).
The base commit is `1d71227a7c458228ae80d42e536d08c02883ae51` (`v0.0.37`).

## Changes

1. **GDN with 128-dimensional key/value heads:** backport the two fixed-order
   reductions from upstream commit
   [`900cf17`](https://github.com/turboderp-org/exllamav3/commit/900cf17b1ebc87e80f68cdcf0e5a284ae28d2e0a).
   Each of four key slices writes its own FP32 partial; consumers add the four
   partials in index order. State arithmetic, dispatch, BF16 round-toward-zero
   output, and `-O3 --use_fast_math` are retained. Other head shapes use the
   original function and are not covered by this patch.
2. **Fused MoE output accumulation:** apply the independent-output-slot and
   fixed-order-gather design of upstream commit
   [`1c8aff4`](https://github.com/turboderp-org/exllamav3/commit/1c8aff478a787fe096d231f9a0eaacc23b08701e).
   This is a compatibility adaptation, not a wholesale cherry-pick. The old
   native fused kernel is kept byte for byte. Input rows are gathered to one row
   per expert assignment, so the existing atomic output writes have disjoint
   destinations. A new gather kernel adds the weighted FP32 slots in fixed
   expert order. The newer upstream implementation uses routing-slot order;
   both impose a fixed sum, but their rounded results need not be identical.

The installed quantizer, calibration flow, trellis decoder, codebooks, scales,
expert GEMMs, expert routing and attention dispatch are unchanged. This patch
does not expand or rewrite model weights. It does allocate temporary activation
storage of approximately `assignments * hidden_size * 6` bytes, plus indices.
No newer EXL3 performance features or mixed-precision changes are included.

## Supported environment and use

Validated target: Qwen3.6-35B-A3B, Qwen3.5 MoE architecture, head dimensions 128;
NVIDIA RTX PRO 6000 Blackwell, Python 3.12, Torch 2.9.0+cu128, CUDA toolkit 12.8,
and the **existing** `exllamav3-0.0.37+cu128.torch2.9.0` wheel. The module checks
the original source hashes and native binary hash and refuses an untested
binary. It does not install dependencies or upgrade the environment.

Use the existing wheel environment and put this checkout ahead of site-packages
on `PYTHONPATH`. This selects the unchanged base Python package plus the added
opt-in module, and reuses the wheel's compiled quantized operators:

```bash
export PYTHONPATH=/absolute/path/to/this/checkout${PYTHONPATH:+:$PYTHONPATH}
export TORCH_EXTENSIONS_DIR=/absolute/path/to/writable/build-cache
python your_inference_script.py
```

Inside the inference script, enable the patch explicitly:

```python
import torch
from exllamav3.determinism import enabled

# Load the model using the usual EXL3 loader.
with torch.inference_mode(), enabled() as counters:
    # Run model.forward(...) or the normal EXL3 generator here.
    ...
print(counters)
```

The first activation compiles only the small additional CUDA extension.
Exiting the context restores both original function entry points, including on
exceptions. Activation affects the current Python process; use a dedicated
inference process and do not overlap different patch contexts. Quantization and
training should run normally, outside this context. Calls with autograd enabled
are rejected.

GPU tests require no model download:

```bash
python -m unittest discover -s tests -p test_issue353_determinism.py -v
```

## Scope of the result

This targets run-to-run reproducibility. It does **not** promise batch-size
invariance, cross-GPU identity, unchanged generated text, or unchanged benchmark
accuracy. Fixing reduction order can change rounding and near-tie decisions.
Model-file equality and operator-level numerical agreement establish that the
quantization representation and formula were preserved; they do not by
themselves prove task-quality equivalence.

The original batch64-versus-serial numerical gate remains separate and must be
reported even if it fails. The validation report records measured results and
limitations; no model weights, datasets or per-item evaluation text are stored
in this repository.
