"""GPU tests for the opt-in patch, with synthetic tensors and no model download.

Run: python -m unittest discover -s tests -p test_issue353_determinism.py -v
"""
import unittest
import torch
from exllamav3.ext import exllamav3_ext as native
from exllamav3.determinism import build, enabled


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DeterministicReductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = build()

    def test_slot_gather_preserves_each_assignment_once(self):
        # Unsorted expert-assignment rows, repeated destination tokens, nonzero
        # initial output: CPU FP32 accumulation is the independent oracle.
        for rows, top_k, width in [(1, 8, 257), (3, 2, 128), (64, 8, 2048)]:
            generator = torch.Generator().manual_seed(353)
            tokens = torch.arange(rows).repeat_interleave(top_k)
            permutation = torch.randperm(tokens.numel(), generator=generator)
            tokens = tokens[permutation]
            slots = torch.randn(rows * top_k, width, generator=generator)
            order = tokens.argsort(stable=True).view(rows, top_k)
            initial = torch.randn(rows, width, generator=generator)
            expected = torch.zeros_like(initial)
            for k in range(top_k):
                expected += slots[order[:, k]]
            expected += initial
            for _ in range(3):
                output = initial.cuda()
                self.extension.gather_slots(slots.cuda(), order.cuda(), output)
                self.assertTrue(torch.equal(output.cpu(), expected))

    def test_recurrent_repeatability_and_native_formula(self):
        torch.manual_seed(353)
        with torch.inference_mode():
            for batch, sequence, history in [(1, 1, False), (1, 4, True), (2, 4, False), (2, 4, True)]:
                nk, nv = 2, 4
                qkv = torch.randn(batch, sequence, (2 * nk + nv) * 128, device='cuda', dtype=torch.bfloat16)
                g = -torch.rand(batch, sequence, nv, device='cuda')
                beta = torch.rand(batch, sequence, nv, device='cuda', dtype=torch.bfloat16)
                state = torch.randn(batch + 1, sequence if history else 1, nv, 128, 128, device='cuda') * .01
                slots = torch.arange(batch, 0, -1, dtype=torch.int32, device='cuda')
                args = [qkv, g, beta, state, torch.zeros(batch, sequence, nv, 128, device='cuda', dtype=torch.bfloat16), nk, nv, 128, 128, slots, history]
                def run(fn):
                    values = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
                    fn(*values)
                    return values[3].cpu(), values[4].cpu()
                reference = run(native.cuda_recurrent_gated_delta_rule)
                first = run(self.extension.recurrent)
                for _ in range(20):
                    result = run(self.extension.recurrent)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, result)))
                for a, b, tolerance in zip(reference, first, [1e-5, 1e-3]):
                    relative = (a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-12)
                    self.assertLess(float(relative), tolerance)

    def test_activation_is_explicit_restores_and_rejects_training(self):
        original = {name: getattr(native, name) for name in dir(native)}
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with enabled(self.extension):
                changed = {name for name, value in original.items() if getattr(native, name) is not value}
                self.assertEqual(changed, {'cuda_recurrent_gated_delta_rule', 'exl3_moe'})
                with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, 'inference-only'):
                    native.cuda_recurrent_gated_delta_rule()
                raise RuntimeError('intentional')
        self.assertTrue(all(getattr(native, name) is value for name, value in original.items()))


if __name__ == '__main__':
    unittest.main()
