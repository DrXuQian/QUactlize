"""CPU-only layout/contract tests; CUDA kernels are never launched here."""
import argparse
import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from compare import Reference, check_output, validate
from fixture import make_fixture, marlin_metadata, unpack_canonical


class FixtureTests(unittest.TestCase):
    def test_canonical_all_codes_and_metadata(self):
        for dtype in (torch.float16, torch.bfloat16):
            fixture = make_fixture(8, 512, 512, dtype)
            self.assertEqual(set(fixture['q'].reshape(-1)), set(range(16)))
            q, scales, minimum = unpack_canonical(fixture['low'], fixture['units'], 512, 512)
            self.assertTrue(np.array_equal(q, fixture['q']))
            self.assertGreater(np.count_nonzero(minimum), 0)
            self.assertEqual(len(np.unique(scales)), 6)
            zero_code = fixture['minimum'] / fixture['scale']
            self.assertTrue(torch.equal(zero_code * fixture['scale'], fixture['minimum']))

    def test_corruption_and_nonfinite_rejected(self):
        fixture = make_fixture(1, 128, 256, torch.float16)
        expected = fixture['expected']
        storage = torch.full((64+128+64,), -1232., dtype=torch.float16)
        storage[64:192] = expected.reshape(-1)
        check_output(torch, storage, expected, 1, 128)
        storage[65] += 2
        with self.assertRaisesRegex(ValueError, 'dot mismatch'):
            check_output(torch, storage, expected, 1, 128)
        storage[65] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            check_output(torch, storage, expected, 1, 128)
        storage[0] = 0
        with self.assertRaisesRegex(ValueError, 'guard'):
            check_output(torch, storage, expected, 1, 128)

    def test_bad_dimensions_and_timing_rejected(self):
        with self.assertRaises(ValueError):
            make_fixture(1, 129, 256, torch.float16)
        with self.assertRaises(ValueError):
            validate({'status': 'PASS', 'rows': []})
        perm = marlin_metadata(torch.arange(128).reshape(2, 64))
        self.assertEqual(perm[0, :8].tolist(), list(range(0, 64, 8)))


def upstream_marlin_check(root, references):
    """Execute only pinned upstream CPU pack functions, without importing vLLM."""
    paths = root/'vllm/model_executor/layers/quantization/utils'
    env = {'torch': torch, 'np': np, 'GPTQ_MARLIN_TILE': 16,
           'get_pack_factor': lambda bits: 32//bits}
    names = {'get_weight_perm', 'marlin_weights', 'marlin_permute_weights',
             'get_scale_perms', 'marlin_permute_scales'}
    for name in ('marlin_utils_test.py', 'marlin_utils.py'):
        tree = ast.parse((paths/name).read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(paths/name), 'exec'), env)
    ref = Reference(references, 'marlin')
    for dtype_id, dtype in enumerate((torch.float16, torch.bfloat16)):
        for n, k in ((128, 256), (512, 512)):
            fixture = make_fixture(1, n, k, dtype)
            q = fixture['q']
            got = np.zeros(n*k//2, dtype=np.uint8)
            ref.check(ref.fn['pack'](dtype_id, n, k, q.ctypes.data, got.ctypes.data))
            expected = env['marlin_weights'](torch.from_numpy(q), k, n, 4, env['get_weight_perm'](4))
            if not np.array_equal(expected.numpy().view(np.uint8).reshape(-1), got):
                raise ValueError('standalone Marlin pack differs from pinned upstream CPU pack')
            for value in (fixture['scale'], fixture['minimum']):
                expected = env['marlin_permute_scales'](value, k, n, 64)
                if not torch.equal(marlin_metadata(value), expected):
                    raise ValueError('Marlin metadata differs from upstream')
    print('PPU17_REFERENCE_HOST Marlin pack/metadata matches pinned upstream, cases=4', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vllm', type=Path)
    p.add_argument('--references', type=Path)
    args = p.parse_args()
    torch.set_num_threads(8)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.vllm:
        upstream_marlin_check(args.vllm, args.references)
