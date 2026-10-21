import copy
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dev.prefill_point.bench import Call, SF, WORKLOAD, call, summarize
from dev.prefill_point.build import build_native, carrier_source, once
from tools.run_prefill_point import ARMS, native_stage_names, plan

ROOT = Path(__file__).resolve().parents[1]


class PrefillPointTests(unittest.TestCase):
    def rows(self):
        return [dict(round=r, rotate=rotate, mode=mode, stage=stage,
                     width=4 if mode == 1 else 1,
                     samples_us=([1., 2., 3., 4.]*3 if mode == 1 else [10.]*3))
                for r in range(3) for rotate in (0, 1) for mode in range(3)
                for stage in range(4 if mode == 0 else 1)]

    def test_fixed_point_not_search(self):
        self.assertEqual((WORKLOAD.qtype, WORKLOAD.m, WORKLOAD.n, WORKLOAD.k), (12, 2048, 1024, 5120))
        self.assertEqual(ARMS, ('native-fp16', 'sf', 'full-bf16'))
        self.assertEqual(dict(SF.parameters)['split'], 1)
        self.assertEqual(dict(SF.parameters)['dequant_config'], 4)
        self.assertEqual(plan()['full_dequant_config'], 12)
        for name in ('production_changed', 'performance_admitted', 'cuda_graph_timing', 'cublas_graph'):
            self.assertFalse(plan()[name])

    def test_descriptor_layout(self):
        self.assertEqual(ctypes.sizeof(Call), 136)
        self.assertEqual(Call.kind.offset, 96)
        self.assertEqual(Call.stride.offset, 128)
        c = call(None, 3, [42, ctypes.c_void_p(19)], [2, 7], stride=5120)
        self.assertEqual(list(c.p)[:2], [42, 19])
        self.assertEqual(list(c.v)[:2], [2, 7])
        self.assertEqual(c.stride, 5120)

    def test_diagnostic_keeps_oracles_not_old_timers(self):
        text = carrier_source()
        self.assertIn('#include "carrier_timing.cuh"', text)
        self.assertIn('last_weight_copy_negative', text)
        self.assertIn('omitted_producer', text)
        self.assertIn('nc.replay(nc.session,copy,-1)', text)
        self.assertNotIn('nc.measure(nc.session,int(i),rotate', text)
        self.assertEqual(text.count('NC_BIND(point_samples)'), 1)
        with self.assertRaises(ValueError): once('two two', 'two', 'one')

    def test_payload_reuse_builds_native_adapter_in_new_run(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);previous=root/'previous';previous.mkdir()
            old=previous/'native';old.mkdir()
            receipt=old/'adapter.json';receipt.write_text('old receipt remains immutable')
            args=SimpleNamespace(reuse_build=previous,llama_source=root/'llama',
                                 llama_build=root/'build',sdk=root/'sdk')
            target=root/'new'/'native'
            # Stop at the real build seam: never validate or rewrite the old
            # control-script identity merely to reuse unchanged GPU payloads.
            with patch('dev.prefill_point.build.build_adapter',side_effect=RuntimeError('test build seam'))as build:
                with self.assertRaisesRegex(RuntimeError,'test build seam'):
                    build_native(args,target,{})
                build.assert_called_once_with(args.llama_source,args.llama_build,args.sdk,target)
            self.assertEqual(receipt.read_text(),'old receipt remains immutable')

    def test_summary_requires_complete_three_round_inventory(self):
        rows = self.rows()
        out = summarize(rows, ['input', 'expansion', 'main', 'output'])
        self.assertEqual(out['repeated']['sequence_stage_us_sum'], 10)
        self.assertEqual(out['repeated']['isolated_us_sum'], 40)
        self.assertEqual(out['rotating_weights']['complete_us'], 10)
        bads = [rows[:-1], rows+[rows[0]], [r for r in rows if r['round'] != 2]]
        for name, value in [('width', 7), ('samples_us', [float('nan')]*3), ('samples_us', [0.]*3)]:
            bad = copy.deepcopy(rows); bad[0][name] = value; bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError): summarize(bad, ['input', 'expansion', 'main', 'output'])

    def test_native_inventory_matches_actual_ppu_names(self):
        names = ['void convert_cont_ppu_kernel<float, __half>(...)',
                 'void dequantize_block_q4_K(...)', 'cublasGemmEx',
                 'void convert_cont_ppu_kernel<__half, float>(...)']
        rows = [dict(name=n, symbol=n, entry='PROVIDER' if n == 'cublasGemmEx' else 'KERNEL') for n in names]
        self.assertEqual(native_stage_names(rows), ['input', 'expansion', 'main', 'output'])
        with self.assertRaises(ValueError): native_stage_names(rows[::-1])
        with self.assertRaises(ValueError): native_stage_names([dict(name='unknown', symbol='unknown', entry='KERNEL')])

    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler unavailable')
    def test_native_event_loop_order_and_error_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td)/'point-events'
            subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            str(ROOT/'tests/prefill_point_events_host.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
