"""Host contracts for the bounded real-model GEMV experiment (no device)."""

from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dev.gemv_model.plan import POINTS, candidates, inventory
from dev.gemv_model.source import source, direct_body, once
from dev.gemv_model.fixture import varied_weight, expert_payload
from dev.gemv_model.access import access
from dev.gemv_model.run import summarize, profile_records, logged
from dev.gate_up_perf.bench import physical
from dev.bf16_compute.fixture import planes, round_compute
from reference import gguf_kpack as ref


class ModelGemvTests(unittest.TestCase):
    def test_inventory_bounded_and_current_winners_retained(self):
        self.assertEqual(len(POINTS), 8)
        self.assertEqual(sum(len(candidates(p)) for p in POINTS), 68)
        for p in POINTS:
            cs = candidates(p)
            self.assertEqual(len({c.name for c in cs}), len(cs))
            self.assertLessEqual(len(cs), 11)
            for c in cs:
                self.assertEqual(p.physical_n % c.tile_n, 0)
                if p.paired:
                    self.assertIn(c.tile_n, (16, 32))
                    self.assertEqual(c.split, 1)
                if c.direct_meta:
                    self.assertEqual(p.q, 14)
            if not p.tc:
                self.assertEqual(cs[0].name, "clone")
        self.assertEqual(POINTS[1].compute, 1)
        self.assertTrue(all(p.compute == 0 for p in POINTS if not p.mode))

    def test_exact_tc_controls(self):
        self.assertEqual([p.tc[-1] for p in POINTS if p.tc], [8, 8, 1])
        self.assertEqual(POINTS[-1].parent["route"], "fq-dense")
        self.assertEqual(POINTS[-2].parent["route"], "sf-dense")

    def test_wrapper_checks_shape_before_specializing(self):
        for p in POINTS:
            text = source(p)
            self.assertIn(f"c.n!={p.n} || c.k!={p.k}", text)
            self.assertIn(f"d.compute_type!={p.compute}", text)
            self.assertIn("c.input_type!=QKG_F32", text)
            self.assertEqual(text.count("case "), len(candidates(p)))
            if p.paired:
                self.assertLess(text.index("Row locate"), text.index('#include "quactlize/execution/simt_q8_vector.cuh"'))
                self.assertIn("row_buffers(*fusion,sizes)", text)
                self.assertIn("call.input_rows=fusion->input_rows", text)
                self.assertIn("SimtFinish", text)
            else:
                self.assertIn("buffers_v2(d,sizes)", text)

    def test_q6_only_metadata_block_changes(self):
        root = Path(__file__).resolve().parents[1]
        original = (root / "quactlize/execution/simt_kernel.cuh").read_text()
        direct = direct_body()
        suffix = original[original.index("        uint4 packet{};"):original.index("\ntemplate<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0>\n__global__")]
        self.assertTrue(direct.endswith(suffix))
        self.assertNotIn("cooperative", direct)
        self.assertIn("d*float(scale),0.f", direct)
        self.assertNotIn("cutlass::half_t", direct)

    def test_direct_q6_bytes_all_fields(self):
        raw, _ = varied_weight(14, 256, 512, 37)
        packed = planes(raw, 14)
        units = packed["units"].reshape(-1)
        spec = ref.SPECS[14]
        seen = set()
        for n in (0, 1, 7, 127, 255):
            for g in range(32):
                ptr = ((g // 32) * 256 + n) * 36 + ((g // 16) & 1) * 18
                d = units[ptr:ptr + 2].copy().view("<f2")[0]
                sc = units[ptr + 2 + (g & 15):ptr + 3 + (g & 15)].view("i1")[0]
                sb = g // 16
                expected_d = raw[n, sb, spec.d_offset:spec.d_offset + 2].copy().view("<f2")[0]
                expected_sc = raw[n, sb, spec.scale_offset + g % 16:spec.scale_offset + g % 16 + 1].view("i1")[0]
                self.assertEqual(float(d)*int(sc), float(expected_d)*int(expected_sc))
                seen.add(int(sc))
        self.assertEqual(seen, set(range(-3, 4)))

    def test_bitwide_q6_signed_scale_contract(self):
        # Boundary values omitted by the performance-friendly dyadic fixture.
        values = np.arange(256, dtype="u1")
        signed = values.view("i1")
        decoded = values.astype("i4")
        decoded[decoded >= 128] -= 256
        np.testing.assert_array_equal(signed, decoded)
        for d in np.array([0., -0., 2**-14, 2**-4, -2**-4], "f2"):
            np.testing.assert_array_equal(d.astype("f4") * signed.astype("f4"),
                                          d.astype("f4") * decoded.astype("f4"))

    def test_paired_lane_ownership(self):
        gate, up = np.arange(512), np.arange(512) + 10000
        values = physical(gate, up, True)
        owner = np.arange(512) // 4 * 8 + np.arange(512) % 4
        np.testing.assert_array_equal(values[owner], gate)
        np.testing.assert_array_equal(values[owner + 4], up)
        for tile_n in (16, 32):
            written = []
            for tile in range(1024 // tile_n):
                for tid in range(tile_n // 2):
                    logical = tile * tile_n // 2 + tid
                    ng = tile * tile_n + tid // 4 * 8 + tid % 4
                    self.assertEqual(values[ng], gate[logical])
                    self.assertEqual(values[ng + 4], up[logical])
                    written.append(logical)
            self.assertEqual(written, list(range(512)))

    def test_half_tile_warp_reduce_and_paired_finish_tags(self):
        # Emulate the actual shuffle ownership, not just logical G/U indices.
        for columns, width in ((4, 4), (4, 8), (8, 4)):
            tile = columns * width
            lane = np.arange(32)
            original = np.array([[1000 * (l // columns) + (l % columns) * width + p
                                  for p in range(width)] for l in lane], dtype="f4")
            values, count, stride = original.copy(), width, columns
            while count > 1:
                odd = (lane & stride) != 0
                keep = np.where(odd[:, None], values[:, 1:count:2], values[:, :count:2])
                send = np.where(odd[:, None], values[:, :count:2], values[:, 1:count:2])
                values[:, :count // 2] = keep + send[lane ^ stride]
                count //= 2
                stride *= 2
            reduced = values[:, 0]
            while stride < 32:
                reduced = reduced + reduced[lane ^ stride]
                stride *= 2
            partial = np.empty(tile, "f4")
            for l in range(tile):
                partial[(l % columns) * width + l // columns] = reduced[l]
            expected = np.array([original[np.arange(c // width, 32, columns), c % width].sum()
                                 for c in range(tile)], "f4")
            np.testing.assert_array_equal(partial, expected)
            for tid in range(tile // 2):
                ng = tid // 4 * 8 + tid % 4
                self.assertEqual(partial[ng], expected[ng])
                self.assertEqual(partial[ng + 4], expected[ng + 4])

    def test_factor_oracle_and_chunk_assembly(self):
        for q in (8, 12, 13, 14):
            base = next(p for p in POINTS if p.q == q)
            p = replace(base, n=512, k=512, paired=False)
            category = np.random.default_rng(3).integers(0, 4, 512)
            packed, sums, absolute, _ = expert_payload(p, 0, category)
            raw, gold = varied_weight(q, 512, 512, 181)
            expected = planes(raw, q)
            for key in ("low", "high", "units"):
                np.testing.assert_array_equal(packed[key], expected[key])
            coeff = np.array([.125, -.5, .25, .0625], "f8")
            np.testing.assert_allclose(coeff @ sums, gold.astype("f8") @ coeff[category], rtol=0, atol=0)
            np.testing.assert_allclose(abs(coeff) @ absolute, abs(gold).astype("f8") @ abs(coeff[category]), rtol=0, atol=0)

    def test_exact_precision_and_bf16_range(self):
        for q in (8, 12, 13, 14):
            _, gold = varied_weight(q, 256, 512, 23)
            for compute in ("bf16", "f16"):
                np.testing.assert_array_equal(gold, round_compute(gold, compute))
        self.assertTrue(np.isfinite(round_compute(np.array([243383.484375], "f4"), "bf16")).all())

    def test_direct_metadata_access(self):
        p = POINTS[-1]
        a = access(p, candidates(p)[1], dict(A=0, low=0, high=0, units=0))
        metadata = [s for s in a["streams"] if s["name"].startswith("metadata-")]
        self.assertEqual(len(metadata), 8)
        self.assertEqual({s["width_bytes"] for s in metadata}, {1, 2})
        self.assertTrue(all(s["name"].startswith("metadata-direct-") for s in metadata))
        self.assertEqual(a["B_contiguous_bytes_per_k_worker_group"], 64)
        self.assertIn("SOURCE", a["scope"])

    def test_split_reducer_is_not_omitted(self):
        p = POINTS[4]
        text = source(p)
        self.assertEqual(text.count("register_reuse_reduce<8>"), 6)
        self.assertIn("quactlize::decode::reduce_decode<8>", text)
        self.assertIn("c.rows==1", text)
        self.assertIn("uintptr_t(c.output)|uintptr_t(c.workspace)", text)

    def test_source_seam_fails_closed(self):
        for text in ("", "x x"):
            with self.assertRaises(ValueError):
                once(text, "x", "y")

    def test_samples_and_model_not_dram(self):
        samples = [[float(i + 1) for i in range(15)] for _ in range(6)]
        r = summarize(samples, 2160000, 40)
        self.assertEqual(r["median_us"], 8)
        self.assertAlmostEqual(r["modeled_weight_MBU_pct"], 10)
        for bad in (samples[:-1], [[1.]*14]*6, [[float("nan")]*15]*6, [[0.]*15]*6):
            with self.assertRaises(ValueError):
                summarize(bad, 100, 40)

    def test_acu_full_call_and_symbol_negatives(self):
        header = '"ID","Kernel Name","Block Size","Grid Size"\n'
        p = POINTS[4]
        row = '"0","void quactlize::execution::model_gemv::point_8_2048_4096_arm_0(qkg_call_v1)","(128, 1, 1)","(512, 1, 1)"\n'
        reducer = '"1","void quactlize::decode::reduce_decode<8>(float*,float*,int)","(32,1,1)","(32,1,1)"\n'
        self.assertEqual(len(profile_records(header+row+reducer, p, "0")), 2)
        for text in (header+row, header+row.replace("arm_0", "arm_1")+reducer,
                     header+row.replace("128, 1", "256, 1")+reducer, header+row+reducer+reducer,
                     header+row+reducer.replace("reduce_decode", "gather")):
            with self.assertRaises(ValueError):
                profile_records(text, p, "0")

    def test_progress_tail_no_buffering_failure(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            rc = logged([sys.executable, "-c", "print('MODEL_GEMV_PROBE PASS')"], log, "test", True)
            self.assertEqual(rc, 0)

    def test_direct_script_imports(self):
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        for name in ("run.py", "build.py", "inspect_native.py"):
            subprocess.run([sys.executable, str(root / "dev/gemv_model" / name), "--help"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    def test_performance_does_not_add_compact_mapping_load(self):
        import ctypes as C
        from types import SimpleNamespace
        from dev.gemv_model.engine import Provider
        from quactlize.execution.native import Call
        from quactlize.fusion.native import Layout, MappedCall
        provider = Provider.__new__(Provider)
        provider.b = SimpleNamespace(point=POINTS[0], mapping=SimpleNamespace(ptr=256),
                                     status=SimpleNamespace(ptr=512), mapped_call=True,
                                     call=lambda copy: Call())
        provider.arm, provider.layout = "0", Layout()
        observed = []
        def run(_, arg, layout, arm):
            d = C.cast(arg, C.POINTER(MappedCall)).contents
            observed.append((d.input_rows, d.status))
            return 0
        provider.fn = run
        provider.prepare()()
        provider.b.mapped_call = False
        provider.prepare()()
        self.assertEqual(observed, [(256, 512), (None, 512)])


if __name__ == "__main__":
    unittest.main()
