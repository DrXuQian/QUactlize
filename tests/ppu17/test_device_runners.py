"""CPU-only checks for physical-device evidence and isolation guards."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2] / "quactlize/backends/ppu17"
sys.path.insert(0, str(HERE))
from check_device import cases
from check_memory import cases as memory_cases
from run_benchmark import benchmark_points, require_idle, run_point, validate_result


class Runners(unittest.TestCase):
    def test_bounded_reconfirmation(self):
        self.assertEqual(len(benchmark_points("baseline")), 36)
        selected = benchmark_points("baseline", [0, 1, 5])
        self.assertEqual(len(selected), 18)
        self.assertEqual(benchmark_points("baseline", [0, 1, 5], True), selected[::-1])
        for invalid in ([], [0, 0], [6], [-1]):
            with self.assertRaises(ValueError): benchmark_points("baseline", invalid)

    def test_memory_coverage(self):
        rows = memory_cases()
        self.assertEqual(len(rows),25)
        self.assertEqual({p["tool"] for p in rows},{"memcheck","racecheck","synccheck"})
        for q in (8,10,11,12,13,14):
            for compute in (0,1):
                self.assertTrue(any(p["tool"] == "memcheck" and
                    p["point"]["q"] == q and p["point"]["compute"] == compute and
                    p["point"]["config"] == 5 for p in rows))

    def test_coverage(self):
        self.assertEqual(len(cases("coverage")), 300)
        rows = cases("extended")
        self.assertEqual(len(rows), 1020)
        self.assertEqual({p["q"] for p in rows}, {8,10,11,12,13,14})
        for q in (8,10,11,12,13,14):
            for compute in (0,1):
                found = {p["m"] for p in rows if p["q"] == q and p["compute"] == compute
                         and p["rows"] is None and p["config"] == 5}
                self.assertTrue(set(range(1,9)) <= found)

    def test_idle(self):
        sample = {"uuid":"GPU-A", "processes":[], "gpu_util":0}
        require_idle(sample,"GPU-A")
        for delta in ({"uuid":"GPU-B"}, {"processes":[{"pid":9}]}, {"gpu_util":1}):
            with self.assertRaises(ValueError): require_idle({**sample, **delta},"GPU-A")

    def test_result(self):
        row = dict(correctness="CPU_DOT_AND_GUARDS_PASS", scope="PHYSICAL_HOPPER_GRAPH_AVERAGE_NOT_PPU17",
                   samples_us=[1.]*11, median_us=1., launches_per_graph=36, weight_copies=3,
                   cache="rotating", weight_bytes=100, l2_bytes=100)
        validate_result(row)
        for delta in ({"correctness":"FAIL"}, {"samples_us":[float("nan")]*11},
                      {"samples_us":[1.]*10}, {"median_us":2.}, {"launches_per_graph":32},
                      {"weight_copies":2}, {"scope":"PPU17"}):
            with self.assertRaises(ValueError): validate_result({**row, **delta})

    def test_foreign_job_stops_only_child(self):
        root = os.environ.get("Q17_TEST_OUTPUT")
        if not root: self.skipTest("set Q17_TEST_OUTPUT to an explicit /workspace directory")
        idle = {"uuid":"GPU-A", "processes":[], "gpu_util":0}
        busy = {**idle, "processes":[{"pid":987654321,"name":"foreign"}]}
        with tempfile.TemporaryDirectory(dir=root) as folder:
            with patch("run_benchmark.snapshot", side_effect=[idle,idle,idle,busy,busy]):
                result = run_point([sys.executable,"-c","import time; time.sleep(30)"],os.environ,
                                   Path(folder)/"log","0","GPU-A",5)
        self.assertEqual(result["status"],"INVALID")
        self.assertIn("CONTAMINATED",result["error"])
        self.assertNotEqual(result["returncode"],0)


if __name__ == "__main__":
    unittest.main()
