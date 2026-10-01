"""Host contracts for the bounded first-B-tile experiment, no device needed."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).parent / "pdl_prefetch/run.py"
spec = importlib.util.spec_from_file_location("pdl_prefetch_run", path)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class PrefetchContracts(unittest.TestCase):
    def test_inventory(self):
        points = experiment.correctness_points()
        self.assertEqual(len(points), 96)
        self.assertEqual(len({tuple(p.items()) for p in points}), 96)
        self.assertEqual({p["q"] for p in points}, {8,10,11,12,13,14})
        self.assertEqual({p["compute"] for p in points}, {0,1})
        self.assertEqual({p["route"] for p in points}, {0,1})
        for p in points:
            self.assertEqual(p["k"] % (512 if p["q"] in (11,14) else 256), 0)
        self.assertTrue(any(p["rows"] == "0,0,1,0" for p in points))
        self.assertTrue(any(p["n"] == 48 and p["k"] == 1024 for p in points))
        self.assertEqual(len(experiment.timing_points()), 24)

    def test_chain_receipt(self):
        point = experiment.correctness_points()[0]
        line = ("PPU17_PDL_CHAIN PASS q=8 compute=0 route=0 m=1 n=256 k=512 "
                "input=0 output=2 config=5 grouped=0 comparisons=36 prefetch=1 "
                "raw_equal=1 cpu_oracle=PASS stream=PASS graph=PASS timing=NOT_MEASURED")
        experiment.validate_chain(line, point)
        for old, new in (("m=1 ","m=2 "), ("comparisons=36","comparisons=24"),
                         ("prefetch=1","prefetch=0"), ("n=256","n=48"),
                         ("raw_equal=1","raw_equal=0"), ("cpu_oracle=PASS","cpu_oracle=SKIP")):
            with self.assertRaises(ValueError):
                experiment.validate_chain(line.replace(old,new),point)
        with self.assertRaises(ValueError):
            experiment.validate_chain(line+"\n"+line, point)

    def test_timing_receipt(self):
        point = experiment.timing_points()[0]
        row = dict(q=12, compute=0, route=0, m=1, n=512, k=2048, experts=1,
                   cache="warm", raw_equal=True, producer_config=2, consumer_config=5,
                   metric_scope="GEMM_OPTIONAL_SF_GEMM_CHAIN", warmup_graphs_per_arm=3,
                   arms={name:dict(median_us=2., samples_us=[2.]*11)
                         for name in ("ordinary","protocol","prefetch")})
        experiment.validate_timing(row, point)
        for key, value in (("raw_equal",False), ("m",2), ("metric_scope","CONSUMER_ONLY")):
            with self.assertRaises(ValueError):
                experiment.validate_timing({**row, key:value},point)
        row["arms"]["prefetch"]["samples_us"][0] = float("nan")
        with self.assertRaises(ValueError):
            experiment.validate_timing(row, point)


if __name__ == "__main__":
    unittest.main()
