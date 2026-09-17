"""Host-only negatives for the immutable model GEMV result audit."""

from copy import deepcopy
from dataclasses import asdict
import unittest

from dev.gemv_model.plan import POINTS, candidates
from tools.review_model_gemv import check_finalists, check_selection, close, raw_rows


class ModelGemvReviewTests(unittest.TestCase):
    def test_finite_comparison(self):
        self.assertTrue(close(1., 1.))
        for a, b in ((float("nan"), 1.), (float("inf"), float("inf")), (1., 1.1)):
            self.assertFalse(close(a, b))

    def test_selection_not_just_a_display_label(self):
        p = POINTS[0]
        config = candidates(p)[3]
        receipt = dict(arm="3", name=config.name, kind="simt", config=asdict(config),
                       scope="PAIRED_GATE_UP_PLUS_SWIGLU")
        check_selection(p, "3", receipt)
        for field, value in (("fixed", False), ("split", 8), ("changes", 0)):
            bad = deepcopy(receipt)
            bad["config"][field] = value
            with self.assertRaisesRegex(ValueError, "selection differs"):
                check_selection(p, "3", bad)

    def test_tc_incumbent_keeps_parent_and_split(self):
        p = POINTS[5]
        receipt = dict(arm="incumbent", name="incumbent", kind="tc",
                       config=p.parent | dict(split=8), scope="GEMV_INCLUDING_SPLITK_REDUCTION")
        check_selection(p, "incumbent", receipt)
        receipt["config"]["split"] = 1
        with self.assertRaisesRegex(ValueError, "selection differs"):
            check_selection(p, "incumbent", receipt)

    def test_finalist_and_winner_denominators(self):
        p = POINTS[0]
        data = dict(screen_us={str(i): [i + 1.] * 3 for i in range(len(candidates(p)))},
                    summary={"incumbent": {"median_us": 10.}, "1": {"median_us": 7.},
                             "2": {"median_us": 6.}}, best_candidate="2")
        check_finalists(p, data)
        bad = deepcopy(data)
        bad["best_candidate"] = "1"
        with self.assertRaisesRegex(ValueError, "winner differs"):
            check_finalists(p, bad)
        bad = deepcopy(data)
        bad["summary"]["3"] = bad["summary"].pop("2")
        with self.assertRaisesRegex(ValueError, "finalists differ"):
            check_finalists(p, bad)

    def test_raw_csv_ignores_only_profiler_preamble(self):
        text = 'Import complete\n"ID","Kernel Name","metric"\n"0","kernel",12\n'
        self.assertEqual(raw_rows(text), [{"ID": "0", "Kernel Name": "kernel", "metric": "12"}])
        with self.assertRaisesRegex(ValueError, "header absent"):
            raw_rows("Profiler crashed before export")


if __name__ == "__main__":
    unittest.main()
