"""Policy-tag normalization must be exact and never normalize instructions."""
import unittest

from compare_ordinary_codegen import match_parent


class OrdinaryCodegenContracts(unittest.TestCase):
    def test_disabled_policy_tags(self):
        old = {"void q::tma_wgmma<T>(int)": "body"}
        for suffix in ("", ", false", ", false, false"):
            candidate = {f"void q::tma_wgmma<T{suffix}>(int)": "body"}
            self.assertEqual(match_parent(candidate, old), old)
        pdl = {"void q::tma_wgmma<T, false>(int)": "body"}
        self.assertEqual(match_parent({"void q::tma_wgmma<T, false, false>(int)": "body"}, pdl), pdl)

    def test_reject_unknown_enabled_or_excess_tags(self):
        old = {"void q::tma_wgmma<T>(int)": "body"}
        for suffix in (", true", ", false, false, false"):
            with self.assertRaises(ValueError):
                match_parent({f"void q::tma_wgmma<T{suffix}>(int)": "body"}, old)
        with self.assertRaises(ValueError):
            match_parent({"void q::simt<T, false, false>(int)": "body"}, {"void q::simt<T>(int)": "body"})

    def test_ambiguous_duplicate_and_changed_code(self):
        old = {"void q::tma_wgmma<T>(int)": "before"}
        candidate = {"void q::tma_wgmma<T, false>(int)": "after"}
        self.assertNotEqual(match_parent(candidate, old), old)
        with self.assertRaises(ValueError):
            match_parent(candidate, {**old, "void q::tma_wgmma<T, false>(int)": "before"})
        with self.assertRaises(ValueError):
            match_parent({**candidate, "void q::tma_wgmma<T>(int)": "after"}, old)


if __name__ == "__main__":
    unittest.main()
