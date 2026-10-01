"""CPU-only contracts for opt-in PDL and its per-entry codegen/device gates."""
import ctypes as C
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[2] / "quactlize/backends/ppu17"
sys.path.insert(0, str(HERE))
from check import inspect, inspect_dependency_sass
from check_pdl import arguments, cases, validate_result
from native import Backend, LaunchOptions


def ptx(pdl):
    lines = [".target sm_90a"]
    for family, count in (("wgmma", 12), ("tma_wgmma", 4), ("simt", 8), ("expand_scales", 2)):
        for enabled in ((False, True) if pdl else (False,)):
            for index in range(count):
                lines.append(f".visible .entry {family}ILi10E_{index}_{enabled}() {{")
                if enabled:
                    lines.append("griddepcontrol.wait;")
                lines.append("ld.global.u32;")
                if "wgmma" in family:
                    lines += ["wgmma.mma_async.f32.f16.f16;", "wgmma.wait_group;"]
                if family == "tma_wgmma":
                    lines.append("cp.async.bulk.tensor;")
                if family == "simt":
                    lines.append("fma.rn.f32;")
                if enabled:
                    lines.append("griddepcontrol.launch_dependents;")
                lines += ["st.global.u32;", "}"]
    return "\n".join(lines)


class PdlContracts(unittest.TestCase):
    def test_optional_binary_abi_without_device(self):
        enabled = os.environ.get("Q17_PDL_TEST_BUNDLE")
        ordinary = os.environ.get("Q17_ORDINARY_TEST_BUNDLE")
        parent = os.environ.get("Q17_PARENT_TEST_BUNDLE")
        if not all((enabled, ordinary, parent)):
            self.skipTest("set the three Q17_*TEST_BUNDLE paths for the host-only binary ABI check")
        for folder, compiled in ((enabled, True), (ordinary, False)):
            backend = Backend(folder, allow_source_validation=True)
            self.assertEqual(backend.pdl_compiled, compiled)
            options = LaunchOptions(programmatic_dependency=1)
            handle = C.c_void_p(123)
            # NULL call is deliberately invalid: all checks must return before
            # querying a device or initializing a context.
            rc = backend.functions["prepare_with_options"](None, C.byref(options), C.byref(handle))
            self.assertEqual(rc, 1 if compiled else 2)
            self.assertIsNone(handle.value)
            options.reserved = 1
            self.assertEqual(backend.functions["prepare_with_options"](None, C.byref(options), C.byref(handle)), 1)
        old = Backend(parent, allow_source_validation=True)
        self.assertFalse(old.pdl_compiled)
        with self.assertRaisesRegex(ValueError, "PDL was not compiled"):
            old.prepare(None, pdl=True)

    def test_abi_defaults(self):
        ordinary = LaunchOptions()
        self.assertEqual(C.sizeof(ordinary), 16)
        self.assertEqual((ordinary.version, ordinary.struct_bytes, ordinary.programmatic_dependency, ordinary.reserved),
                         (1, 16, 0, 0))

    def test_every_specialization(self):
        self.assertEqual(sum(inspect(ptx(False), 10).values()), 26)
        self.assertEqual(sum(inspect(ptx(True), 10, True).values()), 52)
        for changed in (ptx(True).replace("griddepcontrol.wait;", "", 1),
                        ptx(True).replace("griddepcontrol.launch_dependents;", "", 1),
                        ptx(True).replace("griddepcontrol.wait;\nld.global.u32;", "ld.global.u32;\ngriddepcontrol.wait;", 1)):
            with self.assertRaises(ValueError):
                inspect(changed, 10, True)
        with self.assertRaises(ValueError):
            inspect(ptx(False), 10, True)
        with self.assertRaises(ValueError):
            inspect(ptx(True), 10, False)

    def test_assembled_protocol_is_per_entry(self):
        source = ".visible .entry ordinary() {st.global;}\n.visible .entry pdl() {griddepcontrol.wait; griddepcontrol.launch_dependents;}"
        native = "Function : ordinary\nSTG.E;\nFunction : pdl\nACQBULK; PREEXIT; STG.E;"
        inspect_dependency_sass(native, source)
        for changed in (native.replace("ACQBULK", "NOP"), native.replace("PREEXIT", "NOP"),
                        native.replace("STG.E", "ACQBULK; PREEXIT; STG.E", 1)):
            with self.assertRaises(ValueError):
                inspect_dependency_sass(changed, source)

    def test_coverage(self):
        smoke, coverage = cases("smoke"), cases("coverage")
        self.assertEqual(len(smoke), 52)
        self.assertEqual(len(coverage), 476)
        self.assertEqual({p["q"] for p in coverage}, {8, 10, 11, 12, 13, 14})
        self.assertEqual({p["config"] for p in coverage if p["q"] == 12}, set(range(11)))
        self.assertEqual({p["m"] for p in coverage if not p["grouped"]}, {1, 2, 3, 4, 5, 6, 7, 8, 65})
        for p in coverage:
            cmd = arguments("pdl-chain", p)
            self.assertEqual("--sf" in cmd, bool(p["route"]))
            self.assertEqual("--rows" in cmd, p["grouped"])
        self.assertEqual(len({tuple(p.items()) for p in coverage}), len(coverage))

    def test_receipt(self):
        point = cases("smoke")[0]
        line = ("PPU17_PDL_CHAIN PASS q=12 compute=0 input=0 output=2 config=0 route=0 grouped=0 "
                "comparisons=24 raw_equal=1 cpu_oracle=PASS stream=PASS graph=PASS timing=NOT_MEASURED backend=test")
        validate_result(line, point)
        for changed in (line + "\n" + line, line.replace("comparisons=24", "comparisons=1"),
                        line.replace("q=12", "q=14"), line.replace("raw_equal=1", "raw_equal=0"),
                        line.replace("graph=PASS", "graph=FAIL"), line.replace("cpu_oracle=PASS", "cpu_oracle=SKIP")):
            with self.assertRaises(ValueError):
                validate_result(changed, point)


if __name__ == "__main__":
    unittest.main()
