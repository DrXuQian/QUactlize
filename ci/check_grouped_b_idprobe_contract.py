#!/usr/bin/env python3
"""Permanent contract for L130's exhaustive G5 B-side ID probe.

This checker has two independent jobs:

* compile and run L130, then parse all 256 expert rows and the two anchors,
  exact historical red control, scope statement, and compiled type negative;
* mutate the source in memory and prove that deleting any load-bearing arm is
  rejected by the structural audit.

The source plants are intentionally not compiled.  The real, unmodified
oracle is compiled and executed exactly once below; the plants prove that the
checker cannot silently stop asking one of the questions that made that run
meaningful.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "dev/fold_derivation/l130_grouped_b_idprobe.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l130_grouped_b_idprobe.sh"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def audit(oracle: str, runner: str) -> list[str]:
    """Reject a source that no longer contains every load-bearing arm."""
    bad: list[str] = []
    flat = compact(oracle)
    runner_flat = compact(runner)

    try:
        sweep = compact(section(
            oracle,
            "std::vector<std::int8_t> all",
            "// Required red control",
        ))
        red = compact(section(
            oracle,
            "// Required red control",
            "// Two small controls",
        ))
        controls = compact(section(
            oracle,
            "// Two small controls",
            "bool const positive",
        ))
    except ValueError as exc:
        return [str(exc)]

    type_tokens = (
        "usingShipping=m8n16_g5_contract::M8",
        "std::is_same_v<typenameSelectedPolicy::CollectiveOp,Mainloop>",
        "Descriptor::quant_mode==ppu_mixed_policy::QuantMode::FinegrainedScaleZero",
        "!Descriptor::interleaved",
        "Descriptor::BProviderType,ppu_mixed_policy::OrdinaryBProvider",
        "Mainloop::DispatchPolicy::kContinous,cute::_1",
        "Mainloop::GmemTiledCopyB,typenameExpectedOperand::GmemTiledCopy",
        "Mainloop::SmemCopyAtomB,typenameExpectedOperand::SmemCopyAtom",
        "int(cute::size(typenameMainloop::TiledMma{}))==32",
    )
    for token in type_tokens:
        if token not in flat:
            bad.append(f"L130 lost exact shipping-type token {token!r}")
    for bypass in (
        "true||std::is_same_v<typenameSelectedPolicy::CollectiveOp,Mainloop>",
        "false&&std::is_same_v<typenameSelectedPolicy::CollectiveOp,Mainloop>",
    ):
        if bypass in flat:
            bad.append(f"L130 contains constant type bypass {bypass!r}")

    sweep_tokens = (
        "for(inte=0;e<spec::kExperts;++e){autoconstq=probe_codes(e)",
        "xplane::place_derived<4,8,32,64,8,32,1>(resident,q,spec::kN,spec::kK)",
        "xplane::recover_derived<4,8,32,64,8,32,1>(resident,recovered[e],spec::kN,spec::kK)",
        "total.roundtrip_code_diff+=recovered[e][i]!=q[i]",
        "exact_output(recovered[e],n)!=e",
        "e* kArtifactBytes".replace(" ", ""),
        "columns=%s",
    )
    for token in sweep_tokens:
        if token not in sweep:
            bad.append(f"L130 lost exhaustive e0..255 sweep token {token!r}")

    anchor_tokens = (
        "constexprintkLegacyN=64",
        "anchor_legacy=legacy_pack(anchor_q,kLegacyN,spec::kK)",
        "total.legacy_byte_diff+=std::uint8_t(anchor_placed[i])!=std::uint8_t(anchor_legacy[i])",
        "corrupt_legacy_diff==1",
        "corrupt_roundtrip_diff==1",
    )
    for token in anchor_tokens[:3]:
        if token not in sweep:
            bad.append(f"L130 lost independent legacy anchor token {token!r}")
    for token in anchor_tokens[3:]:
        if token not in controls and token not in flat:
            bad.append(f"L130 lost planted anchor control {token!r}")

    red_tokens = (
        "for(inte=0;e<spec::kExperts;++e)",
        "intconstplanted=e<128?e:e-64",
        "red_mismatched_experts+=mismatch",
        "red_low_bad+=e<128&&mismatch",
        "red_high_inexact+=e>=128&&(!mismatch||!columns_same||got!=e-64)",
        "red_mismatched_experts==128&&red_low_bad==0&&red_high_inexact==0",
    )
    for token in red_tokens:
        if token not in red and token not in flat:
            bad.append(f"L130 lost exact 128-expert red token {token!r}")

    for token in (
        "[l130:scope]B-low-plane-only",
        "interleaveddBanddB2areNOTSELECTEDbythiskerneltype",
        "zero/scaleaddressingiscoveredbyL125,notinferredhere",
    ):
        if token not in flat:
            bad.append(f"L130 lost scope boundary {token!r}")

    runner_tokens = (
        "-DL130_TYPE_ONLY=1",
        "-DL130_SELECTED_WN=16",
        "L130selectedpolicyisnottheshippingG5Btype",
        "wronglegalWN16instancerejectedbytheshipping-typeassertionPASS",
    )
    for token in runner_tokens:
        if token not in runner_flat:
            bad.append(f"L130 runner lost compiled type negative {token!r}")
    return bad


ROW = re.compile(
    r"^\[l130:e\]\s+e=\s*(\d+)\s+scheduler=\s*(\d+)\s+"
    r"dB-code-base=\s*(\d+)\s+resident-bytes=\[\s*(\d+),\s*(\d+)\]\s+"
    r"output=\s*(\d+)\s+columns=32/32$",
    re.MULTILINE,
)


def validate_run(output: str) -> list[str]:
    bad: list[str] = []
    rows = [tuple(map(int, m.groups())) for m in ROW.finditer(output)]
    if len(rows) != 256:
        bad.append(f"runtime printed {len(rows)} expert rows, expected 256")
    else:
        for e, scheduler, code_base, byte_lo, byte_hi, result in rows:
            expected = (e, e, e * 8192, e * 4096, (e + 1) * 4096 - 1, e)
            got = (e, scheduler, code_base, byte_lo, byte_hi, result)
            if got != expected:
                bad.append(f"expert row {e} is {got}, expected {expected}")
                break
        if [row[0] for row in rows] != list(range(256)):
            bad.append("runtime expert rows are not exactly e=0..255 in order")

    required = (
        "[l130:type] exact G5 B: FinegrainedScaleZero gs32 "
        "tile=8x32x64 warp=8x32x64 stages=3 int4 CTA32; "
        "ordinary dB-backed kContinuous=1; dB2/interleaved=NOT-SELECTED PASS",
        "[l130] e0..255 legacy-byte-diff=0 place/recover-code-diff=0 "
        "map-holes=0 map-dups=0 address-bad=0 value-bad=0 output-bad=0 "
        "source=[0,1048575] -> PASS",
        "[l130:anchor] legacy-five-step byte identity at legal N64 companion + "
        "G5-N32 place/recover identity; planted-byte controls legacy=1 "
        "roundtrip=1 expected=1/1 -> PASS",
        "[l130:red] e>=128->e-64 mismatched-experts=128 low-bad=0 "
        "high-inexact=0 expected=128/0/0 -> EXPECTED-RED",
        "[l130:scope] B-low-plane-only; G5 selects ordinary dB. "
        "interleaved dB and dB2 are NOT SELECTED by this kernel type; "
        "zero/scale addressing is covered by L125, not inferred here. result=PASS",
        "L130 compiled negative: wrong legal WN16 instance rejected by the "
        "shipping-type assertion PASS",
    )
    for line in required:
        if line not in output:
            bad.append(f"runtime lost verdict {line!r}")
    return bad


def run_oracle() -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="quactlize-l130-contract-") as tmp:
        env = dict(os.environ, QUACTLIZE_L130_OUT=tmp)
        try:
            proc = subprocess.run(
                ["bash", str(RUNNER)], cwd=ROOT, env=env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=300, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, f"cannot run L130: {exc}"
    return proc.returncode, proc.stdout


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in (ORACLE, RUNNER) if not p.is_file()]
    if missing:
        print("[l130-contract] FAIL: missing " + ", ".join(missing))
        return 1

    oracle = ORACLE.read_text()
    runner = RUNNER.read_text()
    bad = audit(oracle, runner)
    if bad:
        print("[l130-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        ("for (int e = 0; e < spec::kExperts; ++e) {\n    auto const q = probe_codes(e);",
         "for (int e = 0; e < spec::kExperts - 1; ++e) {\n    auto const q = probe_codes(e);",
         "full e0..255 sweep"),
        ("auto const anchor_legacy = legacy_pack(anchor_q, kLegacyN, spec::kK);",
         "auto const anchor_legacy = anchor_placed;",
         "independent legacy anchor"),
        ("total.roundtrip_code_diff += recovered[e][i] != q[i];",
         "total.roundtrip_code_diff += 0;",
         "place/recover anchor"),
        ("int const planted = e < 128 ? e : e - 64;",
         "int const planted = e;",
         "exact historical e-64 red"),
        ("[l130:scope] B-low-plane-only; G5 selects ordinary dB. ",
         "[l130:scope] B-plane-unspecified; G5 selects ordinary dB. ",
         "B-only scope boundary"),
        ("std::is_same_v<typename SelectedPolicy::CollectiveOp, Mainloop>",
         "true || std::is_same_v<typename SelectedPolicy::CollectiveOp, Mainloop>",
         "exact shipping type"),
    )
    for old, new, label in plants:
        if oracle.count(old) != 1:
            print(f"[l130-contract] FAIL: cannot plant {label}; matches={oracle.count(old)}")
            return 1
        planted = oracle.replace(old, new, 1)
        if not audit(planted, runner):
            print(f"[l130-contract] FAIL: checker accepted planted {label}")
            return 1

    rc, output = run_oracle()
    if rc != 0:
        tail = "\n".join(output.splitlines()[-20:])
        print(f"[l130-contract] FAIL: L130 rc={rc}\n{tail}")
        return 1
    bad = validate_run(output)
    if bad:
        print("[l130-contract] FAIL: " + "; ".join(bad))
        return 1
    print("[l130-contract] PASS: real L130 proved 256 experts, legacy+roundtrip "
          f"anchors, exact 128 red, and compiled type negative; {len(plants)} source plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
