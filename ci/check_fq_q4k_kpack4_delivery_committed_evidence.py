#!/usr/bin/env python3
"""Validate SHA-bound host evidence without executing nvcc on a PPU box."""

from __future__ import annotations

import argparse
import hashlib
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "dev/fold_derivation/q4_kpack4_delivery_host.expected.txt"
L229 = ROOT / "dev/fold_derivation/l229_q4_kpack4_production_type.cu"
L231 = ROOT / "dev/fold_derivation/l231_q4_kpack4_production_fragment.cu"
L231_RUNNER = ROOT / "dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh"
L232 = ROOT / "dev/fold_derivation/l232_q4_kpack4_fused_metadata_read.cu"
BOX_RUNNER = ROOT / "tools/run_fq_q4k_kpack4_delivery_ab_box.sh"
SCHEMA = "quactlize.fq-q4k-kpack4-delivery-host-evidence.v2"


class CheckError(ValueError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw or "=" not in raw:
            raise CheckError(f"malformed evidence line: {raw!r}")
        key, value = raw.split("=", 1)
        if key in rows:
            raise CheckError(f"duplicate evidence key: {key}")
        rows[key] = value
    return rows


def validate(evidence: str, l229: str, l231: str, l231_runner: str, l232: str,
             box_runner: str) -> None:
    rows = parse(evidence)
    expected_keys = {
        "schema", "l229_source_sha256", "l231_source_sha256",
        "l231_runner_sha256", "l232_source_sha256", "l229", "l231_source",
        "l231_d32", "l231_d16", "l231_rotate", "l231_legacy", "l232",
    }
    if set(rows) != expected_keys or rows["schema"] != SCHEMA:
        raise CheckError(f"evidence schema/denominator differs: {sorted(rows)}")
    for key, source in (("l229_source_sha256", l229),
                        ("l231_source_sha256", l231),
                        ("l231_runner_sha256", l231_runner),
                        ("l232_source_sha256", l232)):
        if rows[key] != sha256_text(source):
            raise CheckError(f"{key} differs from the committed source")
    exact = {
        "l229": "L229 Q4_K KPACK4 production-type PASS layout=0x00000001 "
                "mapping=0x51344b5034540001 physical=NxK/4 "
                "transport=N16xK64 delivery=auto64+32+16 "
                "tactic=8x64x256_w8x16_s2 providers=standard-aiu+packed-row",
        "l231_source": "[l231-source] PASS compute-N-stride=BOUND "
                       "legacy-negative=BOUND separate-layout-converter=BOUND",
        "l231_d32": "[l231-delivery] PASS D=32 geometries=12 candidate=IDENTITY",
        "l231_d16": "[l231-delivery] PASS D=16 geometries=12 candidate=IDENTITY",
        "l231_rotate": "[l231-red] PASS plant=rotated-destination result=RED",
        "l231_legacy": "[l231-red] PASS plant=legacy-loader-stride result=RED",
        "l232": "L232 Q4_K KPACK4 fused-metadata-read layout_bad=0 bits_bad=0 "
                "providers=AP0+AP1 delivery=D32 values=1024-per-provider",
    }
    for key, value in exact.items():
        if rows[key] != value:
            raise CheckError(f"committed host verdict differs: {key}")
    forbidden = (
        'python3 -B "$root/ci/local_gates.py"',
        'bash "$root/dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh"',
    )
    if any(token in box_runner for token in forbidden):
        raise CheckError("box runner executes an NVIDIA-nvcc/stub host oracle")
    required = (
        'git -C "$root" show "$sha:dev/fold_derivation/q4_kpack4_delivery_host.expected.txt"',
        "check_fq_q4k_kpack4_delivery_committed_evidence.py",
        "--committed-only --evidence",
        "fresh_box_execution=0",
    )
    if any(box_runner.count(token) != 1 for token in required):
        raise CheckError("box runner lost the exact committed-evidence boundary")


def self_test(evidence: str, sources: tuple[str, str, str, str], box: str) -> None:
    validate(evidence, *sources, box)
    plants = (
        (evidence.replace("l231_d32=", "l231_d33=", 1), sources, box),
        (evidence.replace("candidate=IDENTITY", "candidate=NONIDENTITY", 1),
         sources, box),
        (evidence, (sources[0] + "\n// drift", sources[1], sources[2], sources[3]), box),
        (evidence, (sources[0], sources[1], sources[2], sources[3] + "\n// drift"), box),
        (evidence, sources, box.replace(
            "--committed-only --evidence", "--evidence", 1)),
        (evidence, sources, box +
         '\npython3 -B "$root/ci/local_gates.py" -k l229\n'),
    )
    for broken, broken_sources, broken_box in plants:
        try:
            validate(broken, *broken_sources, broken_box)
        except CheckError:
            pass
        else:
            raise CheckError("committed-evidence negative stayed green")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed-only", action="store_true")
    parser.add_argument("--evidence", type=pathlib.Path, default=EXPECTED)
    args = parser.parse_args()
    try:
        evidence = args.evidence.read_text()
        sources = (L229.read_text(), L231.read_text(), L231_RUNNER.read_text(),
                   L232.read_text())
        box = BOX_RUNNER.read_text()
        validate(evidence, *sources, box)
        self_test(evidence, sources, box)
    except (OSError, CheckError, AssertionError) as exc:
        print(f"[fq-kpack4-delivery-committed] FAIL: {exc}")
        return 2
    print("[fq-kpack4-delivery-committed] PASS exact L229 type + "
          "L231 D32/D16/2-RED + L232 fused-read map evidence; "
          "fresh_box_execution=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
