#!/usr/bin/env python3
"""Validate archived host-only K-pack4 delivery evidence.

This file deliberately does not bind old host-proof hashes to the current
source tree.  It records what the historical proof covered, while current
device admission remains a fresh PPU-box responsibility.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "dev/fold_derivation/q4_kpack4_delivery_host.expected.txt"
BOX_RUNNER = ROOT / "tools/run_fq_q4k_kpack4_delivery_ab_box.sh"
SCHEMA = "quactlize.fq-q4k-kpack4-delivery-host-evidence.v3"
HISTORICAL_COMMIT = "d5998079af9d4c1afa7e6e79e62cf528676acc52"
HISTORICAL_SOURCES = {
    "historical_l229_source_sha256":
        "dev/fold_derivation/l229_q4_kpack4_production_type.cu",
    "historical_l231_source_sha256":
        "dev/fold_derivation/l231_q4_kpack4_production_fragment.cu",
    "historical_l231_runner_sha256":
        "dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh",
    "historical_l232_source_sha256":
        "dev/fold_derivation/l232_q4_kpack4_fused_metadata_store.cu",
}


class CheckError(ValueError):
    pass


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


def historical_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(ROOT), "show", f"{commit}:{path}"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CheckError(f"historical source is unavailable: {commit}:{path}: {detail}")
    return hashlib.sha256(result.stdout).hexdigest()


def validate(evidence: str, box_runner: str) -> None:
    rows = parse(evidence)
    expected_keys = {
        "schema", "evidence_kind", "source_commit", "current_admission",
        "historical_l229_source_sha256", "historical_l231_source_sha256",
        "historical_l231_runner_sha256", "historical_l232_source_sha256", "l229", "l231_source",
        "l231_d32", "l231_d16", "l231_rotate", "l231_legacy", "l232",
    }
    if set(rows) != expected_keys or rows["schema"] != SCHEMA:
        raise CheckError(f"evidence schema/denominator differs: {sorted(rows)}")
    if rows["evidence_kind"] != "historical-host-only":
        raise CheckError("evidence must be explicitly historical host-only")
    if rows["source_commit"] != HISTORICAL_COMMIT:
        raise CheckError("historical evidence commit differs")
    if rows["current_admission"] != "fresh-ppu-box-required":
        raise CheckError("historical evidence cannot admit current device code")
    for key, path in HISTORICAL_SOURCES.items():
        value = rows[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise CheckError(f"archived source digest is malformed: {key}")
        if value != historical_sha256(rows["source_commit"], path):
            raise CheckError(f"archived source digest differs: {key}")
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
        "l232": "L232 Q4_K KPACK4 fused-metadata-store layout_bad=0 bits_bad=0 "
                "providers=AP0+AP1 delivery=auto64 values=1024-per-provider",
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


def self_test(evidence: str, box: str) -> None:
    validate(evidence, box)
    plants = (
        (evidence.replace("l231_d32=", "l231_d33=", 1), box),
        (evidence.replace("candidate=IDENTITY", "candidate=NONIDENTITY", 1),
         box),
        (evidence.replace("evidence_kind=historical-host-only",
                          "evidence_kind=current-device", 1), box),
        (evidence.replace("current_admission=fresh-ppu-box-required",
                          "current_admission=admitted", 1), box),
        (evidence.replace(
            "historical_l229_source_sha256=f1fc2a3492578a83a8854a93e6c2bd06c453440ee584396858951f99ef9e7e27",
            "historical_l229_source_sha256=" + "0" * 64, 1), box),
        (evidence, box.replace(
            "--committed-only --evidence", "--evidence", 1)),
        (evidence, box +
         '\npython3 -B "$root/ci/local_gates.py" -k l229\n'),
    )
    for broken, broken_box in plants:
        try:
            validate(broken, broken_box)
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
        box = BOX_RUNNER.read_text()
        validate(evidence, box)
        self_test(evidence, box)
    except (OSError, CheckError, AssertionError) as exc:
        print(f"[fq-kpack4-delivery-committed] FAIL: {exc}")
        return 2
    print("[fq-kpack4-delivery-committed] PASS historical-host-only "
          f"source_commit={HISTORICAL_COMMIT} current_admission=fresh-ppu-box-required "
          "fresh_box_execution=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
