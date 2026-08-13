#!/usr/bin/env python3
"""Host-only contract for runner-produced adjudication bundles."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
DENSE = ROOT / "tools/run_dense_marlin_wk4_box.sh"
GEMV = ROOT / "tools/run_gemv_sweep_box.sh"
WRITER = ROOT / "tools/write_box_run_provenance.py"
EXPORTER = ROOT / "tools/export_gemv_base_census.py"
POLICY = ROOT / "dev/fold_derivation/BOX_RUN_PREREGISTRATION.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_contract(dense: str, gemv: str) -> None:
    identities = (
        "QUACTLIZE_BOX_DEVICE_MODEL", "QUACTLIZE_BOX_PCI_IDENTITY",
        "QUACTLIZE_BOX_DRIVER_VERSION", "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY",
    )
    for name, text in (("dense", dense), ("gemv", gemv)):
        for identity in identities:
            require(identity in text, f"{name} runner does not require {identity}")
        for token in ("status --porcelain=v1 --untracked-files=all",
                      "submodule status --recursive", "--root-status clean",
                      "--binary-sha256", "--commands-file", "runner.log",
                      "policy-sample-count"):
            require(token in text, f"{name} runner lost {token!r}")
    for token in ("check_dense_marlin_wk4_target.py",
                  "run_l143_wk4_production_delivery.sh",
                  "L143 WK1 shipping map-diff=0 byte-diff=0 result=BIT-IDENTICAL",
                  "thirteen structural plants rejected"):
        require(token in dense, f"dense runner lost exact WK1 admission token {token!r}")
    for token in ("export_gemv_base_census.py", "base-census.json",
                  "BASE_COMMANDS", '"role":"device-build"'):
        require(token in gemv, f"GEMV runner lost bundle authority token {token!r}")


def run(*argv: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if ok and proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {argv}\n{proc.stdout}")
    if not ok and proc.returncode == 0:
        raise RuntimeError(f"negative unexpectedly passed: {argv}")
    return proc


def dynamic_contract() -> None:
    run("bash", "-n", str(DENSE), str(GEMV))
    dense_samples = run(sys.executable, str(WRITER), "policy-sample-count",
                        "--policy", str(POLICY), "--kind", "dense").stdout.strip()
    gemv_samples = run(sys.executable, str(WRITER), "policy-sample-count",
                       "--policy", str(POLICY), "--kind", "gemv").stdout.strip()
    require(dense_samples == gemv_samples == "20", "policy sample authority is not 20/20")

    with tempfile.TemporaryDirectory(prefix="qz-box-runner-contract-") as td:
        out = Path(td)
        commands = out / "commands.jsonl"
        run(sys.executable, str(WRITER), "record", "--path", str(commands),
            "--role", "fixture", "--exit-status", "0", "--", "fixture-bin", "--x=1")
        submodules = out / "submodules.txt"
        submodules.write_text(" " + "0" * 40 + " third_party/actlize\n")
        provenance = out / "provenance.json"
        base = (sys.executable, str(WRITER), "write", "--output", str(provenance),
                "--root-sha", "1" * 40, "--root-status", "clean",
                "--submodule-status-file", str(submodules), "--actlize-sha", "0" * 40,
                "--binary-sha256", "2" * 64, "--device-model", "fixture-device",
                "--pci-identity", "0000:01:00.0", "--driver-version", "fixture-driver",
                "--sdk-compiler-identity", "fixture-sdk", "--commands-file", str(commands),
                "--runner-exit-status", "0", "--protocol-sample-count", "20", "--",
                "tools/fixture-runner.sh")
        run(*base)
        value = json.loads(provenance.read_text())
        require(value["root_status"] == "clean", "root_status not preserved")
        require(value["argv"] == ["tools/fixture-runner.sh"], "runner argv not exact")
        require(value["commands"][0]["argv"] == ["fixture-bin", "--x=1"],
                "child argv not exact")
        bad = list(base)
        bad[bad.index("fixture-device")] = "UNKNOWN"
        run(*bad, ok=False)

        census = out / "base-census.json"
        authority = out / "authority.log"
        run(sys.executable, str(EXPORTER), "--output", str(census),
            "--authority-log", str(authority))
        value = json.loads(census.read_text())
        require(set(value) == {"schema", "total", "legal", "pruned", "prune_reasons"},
                "base census is not the adjudicator's exact five-key schema")
        require((value["total"], value["legal"], value["pruned"]) ==
                (27360, 10260, 17100), "base census differs from C++ authority")
        require(sum(value["prune_reasons"].values()) == value["pruned"],
                "base census histogram does not close")


def negative_controls(dense: str, gemv: str) -> None:
    plants = (
        (dense.replace("--root-status clean", "--root-status dirty"), gemv,
         "dense root-status plant"),
        (dense.replace("run_l143_wk4_production_delivery.sh", "missing-l143.sh", 1), gemv,
         "dense WK1 oracle plant"),
        (dense, gemv.replace("export_gemv_base_census.py", "missing-exporter.py", 1),
         "GEMV census authority plant"),
        (dense, gemv.replace("policy-sample-count", "literal-sample-count", 1),
         "GEMV policy sample plant"),
    )
    for bad_dense, bad_gemv, label in plants:
        try:
            source_contract(bad_dense, bad_gemv)
        except RuntimeError:
            continue
        raise RuntimeError(f"{label} was not rejected")


def main() -> int:
    try:
        dense = DENSE.read_text()
        gemv = GEMV.read_text()
        source_contract(dense, gemv)
        dynamic_contract()
        negative_controls(dense, gemv)
    except RuntimeError as exc:
        print(f"[box-runner-bundle] FAIL: {exc}")
        return 1
    print("[box-runner-bundle] PASS: runner-owned provenance/WK1/base-census bundles; four plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
