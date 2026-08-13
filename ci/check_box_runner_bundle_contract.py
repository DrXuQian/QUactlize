#!/usr/bin/env python3
"""Host-only contract for runner-produced adjudication bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
DENSE = ROOT / "tools/run_dense_marlin_wk4_box.sh"
GEMV = ROOT / "tools/run_gemv_sweep_box.sh"
WRITER = ROOT / "tools/write_box_run_provenance.py"
IDENTITY_PROBE = ROOT / "tools/probe_box_identity.py"
IDENTITY_SCHEMA = ROOT / "tools/box_identity_schema.py"
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
            require(identity in text, f"{name} runner lost {identity} provenance plumbing")
        require("require_identity" not in text,
                f"{name} runner still makes operator identity mandatory")
        require("resolve_box_identity" in text and
                "box-identity-probe" in text and "source" in text,
                f"{name} runner does not measure and source-label identity")
        for token in ("status --porcelain=v1 --untracked-files=all",
                      "submodule status --recursive", "--root-status clean",
                      "--binary-sha256", "--commands-file", "runner.log",
                      "policy-sample-count", "probe_box_identity.py",
                      "identity-probe.json", "--identity-probe-file"):
            require(token in text, f"{name} runner lost {token!r}")
    for token in (
            "check_dense_marlin_wk4_target.py",
            "l143_standalone_marlin.expected.txt",
            "standalone-static-target",
            "committed-standalone-evidence",
            "local-evidence=committed-standalone-oracle",
            "fresh-box-execution=0",
            "[L167] PASS: independent classic/direct and Awesome-CuTe/permutation anchors agree",
            "[l168:runner] positive=PASS negative_controls=3/3_RED result=PASS",
            "[l169] PASS: generated-unit shape instantiates standalone Marlin collective/scheduler/kernel",
            "[l170:runner] positive=PASS negative_controls=7/7_RED result=PASS",
            "ten structural plants rejected",
            "[classic-156] PASS: exact one-launch shape",
            "[l143] PASS: standalone Marlin format + cadence + generated type + scheduler lifecycle",
            "placement=classic-marlin-u32 scale=classic-gs128-permuted",
            "PASS: standalone classic Marlin built on classic u32 bytes"):
        require(token in dense,
                f"dense runner lost standalone admission token {token!r}")
    for token in ("export_gemv_base_census.py", "base-census.json",
                  "bind-build-pair", "BUILD_ATTEMPT_LOG",
                  "promote_canonical_attempt",
                  "verify_source_identity", "CXX=g++",
                  "--role device-build"):
        require(token in gemv, f"GEMV runner lost bundle authority token {token!r}")


def run(*argv: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if ok and proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {argv}\n{proc.stdout}")
    if not ok and proc.returncode == 0:
        raise RuntimeError(f"negative unexpectedly passed: {argv}")
    return proc


def _write(path: Path, value: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _fixture_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "tools").mkdir()
    shutil.copy2(GEMV, repo / "tools/run_gemv_sweep_box.sh")
    shutil.copy2(WRITER, repo / "tools/write_box_run_provenance.py")
    shutil.copy2(IDENTITY_PROBE, repo / "tools/probe_box_identity.py")
    shutil.copy2(IDENTITY_SCHEMA, repo / "tools/box_identity_schema.py")
    # Exercise the runner against the production helper's exact resolve/get
    # boundary without pretending this host has a PPU SDK.  The fixture helper
    # is deliberately installed at the production path: a parallel test-only
    # probe path would prove neither the runner wiring nor its fail-closed
    # behavior before the first build.
    _write(repo / "tools/probe_box_identity.py", r'''#!/usr/bin/env python3
import argparse, json, os, pathlib, sys

FIELDS = ("device_model", "pci_identity", "driver_version", "sdk_compiler_identity")
ENVS = {
    "device_model": "QUACTLIZE_BOX_DEVICE_MODEL",
    "pci_identity": "QUACTLIZE_BOX_PCI_IDENTITY",
    "driver_version": "QUACTLIZE_BOX_DRIVER_VERSION",
    "sdk_compiler_identity": "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY",
}

def fail(message):
    print("box identity probe: " + message, file=sys.stderr)
    raise SystemExit(2)

def resolve(output):
    mode = os.environ.get("FIXTURE_IDENTITY_MODE", "one")
    if mode == "two":
        fail("hggc runtime reports 2 visible devices; candidates: "
             "ordinal=0 name='fixture-ppu-a' pci=0000:01:00.0 cu=72; "
             "ordinal=1 name='fixture-ppu-b' pci=0000:02:00.0 cu=72. "
             "Set CUDA_VISIBLE_DEVICES to exactly one ordinal and rerun; "
             "operator identity strings do not disambiguate devices.")
    if mode not in {"one", "empty"}:
        fail("unknown fixture mode " + repr(mode))
    if mode == "one":
        values = {
            "device_model": "fixture-ppu",
            "pci_identity": "0000:01:00.0",
            "driver_version": "fixture-driver",
            "sdk_compiler_identity": "fixture-sdk",
        }
        identity = {field: {"value": values[field], "source": "measured"}
                    for field in FIELDS}
        device = {"status": "measured", "method": "fixture-hggc-runtime",
                  "reason": "", "runtime_driver_version": values["driver_version"],
                  "pci_measurement": "fixture", "driver_measurement": "fixture",
                  "device_count": 1, "selected_ordinal": 0,
                  "candidates": [{"ordinal": 0, "name": values["device_model"],
                  "compute_capability": "10.0", "compute_units": 72,
                  "pci_identity": values["pci_identity"]}], "property_errors": []}
        sdk = {"status": "measured", "reason": "",
               "sdk_root_authority": "fixture", "sdk_root": "/fixture-sdk",
               "compiler_path": "/fixture-sdk/bin/hgcc",
               "version_first_line": "fixture-sdk",
               "identity_value": values["sdk_compiler_identity"]}
    else:
        values = {}
        for field in FIELDS:
            value = os.environ.get(ENVS[field], "").strip()
            if not value:
                fail(f"{field} could not be measured and {ENVS[field]} is not a "
                     "usable operator fallback: value is empty")
            values[field] = value
        identity = {field: {"value": values[field], "source": "operator"}
                    for field in FIELDS}
        device = {"status": "unavailable", "method": "", "reason": "fixture-empty",
                  "runtime_driver_version": "", "pci_measurement": "unavailable",
                  "driver_measurement": "unavailable", "device_count": None,
                  "selected_ordinal": None, "candidates": [], "property_errors": []}
        sdk = {"status": "unavailable", "reason": "fixture-empty",
               "sdk_root_authority": "", "sdk_root": "", "compiler_path": "",
               "version_first_line": "", "identity_value": ""}
    document = {"schema": "quactlize-box-identity-probe-v1",
                "identity": identity, "device_probe": device,
                "sdk_compiler_probe": sdk}
    pathlib.Path(output).write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")

def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("resolve"); p.add_argument("--output", required=True)
    p = sub.add_parser("get"); p.add_argument("--file", required=True)
    p.add_argument("--field", choices=FIELDS, required=True)
    p.add_argument("--part", choices=("value", "source"), required=True)
    args = parser.parse_args()
    if args.cmd == "resolve": resolve(args.output); return
    value = json.loads(pathlib.Path(args.file).read_text())
    print(value["identity"][args.field][args.part])

if __name__ == "__main__": main()
''', executable=True)
    _write(repo / "dev/fold_derivation/BOX_RUN_PREREGISTRATION.md", """
<!-- BOX_RUN_POLICY_V1_BEGIN -->
{"gemv":{"sample_count":20}}
<!-- BOX_RUN_POLICY_V1_END -->
""".lstrip())
    _write(repo / "tools/export_gemv_base_census.py", r'''#!/usr/bin/env python3
import argparse, json, os, pathlib
p=argparse.ArgumentParser(); p.add_argument("--output"); p.add_argument("--authority-log")
a=p.parse_args()
if os.environ.get("CXX") != "g++": raise SystemExit("CXX authority differs")
pathlib.Path(a.output).write_text(json.dumps({"schema":"quactlize-gemv-base-census-v1","total":1,"legal":1,"pruned":0,"prune_reasons":{}})+"\n")
pathlib.Path(a.authority_log).write_text("AUTHORITY_COMPILER_ARGV,[\"g++\"]\nAUTHORITY_COMPILER_IDENTITY,fixture-g++\n")
''', executable=True)
    _write(repo / "build.sh", r'''#!/usr/bin/env bash
set -euo pipefail
[[ "${TARGET:-}" == test_gemv_perf ]]
mkdir -p "$PPU_BUILD_DIR"
cat >"$PPU_BUILD_DIR/test_gemv_perf" <<'BIN'
#!/usr/bin/env python3
import json, pathlib, sys
if sys.argv[1:2] != ["--manifest-json"]: raise SystemExit(2)
pathlib.Path(sys.argv[2]).write_text(json.dumps({"space_id":"fixture-space","partial_space":False,"jobs":[],"counts":{}})+"\n")
BIN
chmod +x "$PPU_BUILD_DIR/test_gemv_perf"
echo "fixture-build-sdk=${QUACTLIZE_BOX_SDK_COMPILER_IDENTITY:?}"
''', executable=True)
    _write(repo / "benchmarks/sweep_gemv_perf.py", r'''#!/usr/bin/env python3
import json, os, pathlib, sys, time
def opt(name): return sys.argv[sys.argv.index(name)+1]
action=sys.argv[1]
if action == "run" and "--dry-run" in sys.argv:
    pathlib.Path(opt("--dry-run-manifest")).write_text("{}\n")
    print("{}")
    raise SystemExit(0)
if action == "run":
    marker=os.environ.get("FIXTURE_HOLD_MARKER")
    release=os.environ.get("FIXTURE_HOLD_RELEASE")
    if marker:
        pathlib.Path(marker).write_text("held\n")
        while not pathlib.Path(release).exists(): time.sleep(0.01)
    raw=pathlib.Path(opt("--raw")); progress=pathlib.Path(opt("--progress"))
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("a") as f: f.write(json.dumps({"fixture":1})+"\n")
    with progress.open("a") as f: f.write(json.dumps({"fixture":1})+"\n")
    if os.environ.get("FIXTURE_FAIL_RUN") == "1": raise SystemExit(7)
    print("fixture-run-pass")
    raise SystemExit(0)
if action == "analyse":
    complete = os.environ.get("FIXTURE_INCOMPLETE_ANALYSE") != "1"
    pathlib.Path(opt("--output")).write_text(json.dumps({"groups":[],"complete":complete})+"\n")
    raise SystemExit(0)
raise SystemExit(2)
''', executable=True)
    _write(repo / "fixture.txt", "clean\n")

    child = root / "actlize-origin"
    child.mkdir()
    run("git", "init", "-q", str(child))
    run("git", "-C", str(child), "config", "user.email", "fixture@example.invalid")
    run("git", "-C", str(child), "config", "user.name", "fixture")
    _write(child / "README", "actlize\n")
    run("git", "-C", str(child), "add", "README")
    run("git", "-C", str(child), "commit", "-qm", "fixture actlize")

    run("git", "init", "-q", str(repo))
    run("git", "-C", str(repo), "config", "user.email", "fixture@example.invalid")
    run("git", "-C", str(repo), "config", "user.name", "fixture")
    run("git", "-c", "protocol.file.allow=always", "-C", str(repo), "submodule", "add",
        "-q", str(child), "third_party/actlize")
    run("git", "-C", str(repo), "add", ".")
    run("git", "-C", str(repo), "commit", "-qm", "fixture root")
    return repo


def _runner_env(out: Path, **updates: str) -> dict[str, str]:
    env = dict(os.environ)
    # A developer's shell must not accidentally turn an intended measured
    # fixture into an operator fallback, or make the empty-probe negative green.
    for name in ("QUACTLIZE_BOX_DEVICE_MODEL", "QUACTLIZE_BOX_PCI_IDENTITY",
                 "QUACTLIZE_BOX_DRIVER_VERSION",
                 "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY"):
        env.pop(name, None)
    env.update({
        "GEMV_SWEEP_DIR": str(out), "GEMV_SWEEP_BUILD_TIMEOUT_SECONDS": "30",
        "GEMV_SWEEP_DEADLINE_SECONDS": "30", "GEMV_SWEEP_SHAPE_TIMEOUT_SECONDS": "10",
        "MOE_CORES": "1", "FIXTURE_IDENTITY_MODE": "one",
    })
    env.update(updates)
    return env


def _operator_identity_env(**updates: str) -> dict[str, str]:
    values = {
        "FIXTURE_IDENTITY_MODE": "empty",
        "QUACTLIZE_BOX_DEVICE_MODEL": "fixture-ppu",
        "QUACTLIZE_BOX_PCI_IDENTITY": "0000:01:00.0",
        "QUACTLIZE_BOX_DRIVER_VERSION": "fixture-driver",
        "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY": "fixture-sdk",
    }
    values.update(updates)
    return values


def _probe_digest(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_probe(path: Path, values: dict[str, str], source: str) -> None:
    fields = ("device_model", "pci_identity", "driver_version",
              "sdk_compiler_identity")
    require(set(values) == set(fields), "test probe values are incomplete")
    measured = source == "measured"
    document = {
        "schema": "quactlize-box-identity-probe-v1",
        "identity": {field: {"value": values[field], "source": source}
                     for field in fields},
        "device_probe": {
            "status": "measured" if measured else "unavailable",
            "method": "fixture", "reason": "" if measured else "fixture",
            "runtime_driver_version": values["driver_version"] if measured else "",
            "pci_measurement": "fixture" if measured else "unavailable",
            "driver_measurement": "fixture" if measured else "unavailable",
            "device_count": 1 if measured else None,
            "selected_ordinal": 0 if measured else None,
            "candidates": ([{"ordinal": 0, "name": values["device_model"],
                             "compute_capability": "10.0", "compute_units": 72,
                             "pci_identity": values["pci_identity"]}]
                           if measured else []),
            "property_errors": [],
        },
        "sdk_compiler_probe": {
            "status": "measured" if measured else "unavailable",
            "reason": "" if measured else "fixture",
            "sdk_root_authority": "fixture" if measured else "",
            "sdk_root": "/fixture/sdk" if measured else "",
            "compiler_path": "/fixture/sdk/bin/hgcc" if measured else "",
            "version_first_line": "fixture" if measured else "",
            "identity_value": values["sdk_compiler_identity"] if measured else "",
        },
    }
    _write(path, json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")


def _tree_digest(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def transactional_runner_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-gemv-runner-transaction-") as td:
        temp = Path(td)
        repo = _fixture_repo(temp)
        runner = repo / "tools/run_gemv_sweep_box.sh"

        # A successful device-count observation is authoritative.  Two visible
        # candidates are ambiguity, not an invitation to trust hand-entered
        # strings: even complete operator fallbacks must not select one.
        two_out = temp / "two-devices"
        two_updates = _operator_identity_env()
        two_updates["FIXTURE_IDENTITY_MODE"] = "two"
        two = subprocess.run([str(runner)], cwd=repo,
                             env=_runner_env(two_out, **two_updates), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(two.returncode != 0 and "reports 2 visible devices" in two.stdout and
                "ordinal=0" in two.stdout and "0000:01:00.0" in two.stdout and
                "ordinal=1" in two.stdout and "0000:02:00.0" in two.stdout and
                "operator identity strings do not disambiguate" in two.stdout,
                "two visible devices did not fail closed and enumerate both candidates")
        require(_tree_digest(two_out) == {},
                "ambiguous automatic identity wrote a build or bundle artifact")

        # Empty automatic results are not identities.  Without explicit
        # fallback they fail before build; with all four fallbacks they publish
        # non-empty values whose lower evidence grade remains visible.
        empty_out = temp / "empty-no-fallback"
        empty = subprocess.run(
            [str(runner)], cwd=repo,
            env=_runner_env(empty_out, FIXTURE_IDENTITY_MODE="empty"), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(empty.returncode != 0 and "could not be measured" in empty.stdout and
                "operator fallback" in empty.stdout,
                "empty automatic identity neither failed nor requested operator fallback")
        require(_tree_digest(empty_out) == {},
                "empty automatic identity wrote an empty field into a bundle")

        operator_out = temp / "operator-fallback"
        operator = subprocess.run(
            [str(runner)], cwd=repo,
            env=_runner_env(operator_out, **_operator_identity_env()), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(operator.returncode == 0,
                "complete operator fallback did not run:\n" + operator.stdout)
        operator_bases = list(operator_out.glob("*/*/*-samples20"))
        require(len(operator_bases) == 1,
                "operator fallback did not publish exactly one canonical BASE")
        operator_base = operator_bases[0]
        operator_identity = json.loads((operator_base / "run-identity.json").read_text())
        operator_provenance = json.loads((operator_base / "provenance.json").read_text())
        operator_probe = json.loads((operator_base / "identity-probe.json").read_text())
        operator_sources = {field: "operator" for field in (
            "device_model", "pci_identity", "driver_version", "sdk_compiler_identity")}
        require(operator_identity["identity_sources"] == operator_sources and
                operator_provenance["identity_sources"] == operator_sources,
                "operator fallback was mislabeled as measured evidence")
        require(all(operator_identity[field] and operator_provenance[field] and
                    operator_probe["identity"][field]["value"]
                    for field in operator_sources),
                "operator fallback published an empty identity field")
        operator_digest = _probe_digest(operator_base / "identity-probe.json")
        require(operator_identity["identity_probe_sha256"] == operator_digest and
                operator_provenance["identity_probe_sha256"] == operator_digest,
                "operator fallback did not bind the exact probe evidence digest")

        # One successful run establishes the exact canonical shape, including
        # the explicit g++ base-census authority and immutable GROUPS field.
        success_out = temp / "success"
        success = subprocess.run([str(runner)], cwd=repo, env=_runner_env(success_out),
                                 text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        require(success.returncode == 0, "fixture success runner failed:\n" + success.stdout)
        bases = list(success_out.glob("*/*/*-samples20"))
        require(len(bases) == 1, "success did not publish exactly one hash-qualified BASE")
        base = bases[0]
        provenance = json.loads((base / "provenance.json").read_text())
        identity = json.loads((base / "run-identity.json").read_text())
        measured_sources = {field: "measured" for field in (
            "device_model", "pci_identity", "driver_version", "sdk_compiler_identity")}
        require(identity["identity_sources"] == measured_sources and
                provenance["identity_sources"] == measured_sources,
                "single-device automatic identity was not labeled measured")
        probe_digest = _probe_digest(base / "identity-probe.json")
        require(identity["identity_probe_sha256"] == probe_digest and
                provenance["identity_probe_sha256"] == probe_digest,
                "canonical identity/provenance did not bind identity-probe.json")
        require(identity["groups"] == "all", "immutable identity did not bind exact GROUPS")
        require(provenance["groups"] == identity["groups"] and
                provenance["run_identity_sha256"] == identity["identity_sha256"],
                "canonical provenance did not publish its exact immutable GROUPS/digest")
        require(provenance["runner_exit_status"] == 0, "canonical provenance is not successful")
        require((base / "commands.jsonl").read_text().splitlines() ==
                [json.dumps(item, sort_keys=True, separators=(",", ":"))
                 for item in provenance["commands"]],
                "canonical command journal differs from provenance")
        census = [item for item in provenance["commands"]
                  if item["role"] == "base-tactic-census"]
        require(len(census) == 1 and census[0]["argv"][:2] == ["env", "CXX=g++"],
                "base census did not record literal CXX=g++ authority")
        require("AUTHORITY_COMPILER_IDENTITY,fixture-g++" in
                (base / "base-census-authority.log").read_text(),
                "base census authority log lost compiler identity")
        require((base / "build.log").read_text().strip() ==
                "fixture-build-sdk=fixture-sdk",
                "canonical build log is not the successful attempt's SDK-marked log")

        # A later failing invocation cannot touch an already successful
        # canonical BASE; the completion marker is checked before any BASE IO.
        before_success = _tree_digest(base)
        denied = subprocess.run([str(runner)], cwd=repo,
                                env=_runner_env(success_out, FIXTURE_FAIL_RUN="1"),
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        require(denied.returncode != 0 and _tree_digest(base) == before_success,
                "failed later attempt overwrote a successful canonical BASE")
        require(not list(success_out.rglob(".attempt.*")),
                "completed canonical rerun left an unowned attempt directory")

        # Create a resumable failed attempt, then prove both identity mismatch
        # and a held BASE lock reject before raw or any BASE journal grows.
        resume_out = temp / "resume"
        failed = subprocess.run([str(runner)], cwd=repo,
                                env=_runner_env(resume_out, FIXTURE_FAIL_RUN="1"),
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        require(failed.returncode == 7, "fixture did not produce resumable rc=7")
        resume_bases = list(resume_out.glob("*/*/*-samples20"))
        require(len(resume_bases) == 1, "failed attempt did not make one resumable BASE")
        resume_base = resume_bases[0]

        # A forced rebuild happens before BASE identity can be checked.  Its
        # SDK-marked build output must remain attempt-local when that identity
        # is rejected, and a later correct reuse must inherit the old BASE's
        # command/log pair rather than a run-global last writer.
        before_sdk_mismatch = _tree_digest(resume_base)
        sdk_mismatch = subprocess.run(
            [str(runner)], cwd=repo,
            env=_runner_env(
                resume_out, **_operator_identity_env(
                    QUACTLIZE_BOX_SDK_COMPILER_IDENTITY="other-sdk",
                    GEMV_SWEEP_REUSE_BUILD="0")),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(sdk_mismatch.returncode != 0 and
                "resume identity mismatch" in sdk_mismatch.stdout,
                "forced rebuild under mismatched SDK did not fail identity")
        require("fixture-build-sdk=other-sdk" in sdk_mismatch.stdout,
                "SDK mismatch plant did not actually execute its marked rebuild")
        require(_tree_digest(resume_base) == before_sdk_mismatch,
                "mismatched SDK rebuild polluted the old immutable BASE")
        require(not list(resume_out.rglob(".attempt.*")),
                "SDK identity mismatch left an unowned attempt directory")

        before_mismatch = _tree_digest(resume_base)
        mismatch = subprocess.run(
            [str(runner)], cwd=repo,
            env=_runner_env(resume_out, **_operator_identity_env(
                QUACTLIZE_BOX_DEVICE_MODEL="other-ppu")),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(mismatch.returncode != 0 and "resume identity mismatch" in mismatch.stdout,
                "device identity mismatch did not fail closed")
        require(_tree_digest(resume_base) == before_mismatch,
                "identity mismatch grew raw or a BASE journal")
        require(not list(resume_out.rglob(".attempt.*")),
                "device identity mismatch left an unowned attempt directory")

        lock_path = resume_base.parent / (
            "base-" + resume_base.name.removesuffix("-samples20") + "-samples20.lock")
        locker = subprocess.Popen(["flock", str(lock_path), "sleep", "30"],
                                  start_new_session=True)
        try:
            time.sleep(0.1)
            before_lock = _tree_digest(resume_base)
            locked = subprocess.run([str(runner)], cwd=repo, env=_runner_env(resume_out),
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            require(locked.returncode != 0 and "another sweep owns" in locked.stdout,
                    "held BASE lock did not reject the second runner")
            require(_tree_digest(resume_base) == before_lock,
                    "runner without BASE lock wrote raw or journal bytes")
            require(not list(resume_out.rglob(".attempt.*")),
                    "BASE-lock loser left an unowned attempt directory")
        finally:
            os.killpg(locker.pid, signal.SIGTERM)
            locker.wait(timeout=5)

        resumed = subprocess.run([str(runner)], cwd=repo,
                                 env=_runner_env(
                                     resume_out, GEMV_SWEEP_REUSE_BUILD="1"), text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(resumed.returncode == 0, "resumable second attempt failed:\n" + resumed.stdout)
        resumed_provenance = json.loads((resume_base / "provenance.json").read_text())
        measured = [item for item in resumed_provenance["commands"]
                    if item["role"] == "measured-sweep"]
        require(len(measured) == 2 and [item["exit_status"] for item in measured] == [7, 0],
                "canonical command journal did not retain both measured attempts")
        builds = [item for item in resumed_provenance["commands"]
                  if item["role"] == "device-build"]
        require(len(builds) == 1 and builds[0]["exit_status"] == 0,
                "reuse duplicated or lost the original attempt-owned device-build command")
        require(len((resume_base / "raw.jsonl").read_text().splitlines()) == 2,
                "resumed raw does not contain one attributable row per measured attempt")
        require((resume_base / "commands.jsonl").read_text().splitlines() ==
                [json.dumps(item, sort_keys=True, separators=(",", ":"))
                 for item in resumed_provenance["commands"]],
                "resumed cumulative commands differ from canonical provenance")
        canonical_build = (resume_base / "build.log").read_text()
        require("fixture-build-sdk=fixture-sdk" in canonical_build and
                "fixture-build-sdk=other-sdk" not in canonical_build,
                "correct identity resume published the rejected SDK's build log")

        # The build lock is tested separately with a genuinely concurrent
        # first process held inside the measured command.
        concurrent_out = temp / "concurrent"
        marker, release = temp / "held.marker", temp / "held.release"
        held_env = _runner_env(concurrent_out, FIXTURE_HOLD_MARKER=str(marker),
                               FIXTURE_HOLD_RELEASE=str(release))
        first = subprocess.Popen([str(runner)], cwd=repo, env=held_env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        deadline = time.time() + 10
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        require(marker.exists(), "first concurrent runner never reached held measurement")
        concurrent_bases = list(concurrent_out.glob("*/*/*-samples20"))
        require(len(concurrent_bases) == 1, "held runner did not establish one BASE")
        concurrent_base = concurrent_bases[0]
        before_second = _tree_digest(concurrent_base)
        second = subprocess.run([str(runner)], cwd=repo,
                                env=_runner_env(concurrent_out), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(second.returncode != 0 and "build/select transaction" in second.stdout,
                "concurrent runner did not lose the build lock")
        require(_tree_digest(concurrent_base) == before_second,
                "build-lock loser wrote the active BASE")
        require(not list(concurrent_out.rglob(".attempt.*")),
                "build-lock loser left an unowned attempt directory")
        release.write_text("release\n")
        first_output = first.communicate(timeout=20)[0]
        require(first.returncode == 0, "held first runner failed:\n" + first_output)

        # Final cleanliness is checked after measurements, not inferred from
        # the initial SHA.  Dirty the tracked root while the child is held.
        dirty_out = temp / "dirty"
        dirty_marker, dirty_release = temp / "dirty.marker", temp / "dirty.release"
        dirty = subprocess.Popen(
            [str(runner)], cwd=repo,
            env=_runner_env(dirty_out, FIXTURE_HOLD_MARKER=str(dirty_marker),
                            FIXTURE_HOLD_RELEASE=str(dirty_release)),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        deadline = time.time() + 10
        while not dirty_marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        require(dirty_marker.exists(), "dirty plant runner never reached measurement")
        tracked = repo / "fixture.txt"
        original = tracked.read_text()
        tracked.write_text("dirty\n")
        dirty_release.write_text("release\n")
        dirty_output = dirty.communicate(timeout=20)[0]
        tracked.write_text(original)
        require(dirty.returncode != 0 and "final source identity check" in dirty_output,
                "final root cleanliness plant did not fail")
        dirty_bases = list(dirty_out.glob("*/*/*-samples20"))
        require(len(dirty_bases) == 1 and not (dirty_bases[0] / "provenance.json").exists(),
                "dirty final tree published successful canonical provenance")

        incomplete_out = temp / "incomplete"
        incomplete = subprocess.run(
            [str(runner)], cwd=repo,
            env=_runner_env(incomplete_out, FIXTURE_INCOMPLETE_ANALYSE="1"),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        incomplete_bases = list(incomplete_out.glob("*/*/*-samples20"))
        require(incomplete.returncode != 0 and "result is not a complete object" in incomplete.stdout,
                "analyse rc=0 with complete=false did not fail closed")
        require(len(incomplete_bases) == 1 and
                not (incomplete_bases[0] / "provenance.json").exists(),
                "complete=false published canonical provenance")


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
        probe = out / "identity-probe.json"
        probe_values = {
            # Non-ASCII crosses helper -> writer -> immutable identity.  An
            # ASCII-only fixture cannot detect a split canonical-hash rule.
            "device_model": "fixture-设备",
            "pci_identity": "0000:01:00.0",
            "driver_version": "fixture-driver",
            "sdk_compiler_identity": "fixture-sdk",
        }
        _write_probe(probe, probe_values, "measured")
        identity = out / "run-identity.json"
        identity_args = (
            sys.executable, str(WRITER), "write-identity", "--output", str(identity),
            "--root-sha", "1" * 40, "--submodule-status-file", str(submodules),
            "--actlize-sha", "0" * 40, "--binary-sha256", "2" * 64,
            "--device-model", probe_values["device_model"],
            "--pci-identity", probe_values["pci_identity"],
            "--driver-version", probe_values["driver_version"],
            "--sdk-compiler-identity", probe_values["sdk_compiler_identity"],
            "--identity-probe-file", str(probe),
            "--protocol-sample-count", "20", "--groups", "all")
        run(*identity_args)
        provenance = out / "provenance.json"
        base = (sys.executable, str(WRITER), "write", "--output", str(provenance),
                "--root-sha", "1" * 40, "--root-status", "clean",
                "--submodule-status-file", str(submodules), "--actlize-sha", "0" * 40,
                "--binary-sha256", "2" * 64,
                "--device-model", probe_values["device_model"],
                "--pci-identity", probe_values["pci_identity"],
                "--driver-version", probe_values["driver_version"],
                "--sdk-compiler-identity", probe_values["sdk_compiler_identity"],
                "--identity-probe-file", str(probe), "--commands-file", str(commands),
                "--groups", "all", "--run-identity-file", str(identity),
                "--runner-exit-status", "0", "--protocol-sample-count", "20", "--",
                "tools/fixture-runner.sh")
        run(*base)
        value = json.loads(provenance.read_text())
        require(value["root_status"] == "clean", "root_status not preserved")
        require(value["argv"] == ["tools/fixture-runner.sh"], "runner argv not exact")
        require(value["commands"][0]["argv"] == ["fixture-bin", "--x=1"],
                "child argv not exact")
        require(value["identity_sources"] == {field: "measured" for field in probe_values}
                and value["identity_probe_sha256"] == _probe_digest(probe),
                "writer did not publish the per-field source and exact probe digest")
        bad = list(base)
        bad[bad.index(probe_values["device_model"])] = "UNKNOWN"
        run(*bad, ok=False)

        # Reuse is a paired authority, not "some build command" plus the last
        # build.log seen elsewhere.  Two distinct sibling logs make the
        # authority ambiguous and must leave the new attempt untouched.
        attempts = out / "attempts"
        pair = json.dumps({
            "argv": ["fixture-build"], "exit_status": 0,
            "role": "device-build",
        }, sort_keys=True, separators=(",", ":")) + "\n"
        _write(attempts / "a/commands.jsonl", pair)
        _write(attempts / "a/build.log", "fixture-build-sdk=sdk-a\n")
        _write(attempts / "b/commands.jsonl", pair)
        _write(attempts / "b/build.log", "fixture-build-sdk=sdk-b\n")
        inherited_commands = attempts / "current/commands.jsonl"
        inherited_log = attempts / "current/build.log"
        ambiguous = run(
            sys.executable, str(WRITER), "bind-build-pair",
            "--attempts-dir", str(attempts),
            "--commands-file", str(inherited_commands),
            "--build-log", str(inherited_log), ok=False)
        require("multiple distinct identity-matched" in ambiguous.stdout and
                not inherited_commands.exists() and not inherited_log.exists(),
                "ambiguous build-log/command pairs did not fail before inheritance")

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
        (dense.replace("l143_standalone_marlin.expected.txt",
                       "missing-standalone.txt", 1), gemv,
         "dense standalone oracle plant"),
        (dense.replace("committed-standalone-evidence",
                       "unregistered-standalone-evidence", 1), gemv,
         "dense standalone role plant"),
        (dense.replace("placement=classic-marlin-u32",
                       "placement=shipping-xplane", 1), gemv,
         "dense classic artifact plant"),
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
        transactional_runner_contract()
        negative_controls(dense, gemv)
    except RuntimeError as exc:
        print(f"[box-runner-bundle] FAIL: {exc}")
        return 1
    print("[box-runner-bundle] PASS: canonical transaction, identity-resume, "
          "build/BASE locks, final-clean and six source plants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
