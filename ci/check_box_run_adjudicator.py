#!/usr/bin/env python3
"""Synthetic controls for the preregistered box-run adjudicator."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal


ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = load_module("box_adjudicator", ROOT / "tools/adjudicate_box_runs.py")
S = load_module("box_sweep", ROOT / "benchmarks/sweep_gemv_perf.py")
TEMPLATE = (ROOT / "dev/fold_derivation/BOX_RUN_PREREGISTRATION.md").read_text()


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def check_three_sections(result: dict, noun: str) -> None:
    required = {"cell_results", "registered_verdict", "unregistered_observations"}
    check(required <= set(result), f"{noun}: missing one of the three publication sections")
    check(isinstance(result["cell_results"], list), f"{noun}: cell section is not an array")
    check(isinstance(result["unregistered_observations"], list),
          f"{noun}: unregistered-observation section is not an array")


def replace_region(text: str, begin: str, end: str, body: str) -> str:
    _, start, stop = A._region(text, begin, end)
    return text[:start] + begin + "\n" + body + "\n" + end + text[stop:]


def write_policy(path: pathlib.Path, value: dict) -> A.LoadedPolicy:
    policy = copy.deepcopy(value)
    policy["prose_sha256"] = "0" * 64
    text = replace_region(TEMPLATE, A.POLICY_BEGIN, A.POLICY_END,
                          json.dumps(policy, indent=2))
    text = replace_region(text, A.MIRROR_BEGIN, A.MIRROR_END,
                          A.render_policy_mirror(policy))
    _, start, stop = A._region(text, A.POLICY_BEGIN, A.POLICY_END)
    policy["prose_sha256"] = hashlib.sha256((text[:start] + text[stop:]).encode()).hexdigest()
    text = replace_region(text, A.POLICY_BEGIN, A.POLICY_END,
                          json.dumps(policy, indent=2))
    path.write_text(text)
    return A.load_policy(path)


def provenance(commands: list[dict] | None = None) -> dict:
    return {
        "schema": "quactlize-box-run-provenance-v2",
        "root_sha": "a" * 40, "root_status": "clean",
        "submodule_status": " " + "c" * 40 + " third_party/actlize (heads/fixture)",
        "actlize_sha": "c" * 40,
        "binary_sha256": "b" * 64, "device_model": "fixture-ppu",
        "pci_identity": "0000:00:00.0", "driver_version": "fixture-driver",
        "sdk_compiler_identity": "fixture-sdk", "argv": ["fixture-runner"],
        "commands": commands or [{"role": "fixture", "argv": ["fixture"], "exit_status": 0}],
        "runner_exit_status": 0,
        "protocol_sample_count": 20,
    }


def attach_run_identity(root: pathlib.Path, value: dict, groups: str) -> None:
    """Publish the independently hashed identity used by the real runners."""
    probe = {
        "schema": "quactlize-box-identity-probe-v1",
        "identity": {
            field: {"value": value[field], "source": "measured"}
            for field in ("device_model", "pci_identity", "driver_version",
                          "sdk_compiler_identity")
        },
        "device_probe": {
            "status": "measured", "method": "fixture-hggc-runtime",
            "reason": "", "runtime_driver_version": "fixture-driver",
            "pci_measurement": "runtime-properties",
            "driver_measurement": "runtime-api", "device_count": 1,
            "selected_ordinal": 0,
            "candidates": [{"ordinal": 0, "name": "fixture-ppu",
                            "compute_capability": "10.0", "compute_units": 72,
                            "pci_identity": "0000:00:00.0"}],
            "property_errors": [],
        },
        "sdk_compiler_probe": {
            "status": "measured", "reason": "",
            "sdk_root_authority": "fixture", "sdk_root": "/fixture/sdk",
            "compiler_path": "/fixture/sdk/bin/hgcc",
            "version_first_line": "fixture-sdk",
            "identity_value": "fixture-sdk",
        },
    }
    (root / "identity-probe.json").write_text(json.dumps(probe))
    sources = {field: probe["identity"][field]["source"]
               for field in probe["identity"]}
    probe_digest = hashlib.sha256(
        json.dumps(probe, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    payload = {
        "schema": "quactlize-box-run-identity-v2",
        "root_sha": value["root_sha"],
        "submodule_status": value["submodule_status"],
        "actlize_sha": value["actlize_sha"],
        "binary_sha256": value["binary_sha256"],
        "device_model": value["device_model"],
        "pci_identity": value["pci_identity"],
        "driver_version": value["driver_version"],
        "sdk_compiler_identity": value["sdk_compiler_identity"],
        "identity_sources": sources,
        "identity_probe_sha256": probe_digest,
        "protocol_sample_count": value["protocol_sample_count"],
        "groups": groups,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = dict(payload, identity_sha256=digest)
    (root / "run-identity.json").write_text(json.dumps(identity))
    value["groups"] = groups
    value["identity_sources"] = sources
    value["identity_probe_sha256"] = probe_digest
    value["run_identity_sha256"] = digest


def rewrite_run_identity(root: pathlib.Path, mutate, *, mirror_provenance: bool = False) -> None:
    """Mutate and re-sign identity; optionally make provenance agree with the lie."""
    path = root / "run-identity.json"
    identity = json.loads(path.read_text())
    mutate(identity)
    payload = {key: value for key, value in identity.items()
               if key != "identity_sha256"}
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(identity))
    if mirror_provenance:
        provenance_path = root / "provenance.json"
        provenance_value = json.loads(provenance_path.read_text())
        for field in (
                "root_sha", "submodule_status", "actlize_sha", "binary_sha256",
                "device_model", "pci_identity", "driver_version", "sdk_compiler_identity",
                "identity_sources", "identity_probe_sha256",
                "protocol_sample_count", "groups"):
            provenance_value[field] = identity[field]
        provenance_value["run_identity_sha256"] = identity["identity_sha256"]
        provenance_path.write_text(json.dumps(provenance_value))


def rewrite_provenance_commands(root: pathlib.Path, mutate) -> None:
    """Mutate the embedded and standalone command journals as one fact."""
    path = root / "provenance.json"
    value = json.loads(path.read_text())
    mutate(value["commands"])
    path.write_text(json.dumps(value))
    (root / "commands.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in value["commands"]))


def dense_log(median: Decimal, *, bpc: int = 1, cap: int = 6,
              low: Decimal | None = None, high: Decimal | None = None,
              prereq: bool = True) -> str:
    dec = A._marlin_dense_decomposition(32, 32, 72, bpc)
    locks = "".join(
        f"  [dense marlin lock fingerprint] repeat={i}/8 raw_bitdiff=0 x stable=1 "
        "same-workspace=1 external-lock-reset=0\n" for i in range(1, 9))
    disposition = "Passed" if prereq else "Failed"
    low = median if low is None else low
    high = median if high is None else high
    return (
        "  [dense marlin aligned artifact] batch=0 bytes=8388608 x "
        "roundtrip_bad=0/16777216\n"
        "  [streamk fixture exactness] fixture=a0-exact shape=1x4096x4096 x -> "
        "ORDER-INDEPENDENT+FP16-EXACT\n"
        f"  [dense marlin decomposition] real_cu=72 occupancy_api={cap} blocks_per_cu={bpc} "
        f"Q=32 Kt=32 G={dec['grid_ctas']} I={dec['stripe_iters']} "
        f"active={dec['active_ctas']} idle={dec['idle_ctas']} "
        f"handoffs={dec['handoffs']} max_peers={dec['max_peers']} workspace=1\n"
        f"  Disposition: {disposition}\n"
        f"  [dense kernel-span-upper] n=20 median={median} us mean={median} us "
        f"min={low} us max={high} us spread=(max-min)/mean=0.00% "
        "distinct-event-pairs=20 warmup-event-pairs=1 includes-launch-idle=1 "
        "lock-reset-before-start=0\n" + locks +
        "  [dense marlin lock protocol] fixture_identity=a0-exact shape=1x4096x4096 "
        "repeats=8 stable=1 all-bitexact=1 same-workspace=1 external-lock-reset=0\n")


def dense_bundle(root: pathlib.Path, median: Decimal, *, wk1: bool = True,
                 prereq: bool = True, unknown: bool = False, cap: int = 6,
                 low: Decimal | None = None, high: Decimal | None = None) -> None:
    root.mkdir()
    problem = {"--m": 1, "--n": 4096, "--k": 4096, "--l": 1, "--g": 128,
               "--iterations": 20, "--mode": 1, "--alpha": 1, "--beta": 0}
    common = [f"{name}={value}" for name, value in problem.items()]
    commands = [
        {"role": "box-identity-probe", "argv": ["python3", "probe.py"],
         "exit_status": 0},
        {"role": "wk1-static-target", "argv": ["python3", "static.py"], "exit_status": 0},
        {"role": "wk1-committed-production-delivery",
         "argv": ["git", "-C", str(ROOT), "show",
                  "a" * 40 + ":dev/fold_derivation/"
                  "l143_wk4_production_delivery.expected.txt"],
         "exit_status": 0},
        {"role": "device-build", "argv": ["bash", "build.sh"], "exit_status": 0},
    ]
    for bpc in (1, 2, 4, 6):
        if bpc <= cap:
            argv = ["fixture-bin", "--marlin", "--streamk_exact_fixture", *common]
            if bpc != 1:
                argv.append(f"--marlin-blocks-per-cu={bpc}")
            commands.append({"role": f"dense-wk4-bpc{bpc}", "argv": argv, "exit_status": 0})
    commands.append({"role": "dense-wk4-illegal-bpc",
                     "argv": ["fixture-bin", "--marlin", "--streamk_exact_fixture",
                              *common, f"--marlin-blocks-per-cu={cap + 1}"],
                     "exit_status": 2})
    p = provenance(commands)
    p["argv"] = ["tools/run_dense_marlin_wk4_box.sh"]
    attach_run_identity(root, p, "not-applicable")
    (root / "provenance.json").write_text(json.dumps(p))
    (root / "commands.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in commands))
    (root / "submodule-status.txt").write_text(p["submodule_status"] + "\n")
    (root / "runner.log").write_text(
        f"[marlin-wk4] root-sha={p['root_sha']}\n"
        f"[marlin-wk4] actlize-sha={p['actlize_sha']}\n"
        f"[marlin-wk4] binary-sha256={p['binary_sha256']}\n"
        "[marlin-wk4] PASS: classic-aligned WK4 consumer built on shipping bytes; "
        "supported B points passed exact output + 8-launch locks; over-cap B stayed NOT RUN\n")
    (root / "build.log").write_text("fixture build\n")
    (root / "illegal-bpc.log").write_text(
        f"--marlin-blocks-per-cu={cap + 1} is outside the exact kernel occupancy range 1..{cap}\n")
    for bpc in (1, 2, 4, 6):
        if bpc <= cap:
            (root / f"bpc{bpc}.log").write_text(
                dense_log(median, bpc=bpc, cap=cap, low=low, high=high,
                          prereq=prereq))
        else:
            (root / f"bpc{bpc}.not-run").write_text(
                f"[marlin-wk4] NOT RUN: B={bpc} exceeds Gemm::maximum_active_blocks()={cap}\n")
    if unknown:
        (root / "bpc3.log").write_text("unregistered diagnostic\n")
    if wk1:
        (root / "wk1-admission.log").write_text(
            "[dense-marlin-wk4] PASS: isolated 1Mx2Nx4K type/shipping-artifact/CLI; "
            "historical target unchanged; thirteen structural plants rejected\n"
            "[marlin-wk4] wk1-evidence=committed-local-oracle source-sha=" + "a" * 40 +
            " path=dev/fold_derivation/l143_wk4_production_delivery.expected.txt "
            "fresh-box-execution=0\n"
            "L143 direct-pair pairs=8192/8192 codes=16384/16384 "
            "destinations=8192/8192 bad-pairs=0 formula-mismatch=0 bad-fragments=0 "
            "map-diff=0 shipping-hash=b89b157b5b1bd6c3\n"
            "L143 WK1 shipping map-diff=0 byte-diff=0 result=BIT-IDENTICAL\n"
            "L143 shipping-pair-scatter=EXACT artifact-order=RED compact-order=RED "
            "first32=RED wrong-pair=RED source-swap=RED WK1-BYTES=UNCHANGED result=PASS\n")


INCUMBENT = "int4/native/s16/t128/dense/m1/n8/c2"
CHALLENGER = "int4/native/s32/t128/dense/m1/n8/c2"
THIRD = "int4/native/s64/t128/dense/m1/n8/c2"


def make_manifest(values: dict[str, list[str]]) -> tuple[dict, list[str]]:
    configs = {
        INCUMBENT: {"format": "int4", "layout": "native", "step_k": 16,
                    "threads": 128, "route": "dense", "cta_m": 1,
                    "cta_n": 8, "chunk": 2, "tile_size_k": 0},
        CHALLENGER: {"format": "int4", "layout": "native", "step_k": 32,
                     "threads": 128, "route": "dense", "cta_m": 1,
                     "cta_n": 8, "chunk": 2, "tile_size_k": 0},
        THIRD: {"format": "int4", "layout": "native", "step_k": 64,
                "threads": 128, "route": "dense", "cta_m": 1,
                "cta_n": 8, "chunk": 2, "tile_size_k": 0},
    }
    lines = S._raw_records(values=values, config_overrides=configs)
    parsed = [json.loads(line) for line in lines]
    attempts = [x for x in parsed if x["rec"] == "attempt"]
    shape = {"m": 1, "n": 2048, "k": 2048, "experts": 0, "active": 1,
             "format": "int4", "group_size": 32,
             "quant_op": "finegrained_scale_zero", "route": "dense",
             "semantic": "shipping"}
    for record in parsed:
        if "shape" in record:
            record["shape"] = shape
        if "run_id" in record:
            record["run_id"] = (
                "gemv-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb-samples20")
        if record["rec"] == "run":
            record["build"] = f"{'a' * 40}/bin-sha256:{'b' * 64}/protocol:samples20"
    lines = [S.canonical(x) for x in parsed]
    expected = [{"format": x["format"], "config_id": x["config_id"],
                 "config": x["config"]} for x in attempts]
    return {
        "schema": S.MANIFEST_SCHEMA, "space_id": "full", "partial_space": False,
        "counts": {"total": len(expected), "legal": len(expected), "pruned": 0,
                   "prune_reasons": {}},
        "jobs": [{"job_id": "j", "shape_id": "shape", "shape": shape,
                  "argv": [], "env": {}, "formats": ["int4"], "expected": expected}],
    }, lines


def gemv_bundle(root: pathlib.Path, manifest: dict, lines: list[str]) -> None:
    root.mkdir()
    commands = [
        {"role": "box-identity-probe", "argv": ["python3", "probe.py"],
         "exit_status": 0},
        {"role": "device-build", "argv": ["bash", "build.sh"], "exit_status": 0},
        {"role": "base-tactic-census", "argv": ["python3", "census.py"], "exit_status": 0},
        {"role": "manifest", "argv": ["fixture-bin", "--manifest-json"], "exit_status": 0},
        {"role": "dry-run-audit", "argv": ["python3", "sweep.py", "--dry-run"], "exit_status": 0},
        {"role": "measured-sweep", "argv": ["python3", "sweep.py", "run"], "exit_status": 0},
        {"role": "analyse", "argv": ["python3", "sweep.py", "analyse"], "exit_status": 0},
        {"role": "analyse-completeness", "argv": ["python3", "-c", "check"],
         "exit_status": 0},
    ]
    p = provenance(commands)
    p["argv"] = ["tools/run_gemv_sweep_box.sh"]
    attach_run_identity(root, p, "all")
    (root / "provenance.json").write_text(json.dumps(p))
    (root / "commands.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in commands))
    (root / "submodule-status.txt").write_text(p["submodule_status"] + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "raw.jsonl").write_text("\n".join(lines) + "\n")
    (root / "base-census.json").write_text(json.dumps({
        "schema": "quactlize-gemv-base-census-v1",
        "total": 2, "legal": 2, "pruned": 0, "prune_reasons": {},
    }))
    (root / "logs").mkdir()
    (root / "build.log").write_text("fixture build\n")
    (root / "runner.log").write_text("fixture runner\n")
    (root / "base-census-authority.log").write_text(
        "CENSUS,total,2\nCENSUS,legal,2\nCENSUS,rejected,0\nRESULT,PASS\n")
    for name in ("progress.jsonl", "result.json", "run.log", "pending.audit.jsonl",
                 "pending.summary.jsonl"):
        (root / name).write_text("{}\n")


def sample_series(start: Decimal, count: int = 20) -> list[str]:
    return [str(start + Decimal("0.01") * i) for i in range(count)]


def main() -> None:
    loaded = A.load_policy()
    dense = loaded.value["dense"]
    classic = Decimal(dense["classic_anchor_us"])
    old = Decimal(dense["historical_anchor_us"])
    fraction = Decimal(dense["converged_recovered_fraction"])
    boundary = old - fraction * (old - classic)
    with tempfile.TemporaryDirectory(prefix="box-adjudicator-") as td_raw:
        td = pathlib.Path(td_raw)
        # Publication authority is the result SHA, not today's worktree file.
        git_repo = td / "git-policy"
        policy_rel = pathlib.Path("dev/fold_derivation/BOX_RUN_PREREGISTRATION.md")
        (git_repo / policy_rel.parent).mkdir(parents=True)
        (git_repo / policy_rel).write_text(TEMPLATE)
        for relative in A.PUBLICATION_CODE_PATHS:
            target = git_repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        subprocess.run(["git", "init", "-q", git_repo], check=True)
        subprocess.run(["git", "-C", git_repo, "add", "."], check=True)
        subprocess.run(["git", "-C", git_repo, "-c", "user.name=fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "policy"],
                       check=True)
        sha = subprocess.check_output(["git", "-C", git_repo, "rev-parse", "HEAD"], text=True).strip()
        from_sha = A.load_policy_from_git(sha, git_repo)
        check(from_sha.policy_sha256 == loaded.policy_sha256 and from_sha.source.startswith("git:"),
              "publication policy was not loaded from the recorded result SHA")
        check(not A.publication_code_errors(sha, git_repo, ROOT),
              "unchanged publication code was rejected")
        drift = td / "drifted-reader"
        for relative in A.PUBLICATION_CODE_PATHS:
            target = drift / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        analyzer = drift / "benchmarks/sweep_gemv_perf.py"
        analyzer.write_text(analyzer.read_text() + "\n# planted two-quantum reinterpretation\n")
        check(any("benchmarks/sweep_gemv_perf.py differs from result SHA" in error
                  for error in A.publication_code_errors(sha, git_repo, drift)),
              "changed analyser was allowed to reinterpret an older bundle")
        shutil.copy2(ROOT / "benchmarks/sweep_gemv_perf.py", analyzer)
        reader = drift / "tools/adjudicate_box_runs.py"
        reader.write_text(reader.read_text() + "\n# planted post-result threshold\n")
        check(any("tools/adjudicate_box_runs.py differs from result SHA" in error
                  for error in A.publication_code_errors(sha, git_repo, drift)),
              "changed adjudicator was allowed to reinterpret an older bundle")
        for name, median, expected in (
                ("converged", boundary - Decimal("0.1"), "CONVERGED_OR_BETTER"),
                ("boundary", boundary, "CONVERGED_OR_BETTER"),
                ("partial", boundary + Decimal("0.1"), "PARTIAL"),
                ("no-recovery", old + Decimal("0.1"), "NO_RECOVERY_OR_WORSE")):
            bundle = td / name
            dense_bundle(bundle, median, unknown=(name == "converged"))
            result = A.adjudicate_dense_bundle(loaded, bundle)
            check_three_sections(result, name)
            check(result["registered_verdict"] == expected, f"{name}: {result}")
            if name == "boundary":
                check(not result["boundary_unresolved"],
                      "a singleton exactly on the registered boundary was called crossing")
            if name == "converged":
                check(len(result["unregistered_observations"]) == 1,
                      "unregistered diagnostic must remain separate without voiding registered result")
        crossing = td / "boundary-crossing"
        dense_bundle(crossing, boundary + Decimal("0.02"),
                     low=boundary - Decimal("0.01"), high=boundary + Decimal("0.03"))
        result = A.adjudicate_dense_bundle(loaded, crossing)
        check(result["registered_verdict"] == "PARTIAL" and result["boundary_unresolved"],
              "boundary-crossing band was not reported unresolved")
        bundle = td / "void"
        dense_bundle(bundle, classic, wk1=False)
        result = A.adjudicate_dense_bundle(loaded, bundle)
        check(result["registered_verdict"] == "VOID" and
              any("wk1-admission" in x for x in result["reasons"]),
              f"missing WK1 evidence did not void: {result}")
        missing_identity = td / "missing-dense-run-identity"
        dense_bundle(missing_identity, classic)
        (missing_identity / "run-identity.json").unlink()
        result = A.adjudicate_dense_bundle(loaded, missing_identity)
        check(result["registered_verdict"] == "VOID" and
              any("run-identity.json" in x for x in result["reasons"]),
              "a dense bundle without immutable run identity was admitted")
        tampered_probe_source = td / "tampered-dense-probe-source"
        dense_bundle(tampered_probe_source, classic)
        probe_path = tampered_probe_source / "identity-probe.json"
        probe = json.loads(probe_path.read_text())
        probe["identity"]["device_model"]["source"] = "operator"
        probe_path.write_text(json.dumps(probe))
        result = A.adjudicate_dense_bundle(loaded, tampered_probe_source)
        check(result["registered_verdict"] == "VOID" and
              any("identity-probe" in x and
                  ("sources" in x or "digest" in x or "self-contradictory" in x)
                  for x in result["reasons"]),
              "a tampered measured/operator source was admitted")
        tampered_probe_evidence = td / "tampered-dense-probe-evidence"
        dense_bundle(tampered_probe_evidence, classic)
        probe_path = tampered_probe_evidence / "identity-probe.json"
        probe = json.loads(probe_path.read_text())
        probe["device_probe"]["candidates"][0]["name"] = "planted-other-device"
        probe_path.write_text(json.dumps(probe))
        result = A.adjudicate_dense_bundle(loaded, tampered_probe_evidence)
        check(result["registered_verdict"] == "VOID" and
              any("identity-probe" in x and
                  ("canonical digest" in x or "self-contradictory" in x)
                  for x in result["reasons"]),
              "tampered probe evidence bytes were admitted")
        resigned_contradiction = td / "resigned-contradictory-probe"
        dense_bundle(resigned_contradiction, classic)
        probe_path = resigned_contradiction / "identity-probe.json"
        probe = json.loads(probe_path.read_text())
        probe["device_probe"]["device_count"] = 2
        probe["device_probe"]["candidates"].append({
            "ordinal": 1, "name": "other-visible-ppu",
            "compute_capability": "10.0", "compute_units": 72,
            "pci_identity": "0000:01:00.0",
        })
        probe_path.write_text(json.dumps(probe))
        probe_digest = hashlib.sha256(
            json.dumps(probe, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
        rewrite_run_identity(
            resigned_contradiction,
            lambda identity: identity.__setitem__(
                "identity_probe_sha256", probe_digest),
            mirror_provenance=True)
        result = A.adjudicate_dense_bundle(loaded, resigned_contradiction)
        check(result["registered_verdict"] == "VOID" and
              any("self-contradictory" in x for x in result["reasons"]),
              "a fully re-signed two-device probe claiming measured identity was admitted")
        stale_v1_identity = td / "stale-v1-identity"
        dense_bundle(stale_v1_identity, classic)
        identity_path = stale_v1_identity / "run-identity.json"
        identity = json.loads(identity_path.read_text())
        identity["schema"] = "quactlize-box-run-identity-v1"
        payload = {key: value for key, value in identity.items()
                   if key != "identity_sha256"}
        identity["identity_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        identity_path.write_text(json.dumps(identity))
        result = A.adjudicate_dense_bundle(loaded, stale_v1_identity)
        check(result["registered_verdict"] == "VOID" and
              any("schema differs" in x for x in result["reasons"]),
              "a re-signed v1 identity was admitted by the v2 reader")
        tampered_identity = td / "tampered-dense-run-identity"
        dense_bundle(tampered_identity, classic)
        rewrite_run_identity(
            tampered_identity,
            lambda identity: identity.__setitem__("device_model", "other-ppu"))
        result = A.adjudicate_dense_bundle(loaded, tampered_identity)
        check(result["registered_verdict"] == "VOID" and
              any("device_model differs from provenance" in x
                  for x in result["reasons"]),
              "a re-signed dense identity that differs from provenance was admitted")
        wrong_dense_groups = td / "wrong-dense-groups"
        dense_bundle(wrong_dense_groups, classic)
        rewrite_run_identity(
            wrong_dense_groups,
            lambda identity: identity.__setitem__("groups", "all"),
            mirror_provenance=True)
        result = A.adjudicate_dense_bundle(loaded, wrong_dense_groups)
        check(result["registered_verdict"] == "VOID" and
              any("differs from 'not-applicable'" in x for x in result["reasons"]),
              "a dense identity carrying a GEMV build selection was admitted")
        false_wk1 = td / "false-wk1"
        dense_bundle(false_wk1, classic)
        text = (false_wk1 / "wk1-admission.log").read_text().replace(
            "WK1-BYTES=UNCHANGED result=PASS", "WK1-BYTES=UNCHANGED result=FAIL")
        (false_wk1 / "wk1-admission.log").write_text(text + "unrelated result=PASS\n")
        result = A.adjudicate_dense_bundle(loaded, false_wk1)
        check_three_sections(result, "false-wk1")
        check(result["registered_verdict"] == "VOID" and all(
            x["verdict"] == "NOT_ADJUDICATED" for x in result["cell_results"]),
            "unrelated PASS masked a failed final WK1 line or interpretation continued after VOID")
        missing_rung = td / "missing-rung"
        dense_bundle(missing_rung, classic)
        (missing_rung / "bpc4.log").unlink()
        result = A.adjudicate_dense_bundle(loaded, missing_rung)
        check(result["registered_verdict"] == "VOID" and
              any("WK4/B4" in x for x in result["reasons"]),
              "missing registered B rung did not fail closed")
        cap4 = td / "cap4"
        dense_bundle(cap4, classic, cap=4)
        result = A.adjudicate_dense_bundle(loaded, cap4)
        check(result["registered_verdict"] == "CONVERGED_OR_BETTER" and
              any(x.get("blocks_per_cu") == 6 and x.get("verdict") == "NOT_RUN"
                  for x in result["cell_results"]),
              "exact over-cap B6 NOT RUN was not admitted")
        b2 = next(x for x in result["cell_results"]
                  if x.get("warp_k") == 4 and x.get("blocks_per_cu") == 2)
        check(b2["verdict"] == "DIAGNOSTIC" and b2["median_us"] == str(classic) and
              b2["decomposition"]["handoffs"] == 96,
              "per-rung timing/decomposition evidence was dropped from the adjudication")
        cap2 = td / "cap2"
        dense_bundle(cap2, classic, cap=2)
        result = A.adjudicate_dense_bundle(loaded, cap2)
        cap2_cells = {(x.get("warp_k"), x.get("blocks_per_cu")): x
                      for x in result["cell_results"]}
        check(result["registered_verdict"] == "CONVERGED_OR_BETTER" and
              cap2_cells[(4, 2)]["verdict"] == "DIAGNOSTIC" and
              cap2_cells[(4, 4)]["verdict"] == "NOT_RUN" and
              cap2_cells[(4, 6)]["verdict"] == "NOT_RUN",
              "occupancy_api=2 did not produce B1/B2 RUN plus explicit B4/B6 NOT RUN")
        cap2_missing = td / "cap2-missing-not-run"
        dense_bundle(cap2_missing, classic, cap=2)
        (cap2_missing / "bpc4.not-run").unlink()
        result = A.adjudicate_dense_bundle(loaded, cap2_missing)
        check(result["registered_verdict"] == "VOID" and
              any("WK4/B4" in x for x in result["reasons"]),
              "occupancy_api=2 silently skipped a missing B4 NOT RUN artifact")
        bad_decomposition = td / "bad-decomposition"
        dense_bundle(bad_decomposition, classic)
        path = bad_decomposition / "bpc4.log"
        path.write_text(path.read_text().replace(
            "G=288 I=4 active=256 idle=32 handoffs=224 max_peers=8 workspace=1",
            "G=999 I=999 active=999 idle=999 handoffs=999 max_peers=999 workspace=1"))
        result = A.adjudicate_dense_bundle(loaded, bad_decomposition)
        check(result["registered_verdict"] == "VOID" and
              any("flat-stripe oracle" in x for x in result["reasons"]),
              "self-consistent-looking but false per-rung decomposition was published")
        bad_illegal = td / "bad-illegal"
        dense_bundle(bad_illegal, classic)
        (bad_illegal / "illegal-bpc.log").write_text("generic failure\n")
        result = A.adjudicate_dense_bundle(loaded, bad_illegal)
        check(result["registered_verdict"] == "VOID",
              "generic illegal-B failure was accepted as the exact cap rejection")
        dirty = td / "dirty-submodule"
        dense_bundle(dirty, classic)
        p = json.loads((dirty / "provenance.json").read_text())
        p["submodule_status"] = "+" + p["actlize_sha"] + " third_party/actlize"
        (dirty / "provenance.json").write_text(json.dumps(p))
        result = A.adjudicate_dense_bundle(loaded, dirty)
        check(result["registered_verdict"] == "VOID" and
              any("submodule_status" in x for x in result["reasons"]),
              "dirty submodule was admitted")
        wrong_gitlink = td / "wrong-gitlink"
        dense_bundle(wrong_gitlink, classic)
        p = json.loads((wrong_gitlink / "provenance.json").read_text())
        p["actlize_sha"] = "d" * 40
        (wrong_gitlink / "provenance.json").write_text(json.dumps(p))
        result = A.adjudicate_dense_bundle(loaded, wrong_gitlink)
        check(result["registered_verdict"] == "VOID" and
              any("actlize_sha differs" in x for x in result["reasons"]),
              "an actlize SHA that disagrees with the recursive gitlink was admitted")
        duplicate = td / "duplicate"
        dense_bundle(duplicate, classic)
        (duplicate / "bpc01.log").write_text(dense_log(classic))
        result = A.adjudicate_dense_bundle(loaded, duplicate)
        check(result["registered_verdict"] == "VOID" and
              any("duplicate dense cell" in x for x in result["reasons"]),
              "duplicate registered dense cell did not fail closed")

        bad_default = td / "bad-default-command"
        dense_bundle(bad_default, classic)
        rewrite_provenance_commands(
            bad_default,
            lambda rows: next(row for row in rows
                              if row["role"] == "dense-wk4-bpc1")["argv"].append(
                                  "--marlin-blocks-per-cu=1"))
        result = A.adjudicate_dense_bundle(loaded, bad_default)
        check(result["registered_verdict"] == "VOID" and
              any("B1 used an explicit override" in x for x in result["reasons"]),
              "an explicitly overridden B1 was accepted as the default path")

        stale_wk1_role = td / "stale-wk1-role"
        dense_bundle(stale_wk1_role, classic)
        def restore_stale_wk1_role(rows):
            next(row for row in rows
                 if row["role"] == "wk1-committed-production-delivery")["role"] = \
                "wk1-production-delivery"
        rewrite_provenance_commands(stale_wk1_role, restore_stale_wk1_role)
        result = A.adjudicate_dense_bundle(loaded, stale_wk1_role)
        check(result["registered_verdict"] == "VOID" and
              any("wk1-committed-production-delivery" in x for x in result["reasons"]),
              "the stale role that implied fresh box execution was admitted")

        wrong_wk1_source = td / "wrong-wk1-source"
        dense_bundle(wrong_wk1_source, classic)
        def change_wk1_source(rows):
            next(row for row in rows
                 if row["role"] == "wk1-committed-production-delivery")["argv"][-1] = \
                "a" * 40 + ":dev/fold_derivation/other.expected.txt"
        rewrite_provenance_commands(wrong_wk1_source, change_wk1_source)
        result = A.adjudicate_dense_bundle(loaded, wrong_wk1_source)
        check(result["registered_verdict"] == "VOID" and
              any("exact result-SHA git show" in x for x in result["reasons"]),
              "WK1 evidence from an unregistered result-SHA path was admitted")

        fake_box_wk1 = td / "fake-box-wk1"
        dense_bundle(fake_box_wk1, classic)
        path = fake_box_wk1 / "wk1-admission.log"
        path.write_text(path.read_text().replace(
            "fresh-box-execution=0", "fresh-box-execution=1", 1))
        result = A.adjudicate_dense_bundle(loaded, fake_box_wk1)
        check(result["registered_verdict"] == "VOID" and
              any("committed-not-box" in x for x in result["reasons"]),
              "committed host evidence was allowed to masquerade as a fresh box execution")

        conflicting_shape = td / "conflicting-shape-command"
        dense_bundle(conflicting_shape, classic)
        rewrite_provenance_commands(
            conflicting_shape,
            lambda rows: next(row for row in rows
                              if row["role"] == "dense-wk4-bpc1")["argv"].append("--n=8192"))
        result = A.adjudicate_dense_bundle(loaded, conflicting_shape)
        check(result["registered_verdict"] == "VOID" and
              any("--n=4096" in x for x in result["reasons"]),
              "a conflicting duplicate dense shape option was admitted")

        extra_dense_command = td / "extra-dense-command"
        dense_bundle(extra_dense_command, classic)
        rewrite_provenance_commands(
            extra_dense_command,
            lambda rows: rows.append({"role": "unregistered-device-probe",
                                      "argv": ["fixture-bin", "--probe"],
                                      "exit_status": 0}))
        result = A.adjudicate_dense_bundle(loaded, extra_dense_command)
        check(result["registered_verdict"] == "CONVERGED_OR_BETTER" and
              any(x.get("command_role") == "unregistered-device-probe"
                  for x in result["unregistered_observations"]),
              "an unregistered dense command was silently omitted from section three")

        bad_cap = td / "bad-cap-command"
        dense_bundle(bad_cap, classic)
        def make_illegal_succeed(rows):
            next(row for row in rows
                 if row["role"] == "dense-wk4-illegal-bpc")["exit_status"] = 0
        rewrite_provenance_commands(bad_cap, make_illegal_succeed)
        result = A.adjudicate_dense_bundle(loaded, bad_cap)
        check(result["registered_verdict"] == "VOID" and
              any("illegal-B command" in x for x in result["reasons"]),
              "a successful illegal-B launch was hidden by a correct diagnostic log")

        wrong_occupancy = td / "wrong-occupancy"
        dense_bundle(wrong_occupancy, classic)
        path = wrong_occupancy / "bpc2.log"
        path.write_text(path.read_text().replace("occupancy_api=6", "occupancy_api=5", 1))
        result = A.adjudicate_dense_bundle(loaded, wrong_occupancy)
        check(result["registered_verdict"] == "VOID" and
              any("differs from B1 cap" in x for x in result["reasons"]),
              "a B rung with a different occupancy cap was admitted")

        duplicate_lock = td / "duplicate-lock"
        dense_bundle(duplicate_lock, classic)
        path = duplicate_lock / "bpc1.log"
        path.write_text(path.read_text().replace("repeat=2/8", "repeat=1/8", 1))
        result = A.adjudicate_dense_bundle(loaded, duplicate_lock)
        check(result["registered_verdict"] == "VOID" and
              any("lock_fingerprints_8_stable" in x for x in result["reasons"]),
              "eight lock lines with a duplicated repeat ID were accepted")

        unrelated_pass = td / "unrelated-pass"
        dense_bundle(unrelated_pass, classic)
        path = unrelated_pass / "bpc1.log"
        path.write_text(path.read_text().replace("  Disposition: Passed\n",
                                                "  Disposition: Failed\n", 1) +
                        "  Disposition: Passed\n")
        result = A.adjudicate_dense_bundle(loaded, unrelated_pass)
        check(result["registered_verdict"] == "VOID" and
              any("correctness" in x for x in result["reasons"]),
              "an unrelated Passed line masked the primary failed disposition")

        wrong_protocol = td / "wrong-protocol"
        dense_bundle(wrong_protocol, classic)
        p = json.loads((wrong_protocol / "provenance.json").read_text())
        p["protocol_sample_count"] = 19
        (wrong_protocol / "provenance.json").write_text(json.dumps(p))
        result = A.adjudicate_dense_bundle(loaded, wrong_protocol)
        check(result["registered_verdict"] == "VOID" and
              any("provenance sample count" in x for x in result["reasons"]),
              "a 19-sample provenance claim was admitted")

        duplicate_provenance = td / "duplicate-provenance-key"
        dense_bundle(duplicate_provenance, classic)
        path = duplicate_provenance / "provenance.json"
        text = path.read_text().replace(
            '"root_sha": "' + "a" * 40 + '"',
            '"root_sha": "' + "d" * 40 + '", "root_sha": "' + "a" * 40 + '"', 1)
        path.write_text(text)
        result = A.adjudicate_dense_bundle(loaded, duplicate_provenance)
        check(result["registered_verdict"] == "VOID" and
              any("duplicate JSON key" in x for x in result["reasons"]),
              "duplicate provenance root_sha was silently last-wins")

        wrong_terminator = td / "wrong-runner-terminator"
        dense_bundle(wrong_terminator, classic)
        path = wrong_terminator / "runner.log"
        path.write_text(path.read_text().replace(
            "[marlin-wk4] PASS: classic-aligned WK4 consumer built on shipping bytes; "
            "supported B points passed exact output + 8-launch locks; over-cap B stayed NOT RUN",
            "[marlin-wk4] PASS: unrelated"))
        result = A.adjudicate_dense_bundle(loaded, wrong_terminator)
        check(result["registered_verdict"] == "VOID" and
              any("exact unique runner PASS" in x for x in result["reasons"]),
              "an unrelated runner PASS substituted for the registered terminator")

        # The JSON authority, generated mirror, and prose are a three-way seal.
        prose = td / "prose.md"
        prose.write_text(TEMPLATE.replace("The JSON block above", "The JSON authority above", 1))
        try:
            A.load_policy(prose)
            raise AssertionError("prose drift was accepted")
        except A.PolicyError:
            pass
        block = td / "block.md"
        block.write_text(TEMPLATE.replace('"converged_recovered_fraction": "0.75"',
                                          '"converged_recovered_fraction": "0.74"', 1))
        try:
            A.load_policy(block)
            raise AssertionError("block/mirror drift was accepted")
        except A.PolicyError:
            pass

        # Reanalyse raw events with the policy floor and exact full census.
        q = Decimal(loaded.value["gemv"]["minimum_claimable_us"])
        values = {INCUMBENT: sample_series(Decimal("1.40")),
                  CHALLENGER: sample_series(Decimal("1.00"))}
        manifest, lines = make_manifest(values)
        small = copy.deepcopy(loaded.value)
        small["gemv"]["base_census"] = {
            "total": 2, "legal": 2, "pruned": 0, "prune_reasons": {}}
        small["gemv"]["full_manifest"] = {
            "jobs": 1, "total": 2, "legal": 2, "pruned": 0, "prune_reasons": {}}
        small["gemv"]["incumbent_rules"]["geometry_cta_m"] = {"shape": 1}
        small["gemv"]["incumbent_rules"]["format_axes"] = {
            "int4": small["gemv"]["incumbent_rules"]["format_axes"]["int4"]}
        small_policy = write_policy(td / "small.md", small)
        gb = td / "gemv"
        gemv_bundle(gb, manifest, lines)
        result = A.adjudicate_gemv_bundle(small_policy, gb)
        check_three_sections(result, "gemv-resolved")
        check(result["registered_verdict"] == "ADJUDICATED", f"gemv: {result}")
        cell = result["cell_results"][0]
        check(cell["measurement_verdict"] == "RESOLVED" and
              cell["runner_up"] is not None and
              cell["resolution_floor"]["minimum_claimable_us"] == float(q) and
              cell["routing_verdict"] == "CHANGED_FROM_SHIPPING_ROUTE" and
              cell["leader_evidence"]["raw_samples"] == 20 and
              cell["runner_up_evidence"]["raw_samples"] == 20 and
              len(cell["leader_evidence"]["raw_band_us"]) == 2,
              f"gemv changed cell: {cell}")
        missing_gemv_identity = td / "missing-gemv-run-identity"
        shutil.copytree(gb, missing_gemv_identity)
        (missing_gemv_identity / "run-identity.json").unlink()
        result = A.adjudicate_gemv_bundle(small_policy, missing_gemv_identity)
        check(result["registered_verdict"] == "VOID" and
              any("run-identity.json" in x for x in result["reasons"]),
              "a GEMV bundle without immutable run identity was admitted")
        wrong_gemv_groups = td / "wrong-gemv-groups"
        shutil.copytree(gb, wrong_gemv_groups)
        rewrite_run_identity(
            wrong_gemv_groups,
            lambda identity: identity.__setitem__("groups", "i4-native"),
            mirror_provenance=True)
        result = A.adjudicate_gemv_bundle(small_policy, wrong_gemv_groups)
        check(result["registered_verdict"] == "VOID" and
              any("differs from 'all'" in x for x in result["reasons"]),
              "a bounded-group GEMV identity was admitted as a full sweep")
        (gb / "unregistered-note.txt").write_text("observation outside preregistration\n")
        result = A.adjudicate_gemv_bundle(small_policy, gb)
        check(result["registered_verdict"] == "ADJUDICATED" and
              result["unregistered_observations"] == [{
                  "artifact": "unregistered-note.txt",
                  "reason": "not covered by preregistered bundle contract"}],
              "an unregistered observation was hidden or forced into a registered verdict")
        (gb / "unregistered-note.txt").unlink()

        extra_gemv_command = td / "extra-gemv-command"
        shutil.copytree(gb, extra_gemv_command)
        rewrite_provenance_commands(
            extra_gemv_command,
            lambda rows: rows.append({"role": "unregistered-device-probe",
                                      "argv": ["fixture-bin", "--probe"],
                                      "exit_status": 0}))
        result = A.adjudicate_gemv_bundle(small_policy, extra_gemv_command)
        check(result["registered_verdict"] == "ADJUDICATED" and
              any(x.get("command_role") == "unregistered-device-probe"
                  for x in result["unregistered_observations"]),
              "an unregistered GEMV command was silently omitted from section three")

        wrong_timer = copy.deepcopy(small)
        wrong_timer["gemv"]["timer_normalization_us"] = "0.002"
        wrong_timer_policy = write_policy(td / "wrong-timer-policy.md", wrong_timer)
        result = A.adjudicate_gemv_bundle(wrong_timer_policy, gb)
        check(result["registered_verdict"] == "VOID" and
              any("timer normalization" in x for x in result["reasons"]),
              "an analyser using a different timer normalization was admitted")

        # The same policy must mechanically distinguish unchanged and unresolved.
        unchanged = td / "gemv-unchanged"
        um, ul = make_manifest({INCUMBENT: sample_series(Decimal("1.00")),
                                CHALLENGER: sample_series(Decimal("1.40"))})
        gemv_bundle(unchanged, um, ul)
        ur = A.adjudicate_gemv_bundle(small_policy, unchanged)
        uc = ur["cell_results"][0]
        check(uc["routing_verdict"] == "SHIPPING_ROUTE_REMAINS_LEADER" and
              uc["runner_up"] == CHALLENGER and uc["leader_runner_gap_us"] is not None,
              f"unchanged route lost runner-up evidence: {uc}")
        unresolved = td / "gemv-unresolved"
        xm, xl = make_manifest({INCUMBENT: sample_series(Decimal("1.00")),
                                CHALLENGER: sample_series(Decimal("1.10"))})
        gemv_bundle(unresolved, xm, xl)
        xr = A.adjudicate_gemv_bundle(small_policy, unresolved)
        xc = xr["cell_results"][0]
        check(xc["measurement_verdict"] == "UNRESOLVED" and
              xc["routing_verdict"] == "UNRESOLVED" and
              "BAND_OVERLAP" in xc["measurement_reasons"],
              f"overlapping raw bands did not remain unresolved: {xc}")

        # Non-overlapping bands exactly one inferred quantum apart are still
        # unresolved.  This is the user's resolution-floor rule, distinct from
        # the overlap control above.
        one_quantum = td / "gemv-one-quantum"
        three = copy.deepcopy(small)
        three["gemv"]["full_manifest"] = {
            "jobs": 1, "total": 3, "legal": 3, "pruned": 0, "prune_reasons": {}}
        three_policy = write_policy(td / "three.md", three)
        qm, ql = make_manifest({
            INCUMBENT: ["1.00"] * 20,
            CHALLENGER: ["1.01"] * 20,
            THIRD: ["1.03"] * 20,
        })
        gemv_bundle(one_quantum, qm, ql)
        qr = A.adjudicate_gemv_bundle(three_policy, one_quantum)
        qc = qr["cell_results"][0]
        check(qc["measurement_verdict"] == "UNRESOLVED" and
              qc["measurement_reasons"] == ["WITHIN_ONE_QUANTUM"] and
              qc["registered_resolution"]["normalized_gap_us"] == "0.010" and
              qc["registered_resolution"]["unresolved_limit_us"] == "0.01" and
              qc["leader_evidence"]["raw_band_us"][1] <
              qc["runner_up_evidence"]["raw_band_us"][0],
              f"non-overlap gap exactly at one quantum was resolved: {qc}")

        two_quanta = td / "gemv-two-quanta"
        tm, tl = make_manifest({
            INCUMBENT: ["1.00"] * 20,
            CHALLENGER: ["1.02"] * 20,
            THIRD: ["1.03"] * 20,
        })
        gemv_bundle(two_quanta, tm, tl)
        tr = A.adjudicate_gemv_bundle(three_policy, two_quanta)
        tc = tr["cell_results"][0]
        check(tc["measurement_verdict"] == "RESOLVED" and
              tc["registered_resolution"]["normalized_gap_us"] == "0.020",
              f"non-overlap gap above the registered quantum limit stayed unresolved: {tc}")

        # The analyzer's convenience verdict uses its own one-quantum rule.
        # Change only the sealed policy to two quanta: the exact same decoded
        # facts must now be UNRESOLVED without becoming VOID.  This proves that
        # analyzer labels are not a second adjudication authority.
        two_limit = copy.deepcopy(three)
        two_limit["gemv"]["resolution_rule"]["max_unresolved_quanta"] = "2"
        two_limit_policy = write_policy(td / "two-limit.md", two_limit)
        tlr = A.adjudicate_gemv_bundle(two_limit_policy, two_quanta)
        tlc = tlr["cell_results"][0]
        check(tlr["registered_verdict"] == "ADJUDICATED" and
              tlc["measurement_verdict"] == "UNRESOLVED" and
              tlc["measurement_reasons"] == ["WITHIN_REGISTERED_QUANTUM_LIMIT"] and
              tlc["registered_resolution"]["unresolved_limit_us"] == "0.02",
              "analyzer convenience verdict overrode the sealed two-quantum policy")

        # A three-sample/19-sample shortcut must never satisfy the registered 20-launch protocol.
        short = td / "gemv-short"
        sm, sl = make_manifest({INCUMBENT: sample_series(Decimal("1.00"), 19),
                                CHALLENGER: sample_series(Decimal("1.40"), 19)})
        gemv_bundle(short, sm, sl)
        sr = A.adjudicate_gemv_bundle(small_policy, short)
        check(sr["registered_verdict"] == "VOID" and
              any("expected_samples=19" in x for x in sr["reasons"]),
              "non-20-sample GEMV raw was admitted")

        duplicate_attempt = td / "gemv-duplicate-attempt"
        dm, dl = make_manifest({INCUMBENT: sample_series(Decimal("1.00")),
                                CHALLENGER: sample_series(Decimal("1.40"))})
        parsed = [json.loads(line) for line in dl]
        second_attempt = []
        for record in parsed:
            if record.get("rec") in ("attempt", "sample"):
                planted = copy.deepcopy(record)
                planted["attempt_id"] = "1"
                second_attempt.append(S.canonical(planted))
        gemv_bundle(duplicate_attempt, dm, dl + second_attempt)
        result = A.adjudicate_gemv_bundle(small_policy, duplicate_attempt)
        check(result["registered_verdict"] == "VOID" and
              any("more than one attempt" in x for x in result["reasons"]),
              "two complete 20-sample attempts were merged into one candidate")

        duplicate_run = td / "gemv-duplicate-run"
        rm, rl = make_manifest({INCUMBENT: sample_series(Decimal("1.00")),
                                CHALLENGER: sample_series(Decimal("1.40"))})
        parsed = [json.loads(line) for line in rl]
        second_run = []
        for record in parsed:
            planted = copy.deepcopy(record)
            planted["run_id"] = "planted-second-run"
            second_run.append(S.canonical(planted))
        gemv_bundle(duplicate_run, rm, rl + second_run)
        result = A.adjudicate_gemv_bundle(small_policy, duplicate_run)
        check(result["registered_verdict"] == "VOID" and
              any("exactly one run identity" in x for x in result["reasons"]),
              "two raw run identities were co-ranked")
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest["counts"] = {"total": 3, "legal": 2, "pruned": 1,
                                  "prune_reasons": {"PLANTED": 1}}
        (gb / "manifest.json").write_text(json.dumps(bad_manifest))
        result = A.adjudicate_gemv_bundle(small_policy, gb)
        check(result["registered_verdict"] == "VOID", "census drift did not void")
        (gb / "manifest.json").write_text(json.dumps(manifest))
        base = json.loads((gb / "base-census.json").read_text())
        base["pruned"] = 1
        (gb / "base-census.json").write_text(json.dumps(base))
        result = A.adjudicate_gemv_bundle(small_policy, gb)
        check(result["registered_verdict"] == "VOID", "base prune census drift did not void")
        base["pruned"] = 0
        (gb / "base-census.json").write_text(json.dumps(base))
        path = gb / "base-census-authority.log"
        path.write_text(path.read_text() + "EXCLUSION,PLANTED,1\n")
        result = A.adjudicate_gemv_bundle(small_policy, gb)
        check(result["registered_verdict"] == "VOID" and
              any("exact registered histogram" in x for x in result["reasons"]),
              "an extra base-census exclusion row was ignored")

    source = (ROOT / "tools/adjudicate_box_runs.py").read_text()
    for forbidden in ("17.8", "21.14", "18.635", "0.75", "samples20",
                      "DEFAULT_MIN_QUANTUM"):
        check(forbidden not in source, f"decision literal leaked into reader: {forbidden}")
    print("[box-run-adjudicator] PASS: dense converged/boundary/partial/no-recovery/VOID + "
          "WK1/default-command/B-ladder/lock/submodule/cap controls; GEMV 20-sample "
          "resolved-changed/unchanged/unresolved + raw bands/base+expanded census + "
          "unregistered-observation separation; policy/prose/timer seal")


if __name__ == "__main__":
    main()
