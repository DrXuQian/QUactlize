#!/usr/bin/env python3
"""Build a source-bound, execution-only payload for the L248 PPU probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence

import run_l248_q4_n16k64_prebuilt as gate


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_ROOT = pathlib.Path("/root/autodl-tmp")
L248_BUILD_RUNNER = "dev/fold_derivation/run_l248_q4_n16k64_delivery_rawbit.sh"
L248_SOURCE = "dev/test_q4_n16k64_delivery_rawbit.cu"
SOURCE_INPUTS = (
    "build.sh",
    "quactlize/csrc/CMakeLists.txt.in",
    L248_BUILD_RUNNER,
    L248_SOURCE,
    "dev/fold_derivation/l249_q4_n16k64_multim_warp_layout.cu",
    "dev/fold_derivation/l249_q4_n16k64_multim_warp_layout.expected.txt",
    "quactlize/include/q4_n16k64_direct_offline.hpp",
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "quactlize_b_delivery_policy.hpp",
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "quactlize_b_s2r_adapter.hpp",
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "quactlize_q4_n16k64_delivery.hpp",
)
SUBMODULES = ("third_party/actlize", "third_party/cutlass")
EVIDENCE_FILES = (
    "build.log",
    "build-verdict.log",
    "handoff.env",
    "isa.txt",
    "list-elf.txt",
)


class BuildError(RuntimeError):
    """The source-bound prebuilt payload could not be produced."""


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(
        command: Sequence[str], *, cwd: pathlib.Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: int = 600, binary: bool = False,
        log: pathlib.Path | None = None):
    try:
        result = subprocess.run(
            list(command), cwd=cwd, env=environment, check=False,
            capture_output=True, text=not binary, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError(f"command failed to execute: {shlex.join(command)}: {exc}") from exc
    if log is not None:
        if binary:
            raise BuildError("internal error: binary command cannot be logged as text")
        log.write_text(
            f"command={shlex.join(command)}\n"
            f"returncode={result.returncode}\n"
            "stdout-begin\n" + result.stdout +
            ("" if result.stdout.endswith("\n") else "\n") +
            "stdout-end\n"
            "stderr-begin\n" + result.stderr +
            ("" if result.stderr.endswith("\n") or not result.stderr else "\n") +
            "stderr-end\n",
            encoding="utf-8")
    if result.returncode != 0:
        stderr = result.stderr if not binary else result.stderr.decode(
            "utf-8", errors="replace")
        raise BuildError(
            f"command failed rc={result.returncode}: {shlex.join(command)}\n"
            f"{stderr[-4000:]}")
    return result.stdout


def _git(arguments: Sequence[str], *, repository: pathlib.Path = ROOT,
         binary: bool = False):
    return _run(["git", "-C", str(repository), *arguments], binary=binary)


def _resolve_commit(value: str | None) -> str:
    requested = value or "HEAD"
    commit = str(_git(["rev-parse", f"{requested}^{{commit}}"])).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BuildError(f"source commit is malformed: {commit!r}")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
         gate.REQUIRED_ANCESTOR, commit],
        check=False, capture_output=True, text=True)
    if ancestor.returncode != 0:
        raise BuildError(
            f"source commit {commit} is not descended from committed L249 "
            f"{gate.REQUIRED_ANCESTOR}")
    return commit


def _safe_output(path: pathlib.Path) -> pathlib.Path:
    if not path.is_absolute():
        raise BuildError("--output must be absolute")
    allowed = ALLOWED_OUTPUT_ROOT.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as exc:
        raise BuildError(
            f"all build output must stay below {allowed}: {resolved}") from exc
    if not relative.parts:
        raise BuildError(f"refusing to use the whole output root: {allowed}")
    if path.exists() or path.is_symlink():
        raise BuildError(f"refusing to overwrite prebuilt output: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise BuildError(f"prebuilt output parent is missing or symlinked: {path.parent}")
    path.mkdir(mode=0o755)
    return path


def _regular(path: pathlib.Path, label: str) -> pathlib.Path:
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _sdk_identity(sdk_root: pathlib.Path, archive: pathlib.Path):
    if sdk_root.is_symlink() or not sdk_root.is_dir():
        raise BuildError(f"PPU SDK must be a regular directory: {sdk_root}")
    sdk_root = sdk_root.resolve()
    archive = _regular(archive, "PPU SDK archive")
    archive_digest = _sha256(archive)
    if archive_digest != gate.SDK_ARCHIVE_SHA256:
        raise BuildError(
            f"PPU SDK archive digest is not admitted: {archive_digest}")
    receipt = _regular(sdk_root / "release.yaml", "PPU SDK release receipt")
    versions = [line.split(":", 1)[1].strip()
                for line in receipt.read_text(encoding="utf-8").splitlines()
                if line.startswith("version:")]
    if versions != [gate.SDK_RELEASE]:
        raise BuildError(f"PPU SDK release is not admitted: {versions!r}")
    sdk_files = {
        relative: _regular(sdk_root.joinpath(*relative.split("/")), relative)
        for relative in ("bin/hgcc", "bin/hgobjdump", "lib/libhggc_wrapper.so")
    }
    if _sha256(sdk_files["lib/libhggc_wrapper.so"]) != gate.RUNTIME_SHA256:
        raise BuildError("PPU SDK runtime digest is not admitted")
    compiler_identity = str(_run(
        [str(sdk_files["bin/hgcc"]), "--version"], timeout=30)).replace(
            "\n", " ").strip()
    if gate.SDK_RELEASE not in compiler_identity or "stub" in compiler_identity.lower():
        raise BuildError(f"hgcc identity is not admitted: {compiler_identity!r}")
    return sdk_root, archive, compiler_identity, {
        relative: _sha256(path) for relative, path in sdk_files.items()
    }


def _gitlink(commit: str, path: str) -> str:
    listing = str(_git(["ls-tree", commit, "--", path])).strip()
    match = re.fullmatch(r"160000 commit ([0-9a-f]{40})\t(.+)", listing)
    if not match or match.group(2) != path:
        raise BuildError(f"source commit has no exact gitlink for {path}: {listing!r}")
    return match.group(1)


def _source_authority(commit: str):
    tree = str(_git(["rev-parse", f"{commit}^{{tree}}"])).strip()
    submodules = []
    for path in SUBMODULES:
        sub_commit = _gitlink(commit, path)
        local = ROOT / path
        _git(["cat-file", "-e", f"{sub_commit}^{{commit}}"], repository=local)
        sub_tree = str(_git(
            ["rev-parse", f"{sub_commit}^{{tree}}"], repository=local)).strip()
        submodules.append({"path": path, "commit": sub_commit, "tree": sub_tree})
    inputs = []
    for path in SOURCE_INPUTS:
        blob = str(_git(["rev-parse", f"{commit}:{path}"])).strip()
        contents = _git(["show", f"{commit}:{path}"], binary=True)
        inputs.append({"path": path, "blob": blob, "sha256": _bytes_sha256(contents)})
    return {
        "commit": commit,
        "tree": tree,
        "required_ancestor": gate.REQUIRED_ANCESTOR,
        "submodules": submodules,
        "inputs": inputs,
    }


def _clean_source_clone(output: pathlib.Path, commit: str) -> pathlib.Path:
    source = output / "work" / "source"
    source.parent.mkdir(parents=True)
    _run(["git", "clone", "--shared", "--no-checkout", str(ROOT), str(source)],
         timeout=180, log=output / "clone.log")
    _run(["git", "-C", str(source), "checkout", "--detach", commit], timeout=120)
    for path in SUBMODULES:
        _run([
            "git", "-C", str(source), "config", f"submodule.{path}.url",
            str(ROOT / path)])
    _run([
        "git", "-c", "protocol.file.allow=always", "-C", str(source),
        "submodule", "update", "--init", "--recursive"],
        timeout=300, log=output / "submodule.log")
    status = str(_run([
        "git", "-C", str(source), "status", "--porcelain=v1",
        "--untracked-files=all"])).strip()
    if status:
        raise BuildError(f"clean source clone is unexpectedly dirty:\n{status}")
    return source


def _parse_handoff(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise BuildError(f"malformed L248 handoff line: {line!r}")
        key, value = line.split("=", 1)
        if key in result or not key or not value:
            raise BuildError(f"malformed or duplicate L248 handoff key: {key!r}")
        result[key] = value
    required = {
        "source_sha", "submodule.third_party/actlize",
        "submodule.third_party/cutlass", "sdk", "sdk_release",
        "sdk_archive_sha256", "compiler", "arch", "target", "ppu_defs",
        "binary", "binary_size", "binary_sha256",
    }
    if set(result) != required:
        raise BuildError(
            f"L248 handoff fields differ: got={sorted(result)} expected={sorted(required)}")
    return result


def _isa_contract(list_elf: pathlib.Path, isa: pathlib.Path):
    listed = list_elf.read_text(encoding="utf-8")
    disassembly = isa.read_text(encoding="utf-8")
    symbols = re.findall(r"^Func [0-9]+: (\S+)$", listed, flags=re.MULTILINE)
    if len(symbols) != 1 or gate.KERNEL_SYMBOL_TOKEN not in symbols[0]:
        raise BuildError(f"L248 ELF kernel authority differs: {symbols!r}")
    symbol = symbols[0]
    if f"Disassembly of section .text.kernel.{symbol}" not in disassembly:
        raise BuildError("L248 ISA is not bound to the listed kernel symbol")
    counts = {
        "aiu_plain_b32": disassembly.count(
            "vmem.aiu.ld.tsm.l1.t0.p0.s0.m0.2d.b32.kp1"),
        "aiu_all": disassembly.count("vmem.aiu.ld.tsm"),
        "universal_tsm_b32x4": disassembly.count("tsm.ld.b32x4"),
        "tsm_load_all": len(re.findall(r"\btsm\.ld\.", disassembly)),
        "commit": int("vmem.acp.commit.grp" in disassembly),
        "wait": int("commit_group(0)" in disassembly),
        "barrier": int("s.blksyn" in disassembly),
        "swizzle": int(bool(re.search(
            r"tsm\.ld\.swzl|vmem\.aiu\.ld\.tsm\.[^\s]*\.s1\.",
            disassembly))),
    }
    expected = {
        "aiu_plain_b32": 4, "aiu_all": 4,
        "universal_tsm_b32x4": 16, "tsm_load_all": 16,
        "commit": 1, "wait": 1, "barrier": 1, "swizzle": 0,
    }
    if counts != expected:
        raise BuildError(f"L248 ISA contract differs: got={counts} expected={expected}")
    return symbol, counts


def _entry(path: pathlib.Path, relative: str) -> dict[str, object]:
    return {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package(
        output: pathlib.Path, source: pathlib.Path, l248: pathlib.Path,
        source_authority: dict[str, object], compiler_identity: str,
        sdk_files: dict[str, str]):
    payload = output / "payload"
    binary_dir = payload / "bin"
    evidence_dir = payload / "evidence"
    binary_dir.mkdir(parents=True)
    evidence_dir.mkdir()
    handoff = _parse_handoff(l248 / "handoff.env")
    commit = str(source_authority["commit"])
    if handoff["source_sha"] != commit:
        raise BuildError("L248 handoff source SHA differs from clean source clone")
    submodule_map = {
        str(item["path"]): str(item["commit"])
        for item in source_authority["submodules"]
    }
    for path in SUBMODULES:
        if handoff[f"submodule.{path}"] != submodule_map[path]:
            raise BuildError(f"L248 handoff submodule differs for {path}")
    if handoff["sdk_release"] != gate.SDK_RELEASE or \
            handoff["sdk_archive_sha256"] != gate.SDK_ARCHIVE_SHA256 or \
            handoff["arch"] != gate.ARCH or handoff["target"] != gate.TARGET or \
            handoff["ppu_defs"].split() != gate.PPU_DEFS:
        raise BuildError("L248 handoff build identity differs")

    built_binary = pathlib.Path(handoff["binary"])
    if built_binary.is_symlink() or not built_binary.is_file():
        raise BuildError(f"L248 built binary is missing or symlinked: {built_binary}")
    packaged_binary = binary_dir / gate.TARGET
    shutil.copy2(built_binary, packaged_binary)
    packaged_binary.chmod(0o755)
    if (str(packaged_binary.stat().st_size) != handoff["binary_size"] or
            _sha256(packaged_binary) != handoff["binary_sha256"]):
        raise BuildError("packaged L248 binary differs from its build handoff")

    evidence = []
    for name in EVIDENCE_FILES:
        source_path = l248 / name
        if source_path.is_symlink() or not source_path.is_file():
            raise BuildError(f"L248 evidence is missing or symlinked: {name}")
        destination = evidence_dir / name
        shutil.copy2(source_path, destination)
        evidence.append(_entry(destination, f"evidence/{name}"))
    committed_l249 = source / "dev/fold_derivation/l249_q4_n16k64_multim_warp_layout.expected.txt"
    l249_copy = evidence_dir / "l249.expected.txt"
    shutil.copy2(committed_l249, l249_copy)
    evidence.append(_entry(l249_copy, "evidence/l249.expected.txt"))

    symbol, isa_counts = _isa_contract(
        evidence_dir / "list-elf.txt", evidence_dir / "isa.txt")
    runner_source = ROOT / "tools" / gate.RUNNER_PATH
    if runner_source.is_symlink() or not runner_source.is_file():
        raise BuildError(f"prebuilt execution runner is missing: {runner_source}")
    packaged_runner = payload / gate.RUNNER_PATH
    shutil.copy2(runner_source, packaged_runner)
    packaged_runner.chmod(0o755)

    manifest = {
        "schema": gate.SCHEMA,
        "schema_version": gate.SCHEMA_VERSION,
        "source": source_authority,
        "sdk": {
            "release": gate.SDK_RELEASE,
            "archive_sha256": gate.SDK_ARCHIVE_SHA256,
            "compiler_identity": compiler_identity,
            "files": sdk_files,
        },
        "target": {
            "name": gate.TARGET,
            "arch": gate.ARCH,
            "ppu_defs": gate.PPU_DEFS,
            "layout": gate.LAYOUT,
            "mapping_id": gate.MAPPING_ID,
            "words": gate.WORDS,
            "stage_bytes": gate.STAGE_BYTES,
        },
        "artifact": {
            "path": gate.ARTIFACT_PATH,
            "size": packaged_binary.stat().st_size,
            "sha256": _sha256(packaged_binary),
            "kernel_count": 1,
            "symbol": symbol,
            "isa_counts": isa_counts,
        },
        "evidence": sorted(evidence, key=lambda item: str(item["path"])),
        "runner": _entry(packaged_runner, gate.RUNNER_PATH),
    }
    manifest_path = payload / "manifest.json"
    _write_json(manifest_path, manifest)
    authority_paths = [
        manifest_path, packaged_runner, packaged_binary,
        *(payload / str(item["path"]) for item in manifest["evidence"]),
    ]
    authority_lines = []
    for path in sorted(authority_paths, key=lambda item: item.relative_to(payload).as_posix()):
        relative = path.relative_to(payload).as_posix()
        authority_lines.append(f"{_sha256(path)}  {relative}\n")
    (payload / gate.AUTHORITY_PATH).write_text(
        "".join(authority_lines), encoding="utf-8")
    return payload, manifest


def _run_payload_self_tests(
        output: pathlib.Path, payload: pathlib.Path, sdk_root: pathlib.Path,
        source: pathlib.Path, manifest: dict[str, object]):
    environment = dict(os.environ)
    temporary = output / "work" / "tmp"
    temporary.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    runner = payload / gate.RUNNER_PATH
    self_test = _run(
        [sys.executable, "-B", str(runner), "--self-test"],
        environment=environment, timeout=120)
    (output / "self-test.log").write_text(str(self_test), encoding="utf-8")
    verify = _run([
        sys.executable, "-B", str(runner), str(payload),
        "--ppu-sdk", str(sdk_root), "--source-tree", str(source),
        "--expect-source-sha", str(manifest["source"]["commit"]),
        "--expect-binary-sha256", str(manifest["artifact"]["sha256"]),
        "--expect-manifest-sha256", _sha256(payload / "manifest.json"),
        "--verify-only"], environment=environment, timeout=180)
    (output / "verify-only.log").write_text(str(verify), encoding="utf-8")
    if "L248_Q4_N16K64_PREBUILT_SELF_TEST PASS" not in str(self_test):
        raise BuildError("prebuilt runner self-test did not emit its PASS marker")
    if "L248_Q4_N16K64_PREBUILT_VERIFY PASS" not in str(verify):
        raise BuildError("prebuilt payload verify-only did not emit its PASS marker")


def _archive(output: pathlib.Path, payload: pathlib.Path, commit: str):
    archive = output / f"l248-q4-n16k64-prebuilt-{commit[:12]}.tar.gz"
    _run([
        "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
        "--numeric-owner", "-C", str(output), "-czf", str(archive),
        payload.name], timeout=180, log=output / "archive.log")
    return archive


def build(args) -> dict[str, object]:
    output = _safe_output(args.output)
    source_commit = _resolve_commit(args.source_sha)
    sdk_root, archive, compiler_identity, sdk_files = _sdk_identity(
        args.ppu_sdk, args.sdk_archive)
    source_authority = _source_authority(source_commit)
    source = _clean_source_clone(output, source_commit)

    l248 = output / "work" / "l248"
    temporary = output / "work" / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "PPU_SDK": str(sdk_root),
        "PPU_SDK_ARCHIVE": str(archive),
        "QUACTLIZE_L248_OUT": str(l248),
        "QUACTLIZE_L248_BUILD_ONLY": "1",
        "JOBS": str(args.jobs),
        "TMPDIR": str(temporary),
    })
    print(
        f"[l248-prebuilt-build] source={source_commit} sdk={sdk_root} "
        f"output={output}", flush=True)
    build_stdout = _run(
        ["bash", L248_BUILD_RUNNER], cwd=source, environment=environment,
        timeout=args.build_timeout_seconds, log=output / "l248-driver.log")
    print(str(build_stdout), end="" if str(build_stdout).endswith("\n") else "\n",
          flush=True)
    if "L248_Q4_N16K64_DELIVERY_BUILD PASS" not in str(build_stdout):
        raise BuildError("L248 build-only runner did not emit its PASS marker")

    payload, manifest = _package(
        output, source, l248, source_authority, compiler_identity, sdk_files)
    _run_payload_self_tests(output, payload, sdk_root, source, manifest)
    archive_path = _archive(output, payload, source_commit)
    result = {
        "schema": "quactlize.l248-q4-n16k64-prebuilt-build-result",
        "schema_version": 1,
        "status": "PASS",
        "source_sha": source_commit,
        "source_tree": source_authority["tree"],
        "payload": str(payload),
        "payload_manifest_sha256": _sha256(payload / "manifest.json"),
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "binary_sha256": manifest["artifact"]["sha256"],
        "self_test": "PASS",
        "verify_only": "PASS",
        "fresh_device_execution": 0,
    }
    _write_json(output / "BUILD_RESULT.json", result)
    print(
        "L248_Q4_N16K64_PREBUILT_BUILD PASS "
        f"source_sha={source_commit} "
        f"binary_sha256={result['binary_sha256']} "
        f"archive_sha256={result['archive_sha256']} "
        f"payload={payload} archive={archive_path} "
        "self_test=PASS verify_only=PASS fresh_device_execution=0",
        flush=True)
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value!r}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value!r}")
    return result


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppu-sdk", required=True, type=pathlib.Path)
    parser.add_argument("--sdk-archive", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--source-sha", required=True,
        help="exact committed A03 source authority; never defaults to moving HEAD")
    parser.add_argument("--jobs", type=_positive_int, default=1)
    parser.add_argument("--build-timeout-seconds", type=_positive_int, default=7200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        build(parse_args(argv))
    except BuildError as exc:
        print(f"[l248-prebuilt-build] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
