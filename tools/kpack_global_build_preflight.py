#!/usr/bin/env python3
"""Run and attest the repository-global checks used by K-pack shard builds.

The receipt deliberately covers only the five checks listed in ``CHECKERS``.
It is not a general build cache: build.sh continues to validate its target,
generated sources, CMake inputs, SDK, source revision and submodule checkout on
every invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "quactlize.kpack_global_build_preflight.v1"
TOOL_PATH = "tools/kpack_global_build_preflight.py"
CHECKERS = (
    "dev/fold_derivation/cmake_calls_check.sh",
    "dev/fold_derivation/ppu_portability_check.py",
    "dev/fold_derivation/gen_gemv_units_check.sh",
    "dev/fold_derivation/overlay_targets_check.py",
    "dev/fold_derivation/gen_moe_units_check.sh",
)
RELEVANT_PATHS = (
    ".gitmodules",
    "CMakeLists.txt",
    "build.sh",
    "benchmarks",
    "ci",
    "cmake",
    "dev",
    "quactlize",
    "tests",
    "third_party",
    "tools",
)
CANONICAL_ENVIRONMENT = {
    "BAD": "",
    "GEMV_GROUPS": "",
    "MOE_CHECK_CORES": "192",
    "MOE_FORMATS": "",
    "MOE_STAGES": "",
    "MOE_TM_LIST": "",
    "MOE_TN_LIST": "",
    "MOE_WM_LIST": "",
    "PPU_DEFS": "",
    "PPU_EXTRA_DEFS": "",
}


class PreflightError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    argv: list[str],
    *,
    cwd: Path,
    check: bool = True,
    text: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        stdout = result.stdout if text else result.stdout.decode(errors="replace")
        detail = "\n".join((stdout + stderr).splitlines()[-30:])
        raise PreflightError(f"command failed ({result.returncode}): {' '.join(argv)}\n{detail}")
    return result


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = _run(["git", "-C", str(root), *args], cwd=root, text=text)
    return result.stdout


def _require_plain_file(root: Path, relative: str, *, executable: bool = False) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise PreflightError(f"required tracked file is absent or not a plain file: {relative}")
    if executable and not os.access(path, os.X_OK):
        raise PreflightError(f"required checker is not executable: {relative}")
    return path


def _tracked_state(root: Path) -> dict[str, Any]:
    head = str(_git(root, "rev-parse", "HEAD")).strip()
    tree = str(_git(root, "rev-parse", "HEAD^{tree}")).strip()
    if (
        len(head) not in (40, 64)
        or len(tree) != len(head)
        or any(character not in "0123456789abcdef" for character in head + tree)
    ):
        raise PreflightError("repository does not expose one exact Git HEAD/tree")

    for args, description in (
        (("diff", "--quiet", "HEAD", "--", *RELEVANT_PATHS), "tracked relevant source is dirty"),
        (("diff", "--cached", "--quiet", "HEAD", "--", *RELEVANT_PATHS), "staged relevant source is dirty"),
    ):
        result = subprocess.run(["git", "-C", str(root), *args], cwd=root, check=False)
        if result.returncode != 0:
            raise PreflightError(description)

    untracked = str(
        _git(root, "ls-files", "--others", "--exclude-standard", "--", *RELEVANT_PATHS)
    ).splitlines()
    if untracked:
        raise PreflightError("untracked relevant source exists: " + ", ".join(untracked[:8]))

    manifest = bytes(
        _git(root, "ls-tree", "-rz", "--full-tree", "HEAD", "--", *RELEVANT_PATHS, text=False)
    )
    entries = [record for record in manifest.split(b"\0") if record]
    if not entries:
        raise PreflightError("tracked input manifest is empty")
    return {
        "head": head,
        "tree": tree,
        "relevant_paths": list(RELEVANT_PATHS),
        "tracked_input_count": len(entries),
        "tracked_input_manifest_sha256": sha256_bytes(manifest),
        "clean_relevant_worktree": True,
        "untracked_relevant_paths": [],
    }


def _submodule_state(root: Path) -> list[dict[str, Any]]:
    lines = str(_git(root, "submodule", "status", "--recursive")).splitlines()
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line or line[0] != " ":
            raise PreflightError(f"submodule absent, conflicted, or not at its recorded commit: {line}")
        fields = line[1:].split()
        if (
            len(fields) < 2
            or len(fields[0]) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in fields[0])
        ):
            raise PreflightError(f"cannot parse recursive submodule status: {line}")
        commit, relative = fields[0], fields[1]
        checkout = root / relative
        current = str(_git(checkout, "rev-parse", "HEAD")).strip()
        tree = str(_git(checkout, "rev-parse", "HEAD^{tree}")).strip()
        status = str(_git(checkout, "status", "--porcelain", "--untracked-files=all")).strip()
        if current != commit:
            raise PreflightError(f"submodule current commit differs from recorded status: {relative}")
        if status:
            raise PreflightError(f"recursive submodule worktree is dirty: {relative}")
        records.append(
            {
                "path": relative,
                "gitlink": commit,
                "current": current,
                "tree": tree,
                "clean": True,
            }
        )
    return records


def live_authority(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not (root / ".git").exists():
        raise PreflightError(f"not a git checkout: {root}")
    checker_records = []
    for relative in CHECKERS:
        path = _require_plain_file(root, relative, executable=True)
        checker_records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    tool = _require_plain_file(root, TOOL_PATH)
    return {
        "repository": _tracked_state(root),
        "submodules": _submodule_state(root),
        "receipt_tool": {
            "path": TOOL_PATH,
            "size": tool.stat().st_size,
            "sha256": sha256_file(tool),
        },
        "checkers": checker_records,
        "checker_environment": dict(CANONICAL_ENVIRONMENT),
        "artifact_root_policy": "PRIVATE_TEMPORARY_DIRECTORY",
    }


def _checker_environment(root: Path, scratch: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(CANONICAL_ENVIRONMENT)
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "QUACTLIZE_ARTIFACT_ROOT": str(scratch),
            "QUACTLIZE_ROOT": str(root),
            "QUACTLIZE_SRC_DIRS": "quactlize/include quactlize/csrc/device tests benchmarks dev",
            "QUACTLIZE_GEMV_DIR": str(root / "quactlize/include/gemv_lowbit"),
            "QUACTLIZE_CMAKE": str(root / "quactlize/csrc/CMakeLists.txt.in"),
            "QZ_INTERNAL_SWEEP_CMAKE_AUTHORITY": str(root / "quactlize/csrc"),
        }
    )
    return environment


def _execute_checks(root: Path, receipt_parent: Path) -> list[dict[str, Any]]:
    receipt_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".kpack-global-preflight.", dir=receipt_parent))
    try:
        environment = _checker_environment(root, work)
        results = []
        for relative in CHECKERS:
            path = root / relative
            result = subprocess.run(
                [str(path)],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = result.stdout
            if result.returncode:
                detail = output.decode(errors="replace")
                detail = "\n".join(detail.splitlines()[-50:])
                raise PreflightError(f"global checker failed ({result.returncode}): {relative}\n{detail}")
            results.append(
                {
                    "path": relative,
                    "checker_sha256": sha256_file(path),
                    "returncode": 0,
                    "output_bytes": len(output),
                    "output_sha256": sha256_bytes(output),
                }
            )
        return results
    finally:
        shutil.rmtree(work)


def _validate_receipt_document(document: Any, live: dict[str, Any]) -> None:
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "authority",
        "executions",
        "scope",
    }:
        raise PreflightError("preflight receipt has unknown or missing top-level fields")
    if document["schema"] != SCHEMA:
        raise PreflightError("preflight receipt schema differs")
    if document["scope"] != {
        "cached_checks": list(CHECKERS),
        "excluded_per_invocation_checks": [
            "TARGET_AND_GENERATED_DIRECTORY",
            "CMAKE_AND_BUILD_SOURCE",
            "SOURCE_REVISION_AND_WORKTREE",
            "RECURSIVE_SUBMODULE_STATE",
            "SDK_AND_TOOLCHAIN",
        ],
    }:
        raise PreflightError("preflight receipt scope differs")
    if document["authority"] != live:
        raise PreflightError("preflight receipt authority differs from live repository state")
    executions = document["executions"]
    if not isinstance(executions, list) or len(executions) != len(CHECKERS):
        raise PreflightError("preflight receipt execution denominator differs")
    live_checkers = {record["path"]: record for record in live["checkers"]}
    for expected_path, record in zip(CHECKERS, executions):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "checker_sha256",
            "returncode",
            "output_bytes",
            "output_sha256",
        }:
            raise PreflightError("preflight receipt has malformed execution record")
        if record["path"] != expected_path:
            raise PreflightError("preflight receipt checker order/denominator differs")
        if record["checker_sha256"] != live_checkers[expected_path]["sha256"]:
            raise PreflightError(f"preflight checker hash differs: {expected_path}")
        if record["returncode"] != 0:
            raise PreflightError(f"preflight checker did not pass: {expected_path}")
        if not isinstance(record["output_bytes"], int) or record["output_bytes"] < 0:
            raise PreflightError("preflight checker output length is invalid")
        digest = record["output_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise PreflightError("preflight checker output hash is invalid")


def _read_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightError("preflight receipt is absent or not a plain file")
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PreflightError("preflight receipt is writable rather than immutable")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot decode preflight receipt: {error}") from error
    if not isinstance(document, dict):
        raise PreflightError("preflight receipt is not a JSON object")
    return document


def verify_receipt(root: Path, receipt: Path) -> dict[str, Any]:
    document = _read_receipt(receipt)
    _validate_receipt_document(document, live_authority(root))
    return document


def _publish_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PreflightError("refusing to replace an existing preflight receipt")
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PreflightError("preflight receipt staging path already exists")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_receipt(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise PreflightError("refusing to replace an existing preflight receipt")
    authority = live_authority(root)
    executions = _execute_checks(root, output.parent)
    # Re-read after execution: a checker or concurrent editor may have changed
    # its inputs while the five commands ran. Such a run has no single source
    # authority and must never be published.
    if live_authority(root) != authority:
        raise PreflightError("repository authority changed while global checks ran")
    document = {
        "schema": SCHEMA,
        "authority": authority,
        "executions": executions,
        "scope": {
            "cached_checks": list(CHECKERS),
            "excluded_per_invocation_checks": [
                "TARGET_AND_GENERATED_DIRECTORY",
                "CMAKE_AND_BUILD_SOURCE",
                "SOURCE_REVISION_AND_WORKTREE",
                "RECURSIVE_SUBMODULE_STATE",
                "SDK_AND_TOOLCHAIN",
            ],
        },
    }
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    _publish_immutable(output, encoded)
    verify_receipt(root, output)
    return document


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kpack-global-preflight-test.") as directory:
        base = Path(directory)
        root = base / "repo"
        root.mkdir()
        _run(["git", "init", "-q", "-b", "main"], cwd=root)
        _run(["git", "config", "user.email", "preflight@example.invalid"], cwd=root)
        _run(["git", "config", "user.name", "Preflight Test"], cwd=root)
        _write(root / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.18)\n")
        _write(root / "build.sh", "#!/bin/sh\nexit 0\n", executable=True)
        _write(root / "quactlize/csrc/CMakeLists.txt.in", "# authority\n")
        _write(root / "quactlize/include/gemv_lowbit/anchor.hpp", "// anchor\n")
        _write(root / "tools/kpack_global_build_preflight.py", Path(__file__).read_text())
        for index, relative in enumerate(CHECKERS):
            suffix = "python3\nprint('PASS')\n" if relative.endswith(".py") else "sh\nprintf 'PASS\\n'\n"
            _write(root / relative, "#!/usr/bin/env " + suffix, executable=True)

        subsource = base / "subsource"
        subsource.mkdir()
        _run(["git", "init", "-q", "-b", "main"], cwd=subsource)
        _run(["git", "config", "user.email", "preflight@example.invalid"], cwd=subsource)
        _run(["git", "config", "user.name", "Preflight Test"], cwd=subsource)
        _write(subsource / "anchor.txt", "submodule\n")
        _run(["git", "add", "."], cwd=subsource)
        _run(["git", "commit", "-qm", "submodule"], cwd=subsource)
        _run(
            [
                "git",
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "-q",
                str(subsource),
                "third_party/actlize",
            ],
            cwd=root,
        )
        _run(["git", "add", "."], cwd=root)
        _run(["git", "commit", "-qm", "fixture"], cwd=root)

        receipt = base / "receipt.json"
        create_receipt(root, receipt)
        verify_receipt(root, receipt)
        if receipt.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise AssertionError("published receipt is writable")
        try:
            create_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("existing receipt was replaced")

        receipt.chmod(stat.S_IRUSR | stat.S_IWUSR)
        try:
            verify_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("writable receipt was accepted")
        receipt.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        original_receipt = receipt.read_text()
        tampered = json.loads(original_receipt)
        tampered["executions"][0]["returncode"] = 1
        receipt.write_text(json.dumps(tampered))
        receipt.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            verify_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("tampered checker result was accepted")
        receipt.write_text(original_receipt)
        receipt.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        authority_path = root / "quactlize/csrc/CMakeLists.txt.in"
        authority_path.write_text("# dirty\n")
        try:
            verify_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("dirty tracked input was accepted")
        authority_path.write_text("# authority\n")

        rogue = root / "tools/untracked.py"
        rogue.write_text("pass\n")
        try:
            verify_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("untracked relevant input was accepted")
        rogue.unlink()

        subanchor = root / "third_party/actlize/anchor.txt"
        subanchor.write_text("dirty\n")
        try:
            verify_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("dirty recursive submodule was accepted")
        subanchor.write_text("submodule\n")

        checker = root / CHECKERS[0]
        checker.write_text("#!/bin/sh\nexit 9\n")
        checker.chmod(checker.stat().st_mode | stat.S_IXUSR)
        _run(["git", "add", CHECKERS[0]], cwd=root)
        _run(["git", "commit", "-qm", "failing checker"], cwd=root)
        try:
            verify_receipt(root, receipt)
        except PreflightError:
            pass
        else:
            raise AssertionError("receipt from stale HEAD/tree was accepted")
        try:
            create_receipt(root, base / "failed.json")
        except PreflightError:
            pass
        else:
            raise AssertionError("failed checker produced a receipt")
        print("[kpack-global-preflight:self-test] PASS immutable/source/untracked/submodule/checker plants RED")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", required=True, type=Path)
        option = "--output" if command == "create" else "--receipt"
        child.add_argument(option, required=True, type=Path)
    subparsers.add_parser("self-test")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            create_receipt(arguments.root, arguments.output)
            print(f"[kpack-global-preflight] CREATED receipt={arguments.output}")
        elif arguments.command == "verify":
            verify_receipt(arguments.root, arguments.receipt)
            print(f"[kpack-global-preflight] VERIFIED receipt={arguments.receipt}")
        else:
            self_test()
        return 0
    except (OSError, PreflightError, subprocess.SubprocessError) as error:
        print(f"[kpack-global-preflight] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
