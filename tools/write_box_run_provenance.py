#!/usr/bin/env python3
"""Write fail-closed, machine-readable provenance for box runners.

The shell runners use ``record`` after every material child command and
``write`` exactly once when the runner terminates.  Keeping JSON quoting here
avoids turning shell-escaped diagnostic text into an accidental second schema.
Device identity is deliberately supplied by explicit runner environment; this
tool never guesses which accelerator an OS-level probe happened to enumerate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SHA_RE = re.compile(r"[0-9a-f]{40,64}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
POLICY_BEGIN = "<!-- BOX_RUN_POLICY_V1_BEGIN -->"
POLICY_END = "<!-- BOX_RUN_POLICY_V1_END -->"
IDENTITY_SCHEMA = "quactlize-box-run-identity-v1"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"box provenance: {message}")


def nonempty(value: str, name: str) -> str:
    if not value or not value.strip():
        fail(f"{name} must be non-empty")
    return value


def known_identity(value: str, name: str) -> str:
    value = nonempty(value, name)
    if value.strip().lower() in {"unknown", "unset", "n/a", "na", "none"}:
        fail(f"{name} must be measured explicit identity, not {value!r}")
    if "\n" in value or "\r" in value or "\0" in value:
        fail(f"{name} must be a single-line identity")
    return value


def record(args: argparse.Namespace) -> int:
    argv = list(args.command)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        fail("record requires a command after --")
    entry = {
        "role": nonempty(args.role, "role"),
        "argv": argv,
        "exit_status": args.exit_status,
    }
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


def read_commands(path: Path) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(f"cannot read command journal {path}: {exc}")
    for lineno, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"malformed command journal line {lineno}: {exc}")
        if (not isinstance(item, dict) or
                not isinstance(item.get("role"), str) or not item["role"] or
                not isinstance(item.get("argv"), list) or not item["argv"] or
                not all(isinstance(word, str) for word in item["argv"]) or
                isinstance(item.get("exit_status"), bool) or
                not isinstance(item.get("exit_status"), int)):
            fail(f"malformed command journal entry at line {lineno}")
        commands.append(item)
    if not commands:
        fail("command journal is empty")
    return commands


def read_submodule_status(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as exc:
        fail(f"cannot read submodule status: {exc}")
    nonempty(value, "submodule_status")
    if any(line[:1] in {"+", "-", "U"} for line in value.splitlines()):
        fail("submodule_status contains a non-gitlink checkout")
    return value


def identity_payload(args: argparse.Namespace) -> dict[str, object]:
    root_sha = nonempty(args.root_sha, "root_sha")
    binary_sha = nonempty(args.binary_sha256, "binary_sha256")
    if not SHA_RE.fullmatch(root_sha):
        fail("root_sha is not a full lowercase Git object ID")
    if not SHA_RE.fullmatch(args.actlize_sha):
        fail("actlize_sha is not a full lowercase Git object ID")
    if not SHA256_RE.fullmatch(binary_sha):
        fail("binary_sha256 is not a lowercase SHA-256")
    if args.protocol_sample_count <= 0:
        fail("protocol_sample_count must be positive")
    return {
        "schema": IDENTITY_SCHEMA,
        "root_sha": root_sha,
        "submodule_status": read_submodule_status(Path(args.submodule_status_file)),
        "actlize_sha": args.actlize_sha,
        "binary_sha256": binary_sha,
        "device_model": known_identity(args.device_model, "device_model"),
        "pci_identity": known_identity(args.pci_identity, "pci_identity"),
        "driver_version": known_identity(args.driver_version, "driver_version"),
        "sdk_compiler_identity": known_identity(
            args.sdk_compiler_identity, "sdk_compiler_identity"),
        "protocol_sample_count": args.protocol_sample_count,
    }


def identity_digest(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def write_identity(args: argparse.Namespace) -> int:
    value = identity_payload(args)
    value["identity_sha256"] = identity_digest(value)
    atomic_json_write(Path(args.output), value)
    print(value["identity_sha256"])
    return 0


def read_identity(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read run identity {path}: {exc}")
    expected_keys = {
        "schema", "root_sha", "submodule_status", "actlize_sha",
        "binary_sha256", "device_model", "pci_identity", "driver_version",
        "sdk_compiler_identity", "protocol_sample_count", "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        fail(f"run identity {path} does not have the exact v1 schema")
    digest = value.pop("identity_sha256")
    if value.get("schema") != IDENTITY_SCHEMA or not isinstance(digest, str):
        fail(f"run identity {path} has an invalid schema or digest")
    if identity_digest(value) != digest:
        fail(f"run identity {path} digest does not match its contents")
    value["identity_sha256"] = digest
    return value


def verify_identity(args: argparse.Namespace) -> int:
    expected = read_identity(Path(args.expected))
    candidate = read_identity(Path(args.candidate))
    if expected != candidate:
        differing = sorted(key for key in expected if expected[key] != candidate[key])
        fail("immutable run identity differs in: " + ", ".join(differing))
    print(expected["identity_sha256"])
    return 0


def write(args: argparse.Namespace) -> int:
    root_sha = nonempty(args.root_sha, "root_sha")
    binary_sha = nonempty(args.binary_sha256, "binary_sha256")
    if not SHA_RE.fullmatch(root_sha):
        fail("root_sha is not a full lowercase Git object ID")
    if not SHA_RE.fullmatch(args.actlize_sha):
        fail("actlize_sha is not a full lowercase Git object ID")
    if not SHA256_RE.fullmatch(binary_sha):
        fail("binary_sha256 is not a lowercase SHA-256")
    if args.root_status != "clean":
        fail("root_status must be exactly 'clean'")

    submodule_status = read_submodule_status(Path(args.submodule_status_file))

    runner_argv = list(args.runner_argv)
    if runner_argv[:1] == ["--"]:
        runner_argv = runner_argv[1:]
    if not runner_argv:
        fail("runner_argv must be non-empty")

    value: dict[str, object] = {
        "schema": "quactlize-box-run-provenance-v1",
        "root_sha": root_sha,
        "root_status": "clean",
        "submodule_status": submodule_status,
        "actlize_sha": args.actlize_sha,
        "binary_sha256": binary_sha,
        "device_model": known_identity(args.device_model, "device_model"),
        "pci_identity": known_identity(args.pci_identity, "pci_identity"),
        "driver_version": known_identity(args.driver_version, "driver_version"),
        "sdk_compiler_identity": known_identity(
            args.sdk_compiler_identity, "sdk_compiler_identity"),
        # ``argv`` stays the exact top-level runner argv consumed by the v1
        # adjudicator.  ``commands`` closes the otherwise-unrepresentable
        # multi-command build/run/analyse provenance.
        "argv": runner_argv,
        "commands": read_commands(Path(args.commands_file)),
        "runner_exit_status": args.runner_exit_status,
    }
    if args.protocol_sample_count <= 0:
        fail("protocol_sample_count must be positive")
    value["protocol_sample_count"] = args.protocol_sample_count

    identity = read_identity(Path(args.run_identity_file))
    provenance_identity = {
        "schema": IDENTITY_SCHEMA,
        "root_sha": root_sha,
        "submodule_status": submodule_status,
        "actlize_sha": args.actlize_sha,
        "binary_sha256": binary_sha,
        "device_model": value["device_model"],
        "pci_identity": value["pci_identity"],
        "driver_version": value["driver_version"],
        "sdk_compiler_identity": value["sdk_compiler_identity"],
        "protocol_sample_count": args.protocol_sample_count,
    }
    provenance_identity["identity_sha256"] = identity_digest(provenance_identity)
    if provenance_identity != identity:
        differing = sorted(key for key in identity
                           if identity[key] != provenance_identity[key])
        fail("provenance does not match immutable run identity in: " +
             ", ".join(differing))

    atomic_json_write(Path(args.output), value)
    return 0


def policy_sample_count(args: argparse.Namespace) -> int:
    try:
        text = Path(args.policy).read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read preregistration policy: {exc}")
    if text.count(POLICY_BEGIN) != 1 or text.count(POLICY_END) != 1:
        fail("preregistration must contain exactly one policy block")
    block = text.split(POLICY_BEGIN, 1)[1].split(POLICY_END, 1)[0].strip()
    try:
        policy = json.loads(block)
        count = policy[args.kind]["sample_count"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        fail(f"cannot read {args.kind}.sample_count from policy: {exc}")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        fail(f"{args.kind}.sample_count is not a positive integer")
    print(count)
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="action", required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--path", required=True)
    rec.add_argument("--role", required=True)
    rec.add_argument("--exit-status", type=int, required=True)
    rec.add_argument("command", nargs=argparse.REMAINDER)
    rec.set_defaults(func=record)

    def add_identity_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--root-sha", required=True)
        target.add_argument("--submodule-status-file", required=True)
        target.add_argument("--actlize-sha", required=True)
        target.add_argument("--binary-sha256", required=True)
        target.add_argument("--device-model", required=True)
        target.add_argument("--pci-identity", required=True)
        target.add_argument("--driver-version", required=True)
        target.add_argument("--sdk-compiler-identity", required=True)
        target.add_argument("--protocol-sample-count", type=int, required=True)

    identity = sub.add_parser("write-identity")
    identity.add_argument("--output", required=True)
    add_identity_arguments(identity)
    identity.set_defaults(func=write_identity)

    verify = sub.add_parser("verify-identity")
    verify.add_argument("--expected", required=True)
    verify.add_argument("--candidate", required=True)
    verify.set_defaults(func=verify_identity)

    out = sub.add_parser("write")
    out.add_argument("--output", required=True)
    out.add_argument("--root-sha", required=True)
    out.add_argument("--root-status", required=True)
    out.add_argument("--submodule-status-file", required=True)
    out.add_argument("--actlize-sha", required=True)
    out.add_argument("--binary-sha256", required=True)
    out.add_argument("--device-model", required=True)
    out.add_argument("--pci-identity", required=True)
    out.add_argument("--driver-version", required=True)
    out.add_argument("--sdk-compiler-identity", required=True)
    out.add_argument("--run-identity-file", required=True)
    out.add_argument("--commands-file", required=True)
    out.add_argument("--runner-exit-status", type=int, required=True)
    out.add_argument("--protocol-sample-count", type=int, required=True)
    out.add_argument("runner_argv", nargs=argparse.REMAINDER)
    out.set_defaults(func=write)

    query = sub.add_parser("policy-sample-count")
    query.add_argument("--policy", required=True)
    query.add_argument("--kind", choices=("dense", "gemv"), required=True)
    query.set_defaults(func=policy_sample_count)
    return top


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
