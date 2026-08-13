#!/usr/bin/env python3
"""Write fail-closed, machine-readable provenance for box runners.

The shell runners use ``record`` after every material child command and
``write`` exactly once when the runner terminates.  Keeping JSON quoting here
avoids turning shell-escaped diagnostic text into an accidental second schema.
Device identity is supplied together with the probe artifact that measured it
or recorded the operator fallback.  The values, their per-field evidence
sources, and the canonical probe digest are one immutable identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


TOOLS_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import box_identity_schema as identity_schema


SHA_RE = re.compile(r"[0-9a-f]{40,64}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
POLICY_BEGIN = "<!-- BOX_RUN_POLICY_V1_BEGIN -->"
POLICY_END = "<!-- BOX_RUN_POLICY_V1_END -->"
IDENTITY_SCHEMA = "quactlize-box-run-identity-v2"
PROVENANCE_SCHEMA = "quactlize-box-run-provenance-v2"
IDENTITY_PROBE_SCHEMA = identity_schema.SCHEMA
IDENTITY_FIELDS = identity_schema.FIELDS
IDENTITY_SOURCES = identity_schema.SOURCES


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"box provenance: {message}")


def nonempty(value: str, name: str) -> str:
    if not value or not value.strip():
        fail(f"{name} must be non-empty")
    return value


def known_identity(value: str, name: str) -> str:
    value = nonempty(value, name)
    if value.strip().lower() in {"unknown", "unset", "n/a", "na", "none"}:
        fail(f"{name} must be a concrete measured/operator identity, not {value!r}")
    if "\n" in value or "\r" in value or "\0" in value:
        fail(f"{name} must be a single-line identity")
    return value


def _json_without_duplicates(text: str, path: Path) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                fail(f"identity probe {path} has duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            text, object_pairs_hook=pairs,
            parse_constant=lambda token: fail(
                f"identity probe {path} contains non-finite JSON value {token}"))
    except json.JSONDecodeError as exc:
        fail(f"cannot parse identity probe {path}: {exc}")


def read_identity_probe(path: Path) -> tuple[dict[str, str], dict[str, str], str]:
    """Read and canonically bind the helper's exact identity evidence."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read identity probe {path}: {exc}")
    value = _json_without_duplicates(text, path)
    try:
        values, sources = identity_schema.values_and_sources(value)
        canonical = identity_schema.canonical_bytes(value)
    except identity_schema.IdentityProbeError as exc:
        fail(f"identity probe {path} contradicts its evidence: {exc}")
    return values, sources, hashlib.sha256(canonical).hexdigest()


def identity_probe_for_args(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str], str]:
    values, sources, digest = read_identity_probe(Path(args.identity_probe_file))
    cli_values = {
        "device_model": known_identity(args.device_model, "device_model"),
        "pci_identity": known_identity(args.pci_identity, "pci_identity"),
        "driver_version": known_identity(args.driver_version, "driver_version"),
        "sdk_compiler_identity": known_identity(
            args.sdk_compiler_identity, "sdk_compiler_identity"),
    }
    differing = [field for field in IDENTITY_FIELDS if cli_values[field] != values[field]]
    if differing:
        fail("CLI identity differs from identity probe in: " + ", ".join(differing))
    return values, sources, digest


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


def _successful_commands(path: Path, role: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [item for item in read_commands(path)
            if item["role"] == role and item["exit_status"] == 0]


def _atomic_bytes_write(path: Path, data: bytes) -> None:
    if not data:
        fail(f"refusing an empty authority artifact for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def bind_build_pair(args: argparse.Namespace) -> int:
    """Bind one successful build command and its sibling log to an attempt.

    The attempts directory belongs to one already-verified immutable BASE.
    Reuse may therefore inherit the log from a prior attempt, but only from the
    same attempt as the unique successful command.  The command itself remains
    owned by that prior attempt: copying it into the current journal would
    falsely claim that this attempt executed the compiler and would duplicate
    it when canonical journals are concatenated.  A run-global build log is
    deliberately not an input: it could have been overwritten by a build whose
    device/SDK identity was later rejected.
    """
    attempts = Path(args.attempts_dir)
    commands_path = Path(args.commands_file)
    build_log_path = Path(args.build_log)
    role = args.role

    current = _successful_commands(commands_path, role)
    current_unique = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item
        for item in current
    }
    if current_unique:
        if len(current_unique) != 1 or len(current) != 1:
            fail(f"current attempt has multiple successful {role!r} commands")
        try:
            log_bytes = build_log_path.read_bytes()
        except OSError as exc:
            fail(f"current successful {role!r} command has no sibling build log: {exc}")
        if not log_bytes:
            fail(f"current successful {role!r} command has an empty sibling build log")
        print("current:" + hashlib.sha256(log_bytes).hexdigest())
        return 0

    if build_log_path.exists():
        fail("current attempt has a build log without a successful build command")

    # Key by both the exact command record and log bytes.  Identical repeated
    # pairs collapse, but two different logs or compiler invocations are
    # ambiguous and fail closed rather than selecting by directory order.
    candidates: dict[tuple[str, str], tuple[dict[str, object], bytes]] = {}
    for attempt in sorted(path for path in attempts.iterdir() if path.is_dir()):
        journal = attempt / "commands.jsonl"
        if journal.resolve() == commands_path.resolve() or not journal.exists():
            continue
        successful = _successful_commands(journal, role)
        if not successful:
            continue
        sibling_log = attempt / "build.log"
        try:
            log_bytes = sibling_log.read_bytes()
        except OSError as exc:
            fail(f"successful {role!r} command in {attempt} has no sibling build.log: {exc}")
        if not log_bytes:
            fail(f"successful {role!r} command in {attempt} has an empty build.log")
        log_digest = hashlib.sha256(log_bytes).hexdigest()
        for item in successful:
            encoded = json.dumps(item, sort_keys=True, separators=(",", ":"))
            candidates[(encoded, log_digest)] = (item, log_bytes)

    if not candidates:
        fail(f"no identity-matched successful {role!r} build-log/command pair")
    if len(candidates) != 1:
        fail(f"multiple distinct identity-matched {role!r} build-log/command pairs")

    item, log_bytes = next(iter(candidates.values()))
    _atomic_bytes_write(build_log_path, log_bytes)
    print("inherited:" + hashlib.sha256(log_bytes).hexdigest())
    return 0


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
    groups = nonempty(args.groups, "groups")
    if "\n" in groups or "\r" in groups or "\0" in groups:
        fail("groups must be one exact single-line build selection")
    identity_values, identity_sources, identity_probe_sha256 = identity_probe_for_args(args)
    return {
        "schema": IDENTITY_SCHEMA,
        "root_sha": root_sha,
        "submodule_status": read_submodule_status(Path(args.submodule_status_file)),
        "actlize_sha": args.actlize_sha,
        "binary_sha256": binary_sha,
        **identity_values,
        "identity_sources": identity_sources,
        "identity_probe_sha256": identity_probe_sha256,
        "protocol_sample_count": args.protocol_sample_count,
        "groups": groups,
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
        "sdk_compiler_identity", "identity_sources", "identity_probe_sha256",
        "protocol_sample_count", "groups",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        fail(f"run identity {path} does not have the exact v2 schema")
    digest = value.pop("identity_sha256")
    if value.get("schema") != IDENTITY_SCHEMA or not isinstance(digest, str):
        fail(f"run identity {path} has an invalid schema or digest")
    sources = value.get("identity_sources")
    if (not isinstance(sources, dict) or set(sources) != set(IDENTITY_FIELDS) or
            any(source not in IDENTITY_SOURCES for source in sources.values())):
        fail(f"run identity {path} has an invalid identity_sources map")
    if not isinstance(value.get("identity_probe_sha256"), str) or not SHA256_RE.fullmatch(
            value["identity_probe_sha256"]):
        fail(f"run identity {path} has an invalid identity_probe_sha256")
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
        "schema": PROVENANCE_SCHEMA,
        "root_sha": root_sha,
        "root_status": "clean",
        "submodule_status": submodule_status,
        "actlize_sha": args.actlize_sha,
        "binary_sha256": binary_sha,
        # ``argv`` stays the exact top-level runner argv consumed by the v2
        # adjudicator.  ``commands`` closes the otherwise-unrepresentable
        # multi-command build/run/analyse provenance.
        "argv": runner_argv,
        "commands": read_commands(Path(args.commands_file)),
        "runner_exit_status": args.runner_exit_status,
    }
    if args.protocol_sample_count <= 0:
        fail("protocol_sample_count must be positive")
    value["protocol_sample_count"] = args.protocol_sample_count

    identity_values, identity_sources, identity_probe_sha256 = identity_probe_for_args(args)
    value.update(identity_values)
    value["identity_sources"] = identity_sources
    value["identity_probe_sha256"] = identity_probe_sha256

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
        "identity_sources": value["identity_sources"],
        "identity_probe_sha256": value["identity_probe_sha256"],
        "protocol_sample_count": args.protocol_sample_count,
        "groups": nonempty(args.groups, "groups"),
    }
    provenance_identity["identity_sha256"] = identity_digest(provenance_identity)
    if provenance_identity != identity:
        differing = sorted(key for key in identity
                           if identity[key] != provenance_identity[key])
        fail("provenance does not match immutable run identity in: " +
             ", ".join(differing))

    # Publish the immutable identity digest and the exact build selection in
    # canonical provenance.  The verifier must not need an adjacent mutable
    # file merely to establish which GROUPS selection produced the result.
    value["groups"] = provenance_identity["groups"]
    value["run_identity_sha256"] = identity["identity_sha256"]

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

    inherit = sub.add_parser("bind-build-pair")
    inherit.add_argument("--attempts-dir", required=True)
    inherit.add_argument("--commands-file", required=True)
    inherit.add_argument("--build-log", required=True)
    inherit.add_argument("--role", default="device-build")
    inherit.set_defaults(func=bind_build_pair)

    def add_identity_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--root-sha", required=True)
        target.add_argument("--submodule-status-file", required=True)
        target.add_argument("--actlize-sha", required=True)
        target.add_argument("--binary-sha256", required=True)
        target.add_argument("--device-model", required=True)
        target.add_argument("--pci-identity", required=True)
        target.add_argument("--driver-version", required=True)
        target.add_argument("--sdk-compiler-identity", required=True)
        target.add_argument("--identity-probe-file", required=True)
        target.add_argument("--protocol-sample-count", type=int, required=True)
        target.add_argument("--groups", default="not-applicable")

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
    out.add_argument("--identity-probe-file", required=True)
    out.add_argument("--groups", default="not-applicable")
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
