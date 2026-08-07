#!/usr/bin/env python3
"""Keep written build advice connected to an implemented input path.

This is deliberately separate from check_switch_macros.py.  That checker asks whether a preprocessor
conditional has a recorded setter; this one starts at the other end: every environment assignment presented
as a build command must be consumed by build.sh, and every NAME in a CMake ``Narrow ... (...)`` message must
be either routed by build.sh or declared as a cache variable.

It does not claim that a routed value has an effect.  ci/check_moe_build_knobs.py supplies that independent,
behavioural half for the MoE axes.
"""
from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build.sh"
CMAKE_FILES = (ROOT / "CMakeLists.txt", ROOT / "quactlize/csrc/CMakeLists.txt.in")
DOC_SUFFIXES = {".md", ".txt", ".in", ".cmake", ".sh", ".cu", ".cuh", ".hpp", ".h", ".cpp"}
ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$", re.S)
NARROW = re.compile(r"Narrow[^()\n]*axis[^()\n]*\(([^()\n]*)\)")
NAME = re.compile(r"\b[A-Z][A-Z0-9_]*\b")
VAR_REF = re.compile(r"\$(?:\{)?([A-Z][A-Z0-9_]*)")


@dataclass(frozen=True)
class Advice:
    name: str
    where: str
    kind: str


def tracked_texts() -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    texts: dict[str, str] = {}
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode(errors="surrogateescape")
        path = ROOT / rel
        if (
            rel == ".coord/INBOX.md"
            or rel.startswith("third_party/")
            or path.resolve() == pathlib.Path(__file__).resolve()
            or path.suffix not in DOC_SUFFIXES
        ):
            continue
        try:
            texts[rel] = path.read_text(errors="replace")
        except OSError:
            pass
    return texts


def logical_lines(text: str):
    """Yield backslash-continued shell-ish lines with their first physical line number."""
    pending = ""
    start = 1
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.rstrip()
        if not pending:
            start = lineno
        pending += stripped[:-1] + " " if stripped.endswith("\\") else stripped
        if not stripped.endswith("\\"):
            yield start, pending
            pending = ""
    if pending:
        yield start, pending


def command_advice(texts: dict[str, str]) -> list[Advice]:
    found: list[Advice] = []
    for rel, text in texts.items():
        for lineno, line in logical_lines(text):
            if "./build.sh" not in line:
                continue
            cleaned = re.sub(r"^\s*(?://+|#+|\*+)\s?", "", line)
            try:
                tokens = shlex.split(cleaned, comments=False, posix=True)
            except ValueError:
                # Usually an echo whose quoted payload mentions ./build.sh. It is not a shell command in the
                # tracked source. A real command contributes plenty of standalone tokens and is handled below.
                continue
            for index, token in enumerate(tokens):
                if token.strip("`'\"();,") != "./build.sh":
                    continue
                # Environment assignments form one contiguous block immediately before the command. Walking
                # backwards avoids treating prose such as "TK=256, ... Build: TARGET=x ./build.sh" as shell.
                cursor = index - 1
                while cursor >= 0:
                    match = ASSIGN.match(tokens[cursor].strip("();,"))
                    if not match:
                        break
                    found.append(Advice(match.group(1), f"{rel}:{lineno}", "command"))
                    cursor -= 1
    return found


def narrow_advice(texts: dict[str, str]) -> list[Advice]:
    found: list[Advice] = []
    for rel, text in texts.items():
        for match in NARROW.finditer(text):
            lineno = text.count("\n", 0, match.start()) + 1
            for name in NAME.findall(match.group(1)):
                found.append(Advice(name, f"{rel}:{lineno}", "narrow"))
    return found


def build_mechanisms(build_text: str) -> tuple[set[str], set[str]]:
    """Return (effective environment inputs, explicit CMake-forwarded inputs).

    A bare parameter expansion is not a mechanism: ``echo ${NAME}`` is still accept-then-drop.  Follow simple
    shell assignments into commands that affect configuration, compilation, or the selected build directory.
    The indirect MOE loop is handled separately because its source name is intentionally expanded through ``!``.
    """
    code = "\n".join(line for line in build_text.splitlines() if not line.lstrip().startswith("#"))
    dependencies: dict[str, set[str]] = {}
    for line in code.splitlines():
        for match in re.finditer(r"(?:^|;)\s*([A-Z][A-Z0-9_]*)=([^;\n]*)", line):
            dependencies.setdefault(match.group(1), set()).update(VAR_REF.findall(match.group(2)))

    sink_refs: set[str] = set()
    for _, line in logical_lines(code):
        stripped = line.lstrip()
        if re.match(r"(?:cmake|make|rm|mkdir|cd|find)\b", stripped) or stripped.startswith("export PATH="):
            sink_refs.update(VAR_REF.findall(line))

    effective = set(sink_refs)
    pending = list(sink_refs)
    while pending:
        current = pending.pop()
        for source in dependencies.get(current, ()):
            if source not in effective:
                effective.add(source)
                pending.append(source)

    forwarded: set[str] = set()
    # Require the loop body to construct a -D of the indirect value; an unrelated uppercase for-list is not a route.
    for match in re.finditer(
        r"for\s+(_[A-Za-z0-9_]+)\s+in\s+([^;\n]+);\s*do(?P<body>.*?)done", code, re.S
    ):
        iterator, values, body = match.group(1), match.group(2), match.group("body")
        if f"-D${iterator}" in body and f"${{!{iterator}}}" in body:
            forwarded.update(NAME.findall(values))
    effective.update(forwarded)
    return effective, forwarded


def cache_vars(cmake_text: str) -> set[str]:
    found: set[str] = set()
    for line in cmake_text.splitlines():
        match = re.search(r"\bset\(\s*([A-Z][A-Z0-9_]*)\b.*\bCACHE\s+(?:BOOL|STRING|PATH|FILEPATH|INTERNAL)\b", line)
        if match:
            found.add(match.group(1))
    return found


def violations(advice: list[Advice], command_routes: set[str], forwarded: set[str], caches: set[str]) -> list[str]:
    bad: list[str] = []
    for item in advice:
        # NAME=value ./build.sh is an environment assignment. A same-named CMake cache entry cannot import it;
        # build.sh must read or forward the source name. A CMake-emitted Narrow message may also document direct
        # cmake -D use, so a declared cache variable is sufficient for that class.
        wired = item.name in command_routes if item.kind == "command" else item.name in forwarded or item.name in caches
        if not wired:
            bad.append(f"{item.where}: {item.name} ({item.kind})")
    return bad


def main() -> int:
    if not BUILD.is_file() or any(not path.is_file() for path in CMAKE_FILES):
        print("[build-advice] ERROR: build.sh or a CMake input is missing")
        return 1
    try:
        texts = tracked_texts()
        commands = command_advice(texts)
        narrows = narrow_advice(texts)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[build-advice] ERROR: {exc}")
        return 1
    if len(commands) < 20 or len(narrows) < 5:
        print(f"[build-advice] ERROR: found only {len(commands)} command assignment(s) and "
              f"{len(narrows)} Narrow-axis name(s); the extraction no longer describes the tree")
        return 1

    build_text = BUILD.read_text(errors="replace")
    command_routes, forwarded = build_mechanisms(build_text)
    caches = set()
    for path in CMAKE_FILES:
        caches.update(cache_vars(path.read_text(errors="replace")))
    bad = violations(commands + narrows, command_routes, forwarded, caches)

    # Bidirectional in-memory controls go through the extractors too: an unwired command/message must fail, then
    # each must clear through its permitted mechanism. This catches a regex that quietly stops extracting a side.
    planted_texts = {
        "<planted-command>": "MOE_PLANTED_AXIS=x ./build.sh\n",
        "<planted-message>": 'message(STATUS "Narrow an axis (MOE_PLANTED_CACHE)")\n',
    }
    planted_commands = command_advice(planted_texts)
    planted_narrows = narrow_advice(planted_texts)
    if {item.name for item in planted_commands} != {"MOE_PLANTED_AXIS"} or \
       {item.name for item in planted_narrows} != {"MOE_PLANTED_CACHE"}:
        print("[build-advice] ERROR: the planted command/message extractors did not recover their NAMEs")
        return 1
    if len(violations(planted_commands + planted_narrows, command_routes, forwarded, caches)) != 2:
        print("[build-advice] ERROR: accepted a planted unwired command or Narrow-axis recommendation")
        return 1
    dead_routes, _ = build_mechanisms(build_text + '\nX="${MOE_PLANTED_AXIS:-}"\n')
    if "MOE_PLANTED_AXIS" in dead_routes:
        print("[build-advice] ERROR: a planted dead read was classified as an effective build input")
        return 1
    planted_routes, planted_forwarded = build_mechanisms(
        build_text + '\ncmake -DPLANTED="${MOE_PLANTED_AXIS}" .\n'
    )
    planted_caches = caches | {"MOE_PLANTED_CACHE"}
    if violations(planted_commands + planted_narrows, planted_routes, planted_forwarded, planted_caches):
        print("[build-advice] ERROR: rejected planted advice after adding its build route/cache declaration")
        return 1

    if bad:
        print(f"[build-advice] FAIL: {len(bad)} advertised build input(s) have no implemented route:")
        for item in bad[:20]:
            print(f"    {item}")
        return 1
    unique = {item.name for item in commands + narrows}
    print(f"[build-advice] PASS: {len(unique)} input name(s) across {len(commands)} command assignment(s) and "
          f"{len(narrows)} Narrow-message name(s) are routed; controls fired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
