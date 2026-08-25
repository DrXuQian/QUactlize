#!/usr/bin/env python3
"""Report exact-symbol PPU codegen for one Split-K prefix arm."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instructions(text: str) -> list[str]:
    result: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("//", "#", ".", "File ",
                                        "Function ")):
            continue
        match = re.search(
            r"(?:^|\s)([sv]?\.?[A-Za-z][A-Za-z0-9_]*"
            r"(?:\.[A-Za-z0-9_]+)+)\s+", line)
        if match:
            result.append(match.group(1).lower())
    return result


def count_like(counter: collections.Counter[str], needle: str) -> int:
    return sum(value for opcode, value in counter.items()
               if needle in opcode)


def summarize_resource(text: str) -> list[str]:
    pattern = re.compile(
        r"(?i)\b(register|gpr|vgpr|sgpr|stack|spill|scratch|local[- ]memory)"
    )
    return [line.strip() for line in text.splitlines()
            if pattern.search(line)][:8]


def report(arm: str, kernel: int, line_path: pathlib.Path,
           resource_path: pathlib.Path, demangled_path: pathlib.Path,
           binary_path: pathlib.Path) -> None:
    line = line_path.read_text(errors="replace")
    resource = resource_path.read_text(errors="replace")
    demangled = demangled_path.read_text(errors="replace").strip()
    if not line.strip() or not resource.strip() or not demangled:
        raise ValueError("line/resource/demangled evidence must be nonempty")
    if "GemmUniversalMixedInputSplitKParallel" not in demangled:
        raise ValueError("selected symbol is not the frozen Split-K kernel")
    opcodes = collections.Counter(instructions(line))
    if not opcodes:
        raise ValueError("exact-symbol disassembly contains no parsed opcode")
    focus = {
        name.replace(".", "_"): count_like(opcodes, name)
        for name in ("mma", "smem.st", "tsm.st", "smem.ld", "tsm.ld",
                     "vmem.st", "vmem.ld")
    }
    shared_store_forms = {
        opcode: value for opcode, value in sorted(opcodes.items())
        if "smem.st" in opcode or "tsm.st" in opcode
    }
    # The packed-A schedule wrapper is erased from this SDK's emitted kernel
    # spelling, so absence of its source-level type name is not evidence for
    # the standard provider.  Keep the two exact ELF ordinals and make the
    # provider hint explicitly unresolved unless the token is actually live.
    provider_hint = ("packed-row" if "KernelAiuPackedA<" in demangled
                     else "UNRESOLVED")
    hints = summarize_resource(resource)
    hint_mode = "keywords"
    if not hints:
        hint_mode = "preview"
        hints = [item.strip() for item in resource.splitlines()
                 if item.strip()][:8]
    hint_json = json.dumps(hints, separators=(",", ":"), ensure_ascii=True)
    store_json = json.dumps(shared_store_forms, separators=(",", ":"),
                            ensure_ascii=True)
    focus_text = " ".join(f"{key}={value}" for key, value in focus.items())
    print(
        "FQ_SHARED_PREFIX_CODEGEN "
        f"arm={arm} kernel={kernel} provider_hint={provider_hint} "
        f"instructions={sum(opcodes.values())} "
        f"{focus_text} resource_hint_mode={hint_mode} "
        f"shared_store_forms={store_json} resource_hints={hint_json} "
        f"binary_sha256={sha(binary_path)} line_sha256={sha(line_path)} "
        f"resource_sha256={sha(resource_path)}")


def self_test() -> None:
    sample = (
        "0000 v.mma.f32.f16 r0, r1\n"
        "0004 smem.st.b32 [r2], r3\n"
        "0008 tsm.ld.swzl r4, [r5]\n")
    counts = collections.Counter(instructions(sample))
    assert count_like(counts, "mma") == 1
    assert count_like(counts, "smem.st") == 1
    assert count_like(counts, "tsm.ld") == 1
    assert {opcode: value for opcode, value in counts.items()
            if "smem.st" in opcode or "tsm.st" in opcode} == {
                "smem.st.b32": 1}
    resource = "Registers/Thread = 128\nStack Frame: 0 bytes\n"
    assert summarize_resource(resource) == [
        "Registers/Thread = 128", "Stack Frame: 0 bytes"]
    planted = sample.replace("smem.st.b32", "plain_store", 1)
    planted_counts = collections.Counter(instructions(planted))
    assert count_like(planted_counts, "smem.st") == 0
    print("[fq-shared-prefix-codegen:self-test] PASS opcode parser, flexible "
          "resource hints and shared-store negative")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--kernel", type=int)
    parser.add_argument("--line", type=pathlib.Path)
    parser.add_argument("--resource", type=pathlib.Path)
    parser.add_argument("--demangled", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        required = (args.arm, args.kernel, args.line, args.resource,
                    args.demangled, args.binary)
        if any(value is None for value in required):
            parser.error("all report arguments are required")
        report(args.arm, args.kernel, args.line, args.resource,
               args.demangled, args.binary)
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-prefix-codegen] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
