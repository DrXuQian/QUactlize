#!/usr/bin/env python3
"""Admit the standalone-m8 outer-pipe roll from exact PPU disassembly.

The numerator is the source-bound static instruction footprint of the
``while (k_tiles_remaining > 0)`` body in ``MarlinCollectivePPU``.  Whole
symbol counts are retained but never substituted: helper, scheduler and
epilogue code cannot prove that hgcc honored one loop pragma.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys


SOURCE_EXT = r"(?:c|cc|cpp|cu|cuh|h|hh|hpp|inl|inc)"
SOURCE_MARKERS = (
    re.compile(
        rf"\bFile\s+[\"']?(?P<file>.*?\.{SOURCE_EXT})[\"']?\s*,?\s*line\s+(?P<line>\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<file>(?:[A-Za-z]:)?[/\\][^\t\n\"']+?\.{SOURCE_EXT}):(?P<line>\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<file>[^\s\"']+?\.{SOURCE_EXT})\s*[:,]\s*(?:line\s*)?(?P<line>\d+)\b",
        re.IGNORECASE,
    ),
)
INSTRUCTION = re.compile(
    r"^\s*(?P<addr>[0-9a-f]+):(?P<bytes>(?:\s+[0-9a-f]{2}){4,16})"
    r"\s+(?P<op>[A-Za-z_][A-Za-z0-9_.:]*)\s*(?P<args>.*)$",
    re.IGNORECASE,
)


class ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Inst:
    address: str
    op: str
    locations: frozenset[tuple[str, int]]


@dataclass(frozen=True)
class Resource:
    registers: int
    local_fields: tuple[tuple[str, int], ...]

    @property
    def spill_or_stack(self) -> int:
        return sum(value for _, value in self.local_fields)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_marker(text: str) -> tuple[str, int] | None:
    for pattern in SOURCE_MARKERS:
        match = pattern.search(text)
        if match:
            return Path(match.group("file")).name, int(match.group("line"))
    return None


def parse_disassembly(text: str) -> tuple[list[Inst], set[tuple[str, int]]]:
    current: frozenset[tuple[str, int]] = frozenset()
    pending: set[tuple[str, int]] = set()
    marks: set[tuple[str, int]] = set()
    result: list[Inst] = []
    addresses: set[str] = set()
    for raw in text.splitlines():
        marker = _source_marker(raw)
        if marker is not None:
            pending.add(marker)
            marks.add(marker)
            continue
        match = INSTRUCTION.match(raw)
        if not match:
            continue
        if pending:
            current = frozenset(pending)
            pending.clear()
        address = match.group("addr").lower()
        if address in addresses:
            raise ReportError(
                f"exact-symbol line disassembly repeats PC {address}; "
                "static instructions must be counted by unique address"
            )
        addresses.add(address)
        result.append(
            Inst(address, match.group("op").lower(), current)
        )
    return result, marks


def unique_line(path: Path, needle: str) -> int:
    hits = [i for i, line in enumerate(path.read_text().splitlines(), 1) if needle in line]
    if len(hits) != 1:
        raise ReportError(
            f"{path}: anchor {needle!r} occurs {len(hits)} times, expected exactly one"
        )
    return hits[0]


def parse_resource(text: str) -> Resource:
    register_patterns = (
        re.compile(
            r"(?i)\b(?:numregs|registers?|regs?|reg)\s*(?:per[- ]thread)?\s*[:=]\s*(\d+)"
        ),
        re.compile(r"(?i)\b(?:numregs|registers?|regs?|reg)\s+(\d+)\b"),
        re.compile(r"(?i)\b(\d+)\s+(?:registers?|regs?)\b"),
    )
    registers: list[int] = []
    for line in text.splitlines():
        for pattern in register_patterns:
            match = pattern.search(line)
            if match:
                registers.append(int(match.group(1)))
                break
    registers = sorted(set(registers))
    if len(registers) != 1:
        raise ReportError(
            f"resource report exposes {registers or 'no'} unambiguous register count"
        )

    # Fail closed: absence of a local/spill field is not evidence of zero.
    # Parse values by their labels rather than by line: hgobjdump may print
    # Stack Frame, Spill Stores and Spill Loads together on one line.
    local_pattern = re.compile(
        r"(?i)\b(?P<label>"
        r"spill(?:[- _](?:loads?|stores?|bytes?))?"
        r"|stack(?:[- _]frame)?"
        r"|scratch(?:[- _](?:bytes?|size))?"
        r"|local(?:[- _]memory)?"
        r")\b\s*(?:per[- ]thread\s*)?[:=]?\s*(?P<value>\d+)"
    )
    by_label: dict[str, set[int]] = {}
    raw_label: dict[str, str] = {}
    for match in local_pattern.finditer(text):
        label = re.sub(r"[- _]+", "-", match.group("label").lower())
        by_label.setdefault(label, set()).add(int(match.group("value")))
        raw_label.setdefault(label, match.group("label"))
    ambiguous = {label: sorted(values) for label, values in by_label.items() if len(values) != 1}
    if ambiguous:
        raise ReportError(f"resource report exposes contradictory local/spill fields: {ambiguous}")
    local_fields = [
        (raw_label[label], next(iter(values)))
        for label, values in sorted(by_label.items())
    ]
    if not local_fields:
        raise ReportError("resource report has no explicit spill/stack/local field")
    return Resource(registers[0], tuple(local_fields))


def classify_mode(
    line_path: Path,
    resource_path: Path,
    source: Path,
) -> dict[str, object]:
    instructions, marks = parse_disassembly(line_path.read_text(errors="replace"))
    if not instructions:
        raise ReportError(f"{line_path}: no PPU instruction parsed")
    start = unique_line(source, "while (k_tiles_remaining > 0)")
    end = unique_line(source, "a_pointer += AGlobalOuter * Stages;")
    source_name = source.name
    marked = {
        line for file, line in marks
        if file == source_name and start <= line <= end
    }
    if not marked:
        raise ReportError(
            f"{line_path}: exact symbol has no source binding in {source_name}:{start}-{end}"
        )
    mainloop = [
        inst for inst in instructions
        if any(file == source_name and start <= line <= end
               for file, line in inst.locations)
    ]
    if not mainloop:
        raise ReportError(f"{line_path}: source-bound mainloop owns no instruction")
    resource = parse_resource(resource_path.read_text(errors="replace"))
    return {
        "line_sha256": sha(line_path),
        "resource_sha256": sha(resource_path),
        "whole_symbol_static_instructions": len(instructions),
        "mainloop_static_instructions": len(mainloop),
        "mainloop_source_range": [start, end],
        "mainloop_marked_lines": sorted(marked),
        "registers_per_thread": resource.registers,
        "local_fields": [list(item) for item in resource.local_fields],
        "spill_stack_local_total": resource.spill_or_stack,
    }


def compare(modes: dict[str, dict[str, object]]) -> dict[str, object]:
    baseline = modes["baseline"]
    rolled = modes["outer-roll"]
    inner = modes["inner-roll-control"]
    b_count = int(baseline["mainloop_static_instructions"])
    r_count = int(rolled["mainloop_static_instructions"])
    ratio = b_count / r_count
    if ratio < 3.5:
        raise ReportError(
            f"outer pragma was not exercised: mainloop {b_count}->{r_count}, ratio={ratio:.3f} < 3.5"
        )
    if int(baseline["registers_per_thread"]) != 124:
        raise ReportError(
            f"baseline resource identity drifted: regs={baseline['registers_per_thread']} != 124"
        )
    if int(baseline["spill_stack_local_total"]) != 0:
        raise ReportError("baseline has nonzero spill/stack/local resource")
    if int(rolled["registers_per_thread"]) > 124:
        raise ReportError(
            f"outer-roll exceeds the 124-register admission limit: {rolled['registers_per_thread']}"
        )
    if int(rolled["spill_stack_local_total"]) != 0:
        raise ReportError("outer-roll has nonzero spill/stack/local resource")
    inner_red = (
        int(inner["registers_per_thread"]) > 124
        or int(inner["spill_stack_local_total"]) > 0
    )
    if not inner_red:
        raise ReportError(
            "inner-roll causal control did not violate the register/spill criterion; "
            "the resource admission check has not been falsified"
        )
    return {
        "baseline_to_outer_static_ratio": ratio,
        "outer_pragma_exercised": True,
        "outer_resource_admitted": True,
        "inner_resource_control_red": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    for prefix in ("baseline", "outer-roll", "inner-roll-control"):
        parser.add_argument(f"--{prefix}-line", type=Path, required=True)
        parser.add_argument(f"--{prefix}-resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        modes = {}
        for prefix in ("baseline", "outer-roll", "inner-roll-control"):
            modes[prefix] = classify_mode(
                getattr(args, prefix.replace("-", "_") + "_line"),
                getattr(args, prefix.replace("-", "_") + "_resource"),
                args.source,
            )
        verdict = compare(modes)
    except (OSError, ReportError) as exc:
        print(f"[l182:ppu] FAIL: {exc}", file=sys.stderr)
        return 1
    payload = {
        "schema": "quactlize.l182.marlin-pipe-roll-codegen.v1",
        "source_sha256": sha(args.source),
        "modes": modes,
        "verdict": verdict,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "[l182:ppu] PASS: source-bound mainloop static footprint "
        f"{modes['baseline']['mainloop_static_instructions']} -> "
        f"{modes['outer-roll']['mainloop_static_instructions']} "
        f"({verdict['baseline_to_outer_static_ratio']:.3f}x); "
        f"regs {modes['baseline']['registers_per_thread']} -> "
        f"{modes['outer-roll']['registers_per_thread']}; spills=0; "
        "inner-roll resource control=RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
