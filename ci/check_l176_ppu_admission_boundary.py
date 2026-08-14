#!/usr/bin/env python3
"""Keep local CUDA oracles out of the real PPU codegen runner.

The box's ``nvcc`` delegates device preprocessing to ppu_clang++ and cannot
execute L169/L174/L175's NVIDIA/stub fixtures.  Those proofs are regenerated
by the local tier and committed at the result SHA.  The box must consume that
evidence, then independently build/link/disassemble the real hgcc target.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "dev/fold_derivation/run_l176_standalone_marlin_codegen.sh"


def audit(text: str) -> list[str]:
    errors: list[str] = []
    local_anchor = 'done\n\nif [[ "$MODE" == local ]]; then'
    ppu_anchor = "\nelse\n"
    end_anchor = "\nfi\n\nresolve_executable()"
    if text.count(local_anchor) != 1 or text.count(end_anchor) != 1:
        return ["local/PPU admission branch is not unique"]
    prefix, tail = text.split(local_anchor, 1)
    if ppu_anchor not in tail:
        return ["local admission branch has no PPU alternative"]
    local, ppu_tail = tail.split(ppu_anchor, 1)
    ppu, suffix = ppu_tail.split(end_anchor, 1)

    local_calls = ('bash "$RUN169"', 'bash "$RUN174"', 'bash "$RUN175"')
    for call in local_calls:
        if local.count(call) != 1:
            errors.append(f"local branch lacks exactly one {call}")
        if call in prefix or call in ppu:
            errors.append(f"PPU path can execute local compile oracle {call}")
    for variable in ("RUN169", "RUN174", "RUN175"):
        if variable in ppu or variable in suffix:
            errors.append(f"PPU path retains an equivalent local-oracle seam {variable}")

    required_ppu = (
        'git -C "$ROOT" show "HEAD:$COMMITTED_EVIDENCE_REL"',
        'python3 "$COMMITTED_CHECK" --committed-only --evidence "$committed"',
        "fresh-box-execution=0",
        "The real generated unit is still built,",
    )
    for token in required_ppu:
        if ppu.count(token) != 1:
            errors.append(f"PPU admission lacks {token!r}")
    for forbidden in (
        'cat "$ROOT/$COMMITTED_EVIDENCE_REL"',
        "fresh-box-execution=1",
        "[l176:local] PASS",
    ):
        if forbidden in ppu:
            errors.append(f"PPU admission contains forbidden {forbidden!r}")

    # A committed admission sentence is not a PPU postcondition.  Require the
    # real build, exact generated unit, SDK-owned disassembler and all four
    # disassembly/report phases to remain in the executable suffix.
    required_suffix = (
        'sdk_hgobjdump="$(resolve_executable "$sdk_root/bin/hgobjdump" || true)"',
        '[[ -n "$sdk_hgobjdump" && "$sdk_hgobjdump" == "$hgobjdump" ]]',
        "env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX",
        "PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS=",
        "TARGET=test_lowbit_dense_marlin_wk4_ab JOBS=1",
        'MOE_CORES= GEMV_GROUPS= "$ROOT/build.sh"',
        'python3 "$SOURCE" --generated-unit "$generated" --output "$OUTPUT/source.before.json"',
        "$hgobjdump -lelf \"$bin\"",
        '$hgobjdump -line "-func=$symbol" "$bin"',
        '$hgobjdump "-res-usage=$symbol" "$bin"',
        '$hgobjdump -isa "$bin"',
        'python3 "$REPORT" --line "$OUTPUT/kernel-line.txt"',
        'python3 "$SOURCE" --generated-unit "$generated" --output "$OUTPUT/source.after.json"',
        'cmp "$OUTPUT/source.before.json" "$OUTPUT/source.after.json"',
        'echo "hgobjdump_identity=$hgobjdump_identity"',
        'echo "hgcc_sha256=$hgcc_sha256"',
        'echo "hgobjdump_sha256=$hgobjdump_sha256"',
    )
    for token in required_suffix:
        if suffix.count(token) != 1:
            errors.append(f"real PPU suffix lacks exactly one {token!r}")
    executable_ppu = "\n".join(
        line for line in (ppu + suffix).splitlines()
        if not line.lstrip().startswith("#")
    )
    if re.search(r"\b(?:nvcc|NVCC)\b", executable_ppu):
        errors.append("real PPU path can invoke nvcc instead of the SDK toolchain")
    if 'status --porcelain=v1 --untracked-files=all -- "${source_paths[@]}" || true' in suffix:
        errors.append("source cleanliness still fails open on a git error")
    return errors


def plant(text: str, name: str) -> str:
    if name == "unguarded-l169":
        needle = 'done\n\nif [[ "$MODE" == local ]]; then'
        return text.replace(
            needle, 'done\n\nbash "$RUN169"\n\nif [[ "$MODE" == local ]]; then', 1)
    if name == "working-tree-evidence":
        needle = 'git -C "$ROOT" show "HEAD:$COMMITTED_EVIDENCE_REL"'
        return text.replace(needle, 'cat "$ROOT/$COMMITTED_EVIDENCE_REL"', 1)
    if name == "fresh-box-claim":
        return text.replace("fresh-box-execution=0", "fresh-box-execution=1", 1)
    if name == "delete-real-build":
        return text.replace("env -u CMAKE_GENERATOR", "env -u CMAKE_GENERATOR_BROKEN", 1)
    if name == "ppu-nvcc":
        needle = '  committed="$tmp/l143-committed-evidence.txt"'
        return text.replace(needle, "  nvcc broken.cu\n" + needle, 1)
    if name == "external-objdump":
        return text.replace(
            '[[ -n "$sdk_hgobjdump" && "$sdk_hgobjdump" == "$hgobjdump" ]]',
            '[[ -n "$sdk_hgobjdump" ]]', 1,
        )
    if name == "injected-defines":
        return text.replace(
            "PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS=",
            "CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS=", 1,
        )
    raise ValueError(name)


def main() -> int:
    text = RUNNER.read_text()
    errors = audit(text)
    if errors:
        print("[l176-boundary] FAIL: " + "; ".join(errors))
        return 1
    plants = (
        "unguarded-l169", "working-tree-evidence", "fresh-box-claim",
        "delete-real-build", "ppu-nvcc", "external-objdump", "injected-defines",
    )
    escaped = [name for name in plants if not audit(plant(text, name))]
    if escaped:
        print("[l176-boundary] FAIL: controls escaped: " + ",".join(escaped))
        return 1
    print(
        "[l176-boundary] PASS: local L169/L174/L175 execute only in local mode; "
        "PPU mode consumes result-SHA evidence before an override-free SDK-owned "
        "hgcc/hgobjdump build; negative_controls=7/7_RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
