#!/usr/bin/env python3
"""Bind the standalone Marlin codegen probe to the shipping generated unit.

This file deliberately proves source identity, not PPU opcodes.  The latter
are accepted only from run_l176_standalone_marlin_codegen.sh's hgcc/hgobjdump
arm.  Keeping the boundary explicit prevents a local nvcc/PTX build from
masquerading as evidence about the PPU backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
L169 = ROOT / "dev/fold_derivation/l169_standalone_marlin_unit.cu"
BUILD = ROOT / "build.sh"
ROOT_CMAKE = ROOT / "CMakeLists.txt"
PPU_CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
UNIT = ROOT / "benchmarks/lowbit_dense_unit.inc"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
COLLECTIVE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "marlin_collective_ppu.hpp"
)
LOAD = COLLECTIVE.with_name("marlin_load_ppu.hpp")
DEQUANT = COLLECTIVE.with_name("marlin_dequant_ppu.hpp")
MMA = COLLECTIVE.with_name("marlin_mma_ppu.hpp")
KERNEL = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
    "marlin_kernel_ppu.hpp"
)
OUTPUT_MAP = KERNEL.with_name("marlin_output_map_ppu.hpp")
HANDLE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/device/"
    "marlin_gemm_ppu.hpp"
)
SCHEDULER = KERNEL.with_name("marlin_scheduler_ppu.hpp")

EXPECTED_FN = "lowbit_dense_cfg_tm16_tn128_tk128_wm16_wn64_st4_bc0"
EXPECTED_ROW = (16, 128, 128, 16, 64, 4, 0)
AUTHORITY = (
    L169, BUILD, ROOT_CMAKE, PPU_CMAKE, CMAKE, UNIT, BENCH,
    COLLECTIVE, LOAD, DEQUANT, MMA,
    KERNEL, OUTPUT_MAP, HANDLE, SCHEDULER,
)
SUBMODULE_AUTHORITY = (
    ROOT / "third_party/actlize",
    ROOT / "third_party/cutlass",
)


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_all() -> dict[Path, str]:
    result: dict[Path, str] = {}
    for path in AUTHORITY:
        try:
            result[path] = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContractError(f"cannot read {rel(path)}: {exc}") from exc
    return result


def run_git(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", "") or str(exc)
        raise ContractError(
            f"git {' '.join(args)} failed under {cwd}: {detail.strip()}"
        ) from exc


def submodule_authority() -> dict[str, str]:
    """Bind vendor headers to clean worktrees at the result-SHA gitlinks."""
    result: dict[str, str] = {}
    for path in SUBMODULE_AUTHORITY:
        relative = rel(path)
        tree = run_git("ls-tree", "HEAD", "--", relative)
        match = re.fullmatch(
            rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}", tree
        )
        if match is None:
            raise ContractError(f"{relative}: result SHA has no unique gitlink: {tree!r}")
        expected = match.group(1)
        actual = run_git("rev-parse", "HEAD", cwd=path)
        if actual != expected:
            raise ContractError(
                f"{relative}: checkout {actual} differs from gitlink {expected}"
            )
        dirty = run_git("status", "--porcelain=v1", "--untracked-files=all", cwd=path)
        if dirty:
            raise ContractError(f"{relative}: vendor worktree is dirty: {dirty!r}")
        result[relative] = actual
    return result


def one(pattern: str, text: str, noun: str, flags: int = 0) -> re.Match[str]:
    hits = list(re.finditer(pattern, text, flags))
    if len(hits) != 1:
        raise ContractError(f"{noun}: found {len(hits)}, expected exactly one")
    return hits[0]


def function_body(text: str, anchor: str, noun: str) -> str:
    start = text.find(anchor)
    if start < 0:
        raise ContractError(f"{noun}: missing function anchor {anchor!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise ContractError(f"{noun}: missing opening brace")
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ContractError(f"{noun}: unbalanced function body")


def cmake_value(text: str, name: str) -> str:
    return one(
        rf"set\(\s*{re.escape(name)}\s+([^\s\)]+)\s*\)", text,
        f"CMake authority {name}", re.S,
    ).group(1)


def parse_generated_unit(text: str, *, noun: str, expected_fn: str) -> None:
    if text.count('#include "lowbit_dense_unit.inc"') != 1:
        raise ContractError(f"{noun}: lowbit_dense_unit.inc include is not unique")
    if text.count("#define PPU_B_CHUNK 0") != 1:
        raise ContractError(f"{noun}: standalone B-chunk-off contract is absent")
    match = one(
        r"X\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
        text, f"{noun}: generated config row",
    )
    fn = match.group(1)
    row = tuple(int(x) for x in match.groups()[1:])
    if fn != expected_fn:
        raise ContractError(f"{noun}: function is {fn}, expected {expected_fn}")
    if row != EXPECTED_ROW:
        raise ContractError(f"{noun}: row is {row}, expected {EXPECTED_ROW}")


def apply_plant(plant: str, texts: dict[Path, str]) -> None:
    if plant == "none":
        return
    if plant == "generated-row":
        texts[L169] = texts[L169].replace(
            "16,128,128,16,64,4,0", "16,128,128,16,32,4,0", 1
        )
    elif plant == "generic-wrapper":
        texts[UNIT] = texts[UNIT].replace(
            "using G = typename StandaloneCfg::MarlinGemm;",
            "using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::Gemm;",
            1,
        )
    elif plant == "runtime-nblock":
        texts[COLLECTIVE] = texts[COLLECTIVE].replace(
            "auto multiply = [&](int inner)",
            "auto multiply = [&](int inner) /* switch (n_block) */",
            1,
        )
    elif plant == "flat-accumulator":
        texts[KERNEL] = texts[KERNEL].replace(
            "Accumulator accum;",
            "Accumulator accum; auto* l176_flat = reinterpret_cast<float*>(&accum); (void)l176_flat;",
            1,
        )
    elif plant == "missing-lineinfo":
        texts[CMAKE] = texts[CMAKE].replace(
            "DEV_COMPILE_FLAGS -lineinfo ${_DENSE_MARLIN_WK4_DEFS}",
            "DEV_COMPILE_FLAGS ${_DENSE_MARLIN_WK4_DEFS}",
            1,
        )
    elif plant == "m8-x4-fallback":
        old = "ppu.ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        new = "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1}, [%2];"
        if texts[LOAD].count(old) != 1:
            raise ContractError("m8 x2 opcode plant seam drifted")
        texts[LOAD] = texts[LOAD].replace(old, new, 1)
    elif plant == "m8-discarded-destinations":
        old = ': "=r"(a[0]), "=r"(a[1])\n      : "l"(smem_ptr));'
        new = (
            ': "=r"(a[0]), "=r"(a[1]), "=r"(discarded_v2), '
            '"=r"(discarded_v3)\n      : "l"(smem_ptr));'
        )
        if texts[LOAD].count(old) != 1:
            raise ContractError("m8 discarded-output plant seam drifted")
        texts[LOAD] = texts[LOAD].replace(old, new, 1)
    elif plant == "m8-padded-a":
        old = "static constexpr int AStoredRows = InstructionM == 8 ? 1 : TileM;"
        new = "static constexpr int AStoredRows = InstructionM == 8 ? 8 : TileM;"
        if texts[COLLECTIVE].count(old) != 1:
            raise ContractError("m8 packed-A plant seam drifted")
        texts[COLLECTIVE] = texts[COLLECTIVE].replace(old, new, 1)
    elif plant == "m8-broadens-m":
        old = "bool const m_supported = InstructionM == 8 ? m == 1 : (m > 0 && m <= TileM);"
        new = "bool const m_supported = m > 0 && m <= TileM;"
        if texts[COLLECTIVE].count(old) != 1:
            raise ContractError("m8 M=1 admission plant seam drifted")
        texts[COLLECTIVE] = texts[COLLECTIVE].replace(old, new, 1)
    else:
        raise ContractError(f"unknown plant {plant}")


def validate(texts: dict[Path, str], generated: Path | None) -> dict[str, object]:
    parse_generated_unit(
        texts[L169], noun="L169 generated-unit oracle", expected_fn="l169_standalone_marlin"
    )

    cmake = texts[CMAKE]
    expected_values = {
        "_DENSE_MARLIN_WK4_TM": "16",
        "_DENSE_MARLIN_WK4_TN": "128",
        "_DENSE_MARLIN_WK4_TK": "128",
        "_DENSE_MARLIN_WK4_WM": "16",
        "_DENSE_MARLIN_WK4_WN": "64",
        "_DENSE_MARLIN_WK4_WARP_K": "32",
        "_DENSE_MARLIN_WK4_ST": "4",
    }
    got_values = {name: cmake_value(cmake, name) for name in expected_values}
    if got_values != expected_values:
        raise ContractError(
            f"shipping generated-row CMake values differ: got={got_values} expected={expected_values}"
        )
    for token in (
        "test_lowbit_dense_marlin_wk4_ab",
        '"#include \\"lowbit_dense_unit.inc\\"\\n"',
        "DENSE_MARLIN_WK4_AB=1",
        "DENSE_AB_WARP_K=${_DENSE_MARLIN_WK4_WARP_K}",
        "STAGES=${_DENSE_MARLIN_WK4_ST}",
        "DEV_COMPILE_FLAGS -lineinfo ${_DENSE_MARLIN_WK4_DEFS}",
    ):
        if token not in cmake:
            raise ContractError(f"shipping generated-unit authority lacks {token!r}")

    unit = texts[UNIT]
    function_start = unit.find("static Result lowbit_dense_run_config")
    function_end = unit.find("#if DENSE_GS_ARM(16)", function_start)
    if function_start < 0 or function_end < 0:
        raise ContractError("cannot isolate lowbit_dense_run_config")
    function = unit[function_start:function_end]
    arm = one(
        r"#if defined\(DENSE_MARLIN_WK4_AB\)(.*?)#else",
        function, "standalone wrapper arm", re.S,
    ).group(1)
    required = (
        "using StandaloneCfg = StandaloneMarlinCfg<",
        "using G = typename StandaloneCfg::MarlinGemm;",
        "Kernel::IsStandaloneMarlin",
        "typename StandaloneCfg::MarlinMain",
        'return run<G>(options, dense_tactic(cfg), "marlin");',
    )
    for token in required:
        if token not in arm:
            raise ContractError(f"standalone generated wrapper lacks {token!r}")
    for forbidden in ("::Gemm;", "::StreamKGemm", "::PersistentGemm"):
        if forbidden in arm:
            raise ContractError(f"standalone wrapper reached generic kernel via {forbidden!r}")

    bench = texts[BENCH]
    for token in (
        '#include "actlize_extensions/cutlass/gemm/device/marlin_gemm_ppu.hpp"',
        "using MarlinGemm = cutlass::gemm::device::MarlinGemmPPU<MarlinKernel>;",
    ):
        if token not in bench:
            raise ContractError(f"shipping standalone Cfg lacks {token!r}")
    handle = texts[HANDLE]
    for token in (
        "class MarlinCheckedHandlePPU", "bool installed_ = false",
        "Status update(Arguments const&, void* = nullptr) = delete",
        "static dim3 get_grid_shape(Params const&) = delete",
        "using MarlinGemmPPU = detail::MarlinCheckedHandlePPU<",
    ):
        if token not in handle:
            raise ContractError(f"owned host lowering lacks {token!r}")

    collective = texts[COLLECTIVE]
    load = texts[LOAD]
    for helper in (LOAD, DEQUANT, MMA):
        include = (
            '#include "actlize_extensions/cutlass/gemm/collective/'
            f'{helper.name}"'
        )
        if collective.count(include) != 1:
            raise ContractError(f"collective does not include exactly one {helper.name}")
    for forbidden in (
        "struct FragmentB {", "lop3.b32", "cp.async.cg.shared.global",
    ):
        if forbidden in collective:
            raise ContractError(f"standalone collective regained monolithic/runtime seam {forbidden!r}")
    multiply_start = collective.find("auto multiply =")
    multiply_end = collective.find("\n    };", multiply_start)
    if multiply_start < 0 or multiply_end < 0:
        raise ContractError("cannot isolate production multiply cadence")
    multiply = collective[multiply_start:multiply_end]
    for forbidden in ("for (int n_block", "if (n_block", "switch (n_block)"):
        if forbidden in multiply:
            raise ContractError(f"standalone multiply regained runtime seam {forbidden!r}")
    calls = [int(x) for x in re.findall(r"multiply_n_block<(\d)>", multiply)]
    if calls != [0, 1, 2, 3]:
        raise ContractError(f"standalone multiply cadence is {calls}, expected [0,1,2,3]")

    for token in (
        "struct FragmentA {", "__half2 value[4];", "struct FragmentA8 {",
        "__half2 value[2];", "FragmentAFor = std::conditional_t<",
        "sizeof(FragmentA8) == 2 * sizeof(uint32_t)",
        "sizeof(FragmentA) == 4 * sizeof(uint32_t)",
    ):
        if token not in load:
            raise ContractError(f"standalone A-fragment source lacks {token!r}")
    m16_load = function_body(
        load, "CUTLASS_DEVICE void ldmatrix_a_m16(", "m16 A load"
    )
    m8_load = function_body(
        load, "CUTLASS_DEVICE void ldmatrix_a_m8(", "m8 A load"
    )
    x4 = "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16"
    x2 = "ppu.ldmatrix.sync.aligned.m8n8.x2.shared.b16"
    if m16_load.count(x4) != 1 or x2 in m16_load:
        raise ContractError("m16 A load is not exactly the unchanged x4 path")
    if ': "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3])' not in m16_load:
        raise ContractError("m16 x4 raw-register permutation drifted")
    if m8_load.count(x2) != 1 or x4 in m8_load:
        raise ContractError("m8 A load is not exactly the PPU plain-x2 path")
    if ': "=r"(a[0]), "=r"(a[1])' not in m8_load:
        raise ContractError("m8 x2 no longer publishes exactly a[0],a[1]")
    if "discarded_" in m8_load or len(re.findall(r'"=r"\s*\(a\[\d\]\)', m8_load)) != 2:
        raise ContractError("m8 A load regained x4/discarded destinations")
    for token in (
        "static constexpr int AStoredRows = InstructionM == 8 ? 1 : TileM;",
        ": ASharedStage == 16) &&",
        "sizeof(SharedStorage) == (InstructionM == 8 ? 34816 : 50176)",
        "bool const m_supported = InstructionM == 8 ? m == 1 : (m > 0 && m <= TileM);",
    ):
        if token not in collective:
            raise ContractError(f"m8 packed-A/M=1 source contract lacks {token!r}")

    kernel = texts[KERNEL]
    output_map = texts[OUTPUT_MAP]
    mma = texts[MMA]
    if ("using Accumulator = marlin_ppu_detail::MarlinAccumulatorForN<" not in collective or
            "InstructionM, NBlocksPerWarp>;" not in collective):
        raise ContractError(
            "collective lost its atom-M/N-selected native Marlin accumulator alias"
        )
    if "Accumulator accum;" not in kernel:
        raise ContractError("kernel no longer constructs the native accumulator directly")
    if kernel.count(
        '#include "actlize_extensions/cutlass/gemm/kernel/marlin_output_map_ppu.hpp"'
    ) != 1:
        raise ContractError("kernel lost its authoritative output-map include")
    for token in (
        "constexpr int output_row(", "constexpr int output_n_base(",
        "constexpr int output_col_offset(",
    ):
        if output_map.count(token) != 1:
            raise ContractError(f"authoritative output map lacks {token!r}")
    if ("FragmentC fragments[4];" not in mma or "float value[8];" not in mma or
            "FragmentC8 fragments[4];" not in mma or "float value[4];" not in mma):
        raise ContractError("native 4x8 FP32 accumulator layout drifted")
    joined = "\n".join((collective, mma, kernel))
    if re.search(
        r"reinterpret_cast\s*<\s*(?:const\s+)?float\s*\*\s*>\s*\(\s*&?\s*accum\b",
        joined,
    ):
        raise ContractError("whole accumulator escaped through a flat float pointer")
    for forbidden in ("partition_fragment_C", "make_fragment_like"):
        if forbidden in joined:
            raise ContractError(f"generic accumulator builder returned: {forbidden}")

    generated_record: dict[str, object] | None = None
    if generated is not None:
        try:
            data = generated.read_bytes()
            generated_text = data.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ContractError(f"cannot read generated unit {generated}: {exc}") from exc
        parse_generated_unit(
            generated_text, noun="shipping build generated unit", expected_fn=EXPECTED_FN
        )
        generated_record = {
            "path": str(generated.resolve()),
            "sha256": digest(data),
            "bytes": len(data),
        }

    git_sha = run_git("rev-parse", "HEAD")
    submodules = submodule_authority()
    files = {
        rel(path): {
            "sha256": digest(texts[path].encode()),
            "bytes": len(texts[path].encode()),
        }
        for path in AUTHORITY
    }
    return {
        "schema": "quactlize.l176.standalone-marlin-source.v1",
        "git_sha": git_sha,
        "submodules": submodules,
        "generated_row": {
            "function": EXPECTED_FN,
            "tile": [16, 128, 128],
            "warp": [16, 64, 32],
            "stages": 4,
            "b_chunk": 0,
            "cta_threads": 256,
            "output_threads": 64,
        },
        "source_files": files,
        "generated_unit": generated_record,
        "claims": {
            "generated_route": "standalone-only",
            "helper_cadence": "source-bound-by-L174",
            "accumulator": "native-C:m16-4x8-fp32,m8-4x4-fp32",
            "a_fragment": "m16-x4-4reg-16B,m8-plain-x2-2reg-8B",
            "a_shared_bytes": "m16=50176,m8-M1-packed=34816",
            "m8_problem_m": "exactly-1",
            "generic_fragment": "absent",
            "flat_accumulator_address_escape": "absent",
            "ppu_opcode_or_spill": "not-established-by-this-source-contract",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generated-unit", type=Path)
    parser.add_argument("--plant", default="none")
    args = parser.parse_args()
    try:
        texts = read_all()
        apply_plant(args.plant, texts)
        payload = validate(texts, args.generated_unit)
    except ContractError as exc:
        prefix = "[l176:source] FAIL" if args.plant == "none" else f"[l176:red] plant={args.plant}"
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"[l176:red] FAIL: plant={args.plant} escaped", file=sys.stderr)
        return 1
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(
        "[l176:source] PASS: shipping generated row + standalone wrapper + split helpers + "
        "native m16/m8 C+A fragments and packed-M1 A ledger are hash-bound; "
        "PPU opcode/spill remains a disassembly postcondition"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
