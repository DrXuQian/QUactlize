#!/usr/bin/env python3
"""Reject NVIDIA-only spellings in the source graph of real PPU targets.

Applicability is deliberately structural.  A file is checked only when the real
``CMakeLists.txt.in`` registers it in a PPU executable/library, or a checked
translation unit reaches it through a project-owned include edge.  Merely living
under ``tests/``, ``benchmarks/`` or ``dev/`` does not make a file PPU code.

This distinction matters: an RTX5090-only benchmark used to stop every PPU build
because the old checker enumerated directories before CMake.  Adding an exception
for that filename fixed one island and guaranteed that the next island would fail
in the same way.  Here CMake registration is the opt-in authority, so an
unregistered NVIDIA TU is N/A while registering that same TU is a hard failure.

The check is SDK-free.  It evaluates the branch the box compiles
(``__HGGCCC__`` defined and ``ENABLE_BF16`` undefined), so this property is fully
local.  Missing CMake is an explicit SKIP (rc=2); a broken/empty source graph or a
NVIDIA spelling in a reachable source is always FAIL (rc=1).
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


DENY = re.compile(
    r"cuda_fp16\.h|cuda_bf16\.h|cuda_runtime\.h|cudaStream_t|cudaMalloc|cudaFree|cudaMemcpy"
    r"|cudaDeviceSynchronize|cudaGetLastError|cudaError_t|cudaSuccess|cudaEvent"
    r"|__halves2half2|__halves2bfloat162|__nv_bfloat"
)

# These are the box's relevant preprocessing facts.  Unknown macros remain
# unknown and both sides are scanned; known facts select exactly one side.
DEFINED = {"__HGGCCC__"}
UNDEFINED = {"ENABLE_BF16"}
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')


class GraphError(RuntimeError):
    pass


class GraphSkip(RuntimeError):
    pass


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _condition(expr: str) -> bool | None:
    """Evaluate the small preprocessor subset needed for platform branches.

    Returning None means an unrelated build option participates.  The caller
    then keeps both branches live, which is conservative without misreading
    ``!defined(__HGGCCC__)`` as the PPU branch.
    """

    unknown = False

    def replace_defined(match: re.Match[str]) -> str:
        nonlocal unknown
        name = match.group(1) or match.group(2)
        if name in DEFINED:
            return "1"
        if name in UNDEFINED:
            return "0"
        unknown = True
        return "0"

    value = re.sub(
        r"defined\s*\(\s*([A-Za-z_]\w*)\s*\)|defined\s+([A-Za-z_]\w*)",
        replace_defined,
        expr,
    )

    def replace_identifier(match: re.Match[str]) -> str:
        nonlocal unknown
        name = match.group(0)
        if name in UNDEFINED:
            return "0"
        # Knowing that a macro is defined does not reveal its numeric value.
        # `#ifdef __HGGCCC__` is exact through replace_defined(), whereas
        # `#if __HGGCCC__ >= 12` must keep both branches live unless the box's
        # exact version value is part of this model.
        unknown = True
        return "0"

    value = re.sub(r"\b[A-Za-z_]\w*\b", replace_identifier, value)
    if unknown:
        return None
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", " not ", value)
    if not re.fullmatch(r"[\s0-9()<>!=&|+*/%.-]*(?:and|or|not)?[\s0-9()<>!=&|+*/%a-z.-]*", value):
        return None
    try:
        return bool(eval(value, {"__builtins__": {}}, {}))  # noqa: S307: sanitized expression above
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None


def live_lines(path: Path):
    """Yield lines that can be live under the box's known preprocessing facts."""

    # Each frame stores [current_branch_possible, later_branch_possible].
    stack: list[list[bool]] = []
    conditional = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
    for lineno, line in enumerate(path.open(errors="replace"), 1):
        match = conditional.match(line)
        if match:
            kind, rest = match.group(1), match.group(2).strip()
            if kind in ("if", "ifdef", "ifndef"):
                parent_possible = all(frame[0] for frame in stack)
                if kind == "if":
                    result = _condition(rest)
                else:
                    result = _condition(f"defined({rest})")
                    if kind == "ifndef" and result is not None:
                        result = not result
                current = parent_possible and result is not False
                remaining = parent_possible and result is not True
                stack.append([current, remaining])
            elif kind == "elif" and stack:
                result = _condition(rest)
                remaining = stack[-1][1]
                stack[-1][0] = remaining and result is not False
                stack[-1][1] = remaining and result is not True
            elif kind == "else" and stack:
                stack[-1][0] = stack[-1][1]
                stack[-1][1] = False
            elif kind == "endif" and stack:
                stack.pop()
            continue
        if all(frame[0] for frame in stack):
            yield lineno, line.rstrip("\n")


def _cmake_driver(root: Path, extra_dirs: list[Path], suffix: str) -> str:
    source_dirs = [
        root / "tests",
        root / "benchmarks",
        root / "dev",
        root / "quactlize/csrc/device",
        root / "quactlize/csrc",
    ] + extra_dirs
    dirs = ";".join(str(path.resolve()) for path in source_dirs if path.is_dir())
    authority = root / "quactlize/csrc/CMakeLists.txt.in"
    # The lower-level CUTLASS functions are replaced only to record their
    # resolved source arguments.  All registration, optional-target logic and
    # generated-unit construction above them is the repository's real CMake.
    return f'''cmake_minimum_required(VERSION 3.18)
project(qz_ppu_source_graph NONE)
set(QZ_ROOT "{root.resolve()}")
set(QZ_SRC_DIRS "{dirs}")
set(CUTLASS_PPU_ARCHS "ppu0010" CACHE STRING "PPU graph architecture" FORCE)
set(_QZ_PPU_SOURCE_MANIFEST "${{CMAKE_BINARY_DIR}}/ppu_sources.txt")
file(WRITE "${{_QZ_PPU_SOURCE_MANIFEST}}" "")

function(_qz_record_ppu_sources)
  foreach(_arg ${{ARGN}})
    get_filename_component(_abs "${{_arg}}" ABSOLUTE BASE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
    if(EXISTS "${{_abs}}" AND NOT IS_DIRECTORY "${{_abs}}")
      file(APPEND "${{_QZ_PPU_SOURCE_MANIFEST}}" "${{_abs}}\\n")
    endif()
  endforeach()
endfunction()
function(cutlass_add_executable NAME)
  _qz_record_ppu_sources(${{ARGN}})
  add_custom_target(${{NAME}})
endfunction()
function(cutlass_add_library NAME)
  _qz_record_ppu_sources(${{ARGN}})
  add_custom_target(${{NAME}})
endfunction()
function(target_compile_definitions)
endfunction()
function(target_compile_options)
endfunction()
function(target_include_directories)
endfunction()
function(target_link_libraries)
endfunction()
function(set_target_properties)
endfunction()
function(add_dependencies)
endfunction()

include("{authority.resolve()}")
{suffix}
'''


def _configured_roots(root: Path, workspace: Path, *, extra_dirs=None, suffix="") -> list[Path]:
    cmake = shutil.which("cmake")
    if not cmake:
        raise GraphSkip("cmake is unavailable; the PPU source graph cannot be evaluated")
    source = workspace / "source"
    build = workspace / "build"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        _cmake_driver(root, list(extra_dirs or []), suffix), encoding="utf-8"
    )
    result = subprocess.run(
        [cmake, "-S", str(source), "-B", str(build)],
        text=True,
        capture_output=True,
        cwd=root,
    )
    if result.returncode:
        detail = "\n".join((result.stdout + result.stderr).splitlines()[-12:])
        raise GraphError(f"CMake could not evaluate the PPU source graph:\n{detail}")
    manifest = build / "ppu_sources.txt"
    if not manifest.is_file():
        raise GraphError("CMake completed without writing the PPU source manifest")
    roots = sorted({Path(line).resolve() for line in manifest.read_text().splitlines() if line.strip()})
    if not roots:
        raise GraphError("CMake evaluated an empty PPU translation-unit graph")
    missing = [path for path in roots if not path.is_file()]
    if missing:
        raise GraphError("CMake registered missing PPU source(s): " + ", ".join(map(str, missing[:8])))
    return roots


def _owned(path: Path, root: Path, generated_root: Path, extra_owners: list[Path]) -> bool:
    resolved = path.resolve()
    for owner in (root.resolve(), generated_root.resolve(), *(path.resolve() for path in extra_owners)):
        try:
            resolved.relative_to(owner)
            break
        except ValueError:
            continue
    else:
        return False
    # Vendor portability is governed by the pinned actlize boundary, not by
    # this repository-owned source check.
    try:
        rel = resolved.relative_to(root.resolve())
        if rel.parts and rel.parts[0] == "third_party":
            return False
    except ValueError:
        pass
    # A C/C++ include edge, rather than a suffix allow-list, determines
    # reachability.  In particular production implementation lives in .inl;
    # filtering it out would silently truncate an otherwise correct TU graph.
    return resolved.is_file()


def _include_dirs(root: Path, generated_root: Path, extra_dirs: list[Path]) -> list[Path]:
    return [
        root,
        root / "quactlize/include",
        root / "quactlize/include/gemv_lowbit",
        root / "tests",
        root / "benchmarks",
        root / "dev",
        root / "quactlize/csrc/device",
        root / "quactlize/csrc",
        generated_root,
    ] + extra_dirs


def _resolve_include(
    name: str,
    source: Path,
    search: list[Path],
    root: Path,
    generated_root: Path,
    extra_owners: list[Path],
) -> Path | None:
    candidates = []
    for directory in [source.parent] + search:
        candidate = (directory / name).resolve()
        if candidate.is_file() and _owned(candidate, root, generated_root, extra_owners):
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    if not unique:
        return None
    if len(unique) > 1:
        raise GraphError(
            f"project include {name!r} from {_rel(root, source)} is ambiguous: "
            + ", ".join(_rel(root, path) for path in unique)
        )
    return unique[0]


def _scan_graph(root: Path, generated_root: Path, roots: list[Path], extra_dirs=None):
    extra_owners = list(extra_dirs or [])
    search = _include_dirs(root, generated_root, extra_owners)
    pending = list(roots)
    visited: set[Path] = set()
    hits: list[tuple[Path, int, str]] = []
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        for lineno, text in live_lines(path):
            code = text.split("//", 1)[0]
            if DENY.search(code):
                hits.append((path, lineno, text.strip()))
            include = INCLUDE_RE.match(code)
            if include:
                reached = _resolve_include(
                    include.group(1), path, search, root, generated_root, extra_owners
                )
                if reached is not None and reached not in visited:
                    pending.append(reached)
    return visited, hits


def _print_hits(root: Path, hits) -> None:
    for path, lineno, text in hits:
        print(
            f"  [FAIL] ppu_portability: {_rel(root, path)}:{lineno} is NVIDIA-only "
            f"in a branch the box compiles:\n           {text}"
        )


def main() -> int:
    root = Path(os.environ.get("QUACTLIZE_ROOT") or Path(__file__).resolve().parents[2]).resolve()
    authority = root / "quactlize/csrc/CMakeLists.txt.in"
    if not authority.is_file():
        print(f"  [FAIL] ppu_portability: PPU CMake source authority is missing: {authority}")
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="qz-ppu-portability-") as temp_name:
            temp = Path(temp_name)
            # Structural three-arm control.  The filename is fresh and carries
            # no repository convention: applicability changes only because the
            # real quactlize_ppu_executable registration changes.
            plant = temp / "plant"
            plant.mkdir()
            # Use a .cpp root and .inl include deliberately: both are shipping
            # shapes, and a source/header suffix list once truncated each one.
            subject = plant / "fresh_nvidia_only_151.cpp"
            included_header = plant / "fresh_registered_nvidia_only_151.inl"
            include_owner = plant / "fresh_registered_include_owner_151.cu"
            subject.write_text(
                "#if defined(__HGGCCC__)\n"
                "cudaStream_t registered_stream;\n"
                "#else\n"
                "int registered_stream;\n"
                "#endif\n",
                encoding="utf-8",
            )
            included_header.write_text(
                "#if defined(__HGGCCC__)\n#include <cuda_fp16.h>\n#endif\n",
                encoding="utf-8",
            )
            include_owner.write_text('#include "fresh_registered_nvidia_only_151.inl"\n', encoding="utf-8")

            # First evaluate the normal graph with the planted directory on the
            # resolver path but without registering the subject.  This is the
            # N/A arm for the exact same file registered below.
            normal_workspace = temp / "normal"
            roots = _configured_roots(root, normal_workspace, extra_dirs=[plant])
            visited, hits = _scan_graph(
                root, normal_workspace / "build", roots, extra_dirs=[plant]
            )
            if hits:
                _print_hits(root, hits)
                return 1
            if subject.resolve() in visited:
                raise GraphError("an unregistered fresh NVIDIA-only TU entered the opt-in PPU graph")

            suffix = """
quactlize_ppu_executable(qz_portability_registered_151 fresh_nvidia_only_151.cpp)
quactlize_ppu_executable(qz_portability_include_151 fresh_registered_include_owner_151.cu)
"""
            planted_workspace = temp / "planted"
            planted_roots = _configured_roots(
                root, planted_workspace, extra_dirs=[plant], suffix=suffix
            )
            planted_visited, planted_hits = _scan_graph(
                root, planted_workspace / "build", planted_roots, extra_dirs=[plant]
            )
            hit_paths = {path.resolve() for path, _, _ in planted_hits}
            if subject.resolve() not in planted_visited or subject.resolve() not in hit_paths:
                raise GraphError("the same NVIDIA-only TU did not fail after PPU registration")
            if included_header.resolve() not in hit_paths:
                raise GraphError("a live include from a registered PPU TU did not fail portability")

            print(
                "  [ok]   ppu_portability: "
                f"{len(roots)} CMake-registered PPU TU(s), {len(visited)} reachable owned source(s); "
                "fresh unregistered NVIDIA TU=N/A, the same PPU-live branch after registration and a live include=FAIL"
            )
            return 0
    except GraphSkip as error:
        print(f"  [SKIP] ppu_portability: {error}")
        return 2
    except GraphError as error:
        print(f"  [FAIL] ppu_portability: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
