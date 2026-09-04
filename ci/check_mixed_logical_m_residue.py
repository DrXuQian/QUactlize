#!/usr/bin/env python3
"""Pin mixed-input output residues to logical tiles, not widened load tensors.

The native m8 path deliberately returns a physical 16-row ``gA`` tensor from
``load_init()`` while its scheduler and output tile remain logically 8 rows.
Using ``size<0>(gA)`` for the output residue therefore skips half of a tall M
problem.  This check audits every production mixed-input kernel that constructs
an M/N/K residue:

* M and N advance by the logical ``blk_shape``;
* K retains the loaded-A extent, because that is the real K-tail authority;
* no audited site silently returns to ``gA``/``gB`` for M/N.

The checks operate on source text because the regression is precisely a
kernel/collective contract mismatch: either side type-checks on its own.  Each
accepted site is mutation-tested in memory at the logical-shape binding and in
all three residue dimensions, so a parser weakness cannot make it permanently
green.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Site:
    path: str
    m_coord: str
    n_coord: str
    block_shape: str = "blk_shape"


# Exactly the ten production kernels whose mainloops consume the mixed-input
# load_init() contract.  Adding another such kernel without adding it here is a
# coverage failure (see discover_sites()).
SITES = (
    Site(
        "third_party/actlize/include/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input.hpp",
        "get<0>(blk_coord_mnkl)", "get<1>(blk_coord_mnkl)"),
    Site(
        "third_party/actlize/include/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_serial.hpp",
        "get<0>(blk_coord_mnkl)", "get<1>(blk_coord_mnkl)"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp",
        "m_coord", "n_coord"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp",
        "m_coord", "n_coord"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp",
        "m_coord", "n_coord"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp",
        "m_coord", "n_coord"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_streamk.hpp",
        "m_idx", "n_idx"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_marlin.hpp",
        "m_idx", "n_idx"),
    Site(
        "quactlize/include/ppu_aiu_gemm_mixed_input_group.hpp",
        "m_idx", "n_idx"),
    Site(
        "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
        "ppu_aiu_gemm_mixed_input_group_persistent.hpp",
        "work.m_tile", "work.n_tile", "block_shape"),
)


def code_only(text: str) -> str:
    """Remove comments and whitespace without pretending to parse C++."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def expected_tokens(site: Site) -> tuple[str, str, str]:
    return (
        f"M-size<0>({site.block_shape})*{site.m_coord}",
        f"N-size<1>({site.block_shape})*{site.n_coord}",
        "K-size<1>(gA)*size<2>(gA)",
    )


def check_site(site: Site, text: str) -> list[str]:
    compact = code_only(text)
    errors: list[str] = []
    logical_binding = f"{site.block_shape}=TileShape{{}}"
    binding_count = compact.count(logical_binding)
    if binding_count != 1:
        errors.append(
            f"{site.path}: logical blk_shape binding occurs {binding_count} times, "
            f"expected exactly 1: {logical_binding}")
    for axis, token in zip("MNK", expected_tokens(site)):
        count = compact.count(token)
        if count != 1:
            errors.append(
                f"{site.path}: {axis} logical/load residue occurs {count} times, expected exactly 1: {token}")

    forbidden = {
        "M": f"M-size<0>(gA)*{site.m_coord}",
        "N": f"N-size<0>(gB)*{site.n_coord}",
    }
    for axis, token in forbidden.items():
        if token in compact:
            errors.append(
                f"{site.path}: {axis} residue uses physical load extent: {token}")
    return errors


def discover_sites() -> set[str]:
    """Find the mixed-input kernel bodies that actually construct residues."""
    roots = (
        ROOT / "third_party/actlize/include/cutlass/gemm/kernel",
        ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel",
    )
    found: set[str] = set()
    for directory in roots:
        for path in directory.glob("*mixed_input*.hpp"):
            text = path.read_text()
            if "load_init(" in text and "residue_mnk" in text:
                found.add(path.relative_to(ROOT).as_posix())
    ordinary_grouped = ROOT / "quactlize/include/ppu_aiu_gemm_mixed_input_group.hpp"
    text = ordinary_grouped.read_text()
    if "load_init(" in text and "residue_mnk" in text:
        found.add(ordinary_grouped.relative_to(ROOT).as_posix())
    return found


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"cannot plant {label}: normalized token occurs {text.count(old)} times, expected 1")
    return text.replace(old, new, 1)


def mutation_controls(site: Site, text: str) -> list[str]:
    """Every site must reject a physical binding and physical-M/N or fake-K drift."""
    compact = code_only(text)
    m_token, n_token, k_token = expected_tokens(site)
    mutations = (
        ("physical-shape-binding", f"{site.block_shape}=TileShape{{}}",
         f"{site.block_shape}=shape(gA)"),
        ("physical-M", m_token, f"M-size<0>(gA)*{site.m_coord}"),
        ("physical-N", n_token, f"N-size<0>(gB)*{site.n_coord}"),
        ("logical-K", k_token, f"K-size<2>({site.block_shape})"),
    )
    errors: list[str] = []
    for label, old, new in mutations:
        try:
            planted = replace_once(compact, old, new, f"{site.path}:{label}")
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if not check_site(site, planted):
            errors.append(
                f"{site.path}: planted {label} residue unexpectedly passed the checker")
    return errors


def arithmetic_anchor() -> list[str]:
    """Reconstruct the exact box signature of the TM8/physical-M16 defect."""
    problem_m, problem_n = 2048, 4096
    logical_tm, physical_tm, tile_n = 8, 16, 32
    tiles_m = problem_m // logical_tm
    tiles_n = problem_n // tile_n

    # A physical-M residue becomes empty at m_tile 128; the correct logical
    # residue remains nonempty through m_tile 255.
    false_empty_m_tiles = sum(
        1 for m_tile in range(tiles_m)
        if problem_m - physical_tm * m_tile <= 0
        and problem_m - logical_tm * m_tile > 0)
    tile_elements = logical_tm * tile_n
    all_bad_elements = false_empty_m_tiles * tiles_n * tile_elements

    # The observed Stream-K raster gave one N tile to the SK prefix.  The DP
    # bucket therefore contains the other 127 N tiles for each false-empty M.
    sk_n_tiles = 1
    dp_bad_elements = false_empty_m_tiles * (tiles_n - sk_n_tiles) * tile_elements

    errors: list[str] = []
    if false_empty_m_tiles != 128:
        errors.append(f"TM8 anchor false-empty M tiles={false_empty_m_tiles}, expected 128")
    if all_bad_elements != 4_194_304:
        errors.append(
            f"TM8 anchor all affected elements={all_bad_elements}, expected 4194304")
    if dp_bad_elements != 4_161_536:
        errors.append(
            f"TM8 anchor DP nonfinite elements={dp_bad_elements}, expected 4161536")
    return errors


def main() -> int:
    errors: list[str] = []
    declared = {site.path for site in SITES}
    discovered = discover_sites()
    if discovered != declared:
        missing = sorted(discovered - declared)
        stale = sorted(declared - discovered)
        errors.append(
            "mixed-input residue coverage drift: "
            f"unregistered={missing or 'none'} stale={stale or 'none'}")

    texts: dict[str, str] = {}
    for site in SITES:
        path = ROOT / site.path
        if not path.is_file():
            errors.append(f"missing audited kernel: {site.path}")
            continue
        text = path.read_text()
        texts[site.path] = text
        errors.extend(check_site(site, text))

    errors.extend(arithmetic_anchor())

    # Run controls only after the real tree closes.  Otherwise a pre-existing
    # failure could masquerade as a planted-red witness.
    if not errors:
        for site in SITES:
            errors.extend(mutation_controls(site, texts[site.path]))

    if errors:
        print("[mixed-logical-residue] FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        "[mixed-logical-residue] PASS: 10/10 production kernels use logical M/N "
        "and loaded-A K; TM8 anchor=4194304 all/4161536 DP; "
        "40/40 planted binding/residue drifts EXPECTED-RED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
