#!/usr/bin/env python3
"""Bind L139 to the production standalone-Marlin type and classic formulas."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def require(text: str, token: str, where: str, failures: list[str]) -> None:
    if token not in text:
        failures.append(f"{where}: missing {token!r}")


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle",
        type=pathlib.Path,
        default=repo / "dev/fold_derivation/l139_marlin_warpk_reduce.cu",
    )
    parser.add_argument(
        "--collective",
        type=pathlib.Path,
        default=repo
        / "quactlize/include/quactlize_extensions/cutlass/gemm/collective"
        / "marlin_collective_ppu.hpp",
    )
    parser.add_argument(
        "--kernel",
        type=pathlib.Path,
        default=repo
        / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel"
        / "marlin_kernel_ppu.hpp",
    )
    parser.add_argument(
        "--classic",
        type=pathlib.Path,
        default=repo.parent / "marlin_classic_ppu.cuh",
    )
    args = parser.parse_args()

    failures: list[str] = []
    files: dict[str, str] = {}
    for name, path in {
        "oracle": args.oracle,
        "collective": args.collective,
        "kernel": args.kernel,
        "classic": args.classic,
    }.items():
        try:
            files[name] = path.read_text(encoding="utf-8")
        except OSError as error:
            failures.append(f"{name}: cannot read {path}: {error}")
            files[name] = ""

    oracle = files["oracle"]
    collective = files["collective"]
    kernel = compact(files["kernel"])
    classic = compact(files["classic"])

    for token in (
        "using ProductionStrideA = Stride<int64_t, _1, int64_t>;",
        "using ProductionStrideB = Stride<int64_t, _1, int64_t>;",
        "using ProductionStrideScale = Stride<_1, int64_t, int64_t>;",
        "using ProductionCollective = cutlass::gemm::collective::MarlinCollectivePPU<",
        "ProductionStrideA, ProductionStrideB, ProductionStrideScale>;",
        "using ProductionMma = typename ProductionCollective::TiledMma;",
        "ProductionMma{}.get_thr_layout_vmnk()",
        "ProductionMma{}.get_thread_slice(thread).partition_C(identity)",
        "classic_acc_i",
        "classic_acc_j",
        "generic-layout",
        "compact-layout",
        "production_tree",
    ):
        require(oracle, token, "oracle", failures)
    for forbidden in (
        "quactlize_mma_builder.inl",
        "quactlize_detail::get_tiled_mma",
        "typename Built::TiledMma",
    ):
        if forbidden in oracle:
            failures.append(
                f"oracle: retired generic-builder evidence returned: {forbidden!r}"
            )

    for token in (
        "class MarlinCollectivePPU",
        "using TiledMma = cute::TiledMMA<",
        "cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_4>>",
        "cute::Tile<cute::_16, cute::_32, cute::_64>",
    ):
        require(collective, token, "collective", failures)

    for token in (
        "usingTiledMma=typenameCollectiveMainloop::TiledMma;",
        "returnlane/4+(((value>>2)&1)<<3);",
        "returnlane%4+((value%4)<<2);",
        "constexprintred_off=2;",
        "for(intstep=red_off;step>0;step/=2)",
        "intconstchunk=2*n_block+half;",
        "intconstvalue_base=4*half;",
        "accum.fragments[n_block].value[value_base+i]+=peer[i]+prior[i];",
        "accum.fragments[n_block].value[value_base+i]+=peer[i];",
    ):
        require(kernel, token, "kernel", failures)

    for token in (
        "intacc_i(intlane,intl){return(lane/4)+(((l>>2)&1)<<3);}",
        "intacc_j(intlane,intl){return(lane%4)+((l%4)<<2);}",
        "for(inti=red_off;i>0;i/=2)",
        "frag_c[m_block][jj][l0+k]+=c_rd[k]+c_wr[k];",
        "frag_c[m_block][i/2][4*(i%2)+j]+=c_rd[j];",
    ):
        require(classic, token, "classic", failures)

    if failures:
        for failure in failures:
            print(f"[l139:source] FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "[l139:source] PASS: oracle aliases MarlinCollectivePPU::TiledMma; "
        "production and classic acc_i/acc_j + 4->2->1 cadence are anchored; "
        "retired generic builder absent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
