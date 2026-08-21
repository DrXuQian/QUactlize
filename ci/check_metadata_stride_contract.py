#!/usr/bin/env python3
"""Contract for the caller-declared S/Z metadata stride.

Three shipping collectives used to accept dS in Arguments, drop it before
Params, and rebuild one compact layout in load_init.  L127 supplies the
constructive semantic proof; this checker prevents any one collective from
silently returning to that shape-only ABI.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"
COLLECTIVES = (
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp",
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp",
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp",
)
ORACLE = ROOT / "dev/fold_derivation/l127_metadata_stride.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l127_metadata_stride.sh"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def audit(texts: list[str]) -> list[str]:
    helper, ordinary, folded, two_plane, oracle, runner = texts
    bad: list[str] = []
    helper_flat = compact(helper)
    signature = (
        "template<classScaleTileShape,classElement,classStride>"
        "CUTE_HOST_DEVICEautomake_metadata_tile("
        "Elementconst*base,Strideconst&dS,intN,int64_tscale_k,intL,"
        "intl_coord,intn_coord)"
    )
    if helper_flat.count(signature) != 1:
        bad.append("metadata helper does not expose exactly one caller stride seam")
    if helper_flat.count(
            "CUTE_HOST_DEVICEconstexprStridelower_metadata_stride(Strideconst&dS){returndS;}") != 1:
        bad.append("metadata lowering does not preserve the caller stride through the shared seam")
    if helper_flat.count(
            "cute::make_gmem_ptr(base),cute::make_shape(N,scale_k,L),dS") != 1:
        bad.append("metadata helper does not apply dS to the CuTe tensor")
    if "make_tight_metadata_tile" in helper:
        bad.append("obsolete shape-only metadata helper remains callable")

    for label, source in zip(("ordinary", "fold", "two-plane"),
                             (ordinary, folded, two_plane)):
        flat = compact(source)
        if flat.count("NonVoidStrideScaledS{};") != 2:
            bad.append(f"{label} must carry dS once in Arguments and once in Params")
        if flat.count("p.dS=detail::lower_metadata_stride(args.dS);") != 1:
            bad.append(f"{label} does not lower caller dS exactly once")
        call_prefix = "detail::make_metadata_tile<ScaleTileShape>("
        if flat.count(call_prefix) != 2:
            bad.append(f"{label} must route both S and Z through the shared helper")
        if flat.count("mainloop_params.ptr_S,mainloop_params.dS,") != 1 or \
                flat.count("mainloop_params.ptr_Z,mainloop_params.dS,") != 1:
            bad.append(f"{label} S/Z do not consume the exact lowered dS")
        for stale in (
            "make_tensor(make_gmem_ptr(mainloop_params.ptr_S),make_shape(N,scale_k,L))",
            "make_tensor(make_gmem_ptr(mainloop_params.ptr_Z),make_shape(N,scale_k,L))",
        ):
            if stale in flat:
                bad.append(f"{label} still rebuilds compact metadata via {stale}")

    oracle_flat = compact(oracle)
    for token in (
        "kStridedMetadataTileApi==2",
        "kExperts=256",
        "kPaddedGroupStride=kN+8",
        "kPaddedExpertStride=kPaddedGroupStride*kScaleK+8",
        "md::make_metadata_tile<ScaleTile>",
        "tight_expected=int64_t(n)+",
        "padded_expected=int64_t(n)+",
        "source[std::size_t(i)]=payload(i)",
        "changed_addresses==kExpectedChanged",
        "changed_values==kExpectedChanged",
        "ignored_stride_mismatches==kExpectedChanged",
        "explicit-int64-stride-formula+unique-physical-offset-tags",
        "scope=S-and-Z-shared-stride",
        "md::lower_metadata_stride(padded)",
        "get<1>(lowered)",
        "get<2>(lowered)",
    ):
        if token not in oracle_flat:
            bad.append(f"L127 lost load-bearing token {token!r}")
    if "nvcc-std=c++17-xcu-arch=sm_80" not in compact(runner):
        bad.append("L127 runner no longer compiles the host CuTe oracle")
    return bad


def main() -> int:
    paths = (HELPER, *COLLECTIVES, ORACLE, RUNNER)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        print("[metadata-stride-contract] FAIL: missing " + ", ".join(missing))
        return 1
    texts = [path.read_text() for path in paths]
    bad = audit(texts)
    if bad:
        print("[metadata-stride-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (0, "cute::make_shape(N, scale_k, L), dS);",
         "cute::make_shape(N, scale_k, L));", "helper drops dS"),
        (1, "p.dS = detail::lower_metadata_stride(args.dS);",
         "/* planted dS drop */", "ordinary lowering"),
        (2, "mainloop_params.ptr_Z, mainloop_params.dS,",
         "mainloop_params.ptr_Z, NonVoidStrideScale{},", "fold Z substitutes a default"),
        (3, "NonVoidStrideScale dS{};", "/* planted Params/Arguments hole */",
         "two-plane dS field"),
        (4, "ignored_stride_mismatches == kExpectedChanged",
         "ignored_stride_mismatches >= 0", "constructive ignored-dS negative"),
    )
    for index, old, new, label in plants:
        planted = list(texts)
        if old not in planted[index]:
            print(f"[metadata-stride-contract] FAIL: cannot plant {label}")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(planted):
            print(f"[metadata-stride-contract] FAIL: checker accepted planted {label}")
            return 1

    run = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    required = (
        "lowering=shared-production-seam Arguments.dS=(1,40,328) Params.dS=(1,40,328) -> PASS",
        "changed_addr=65504 changed_value=65504 expected_changed=65504 -> PASS",
        "ignored-dS-mismatches=65504 expected=65504 -> EXPECTED-RED",
        "caller-X=dS implementation-Y=dS result=PASS scope=S-and-Z-shared-stride",
    )
    if run.returncode != 0:
        print(f"[metadata-stride-contract] FAIL: L127 rc={run.returncode}: {run.stdout[-1200:]}")
        return 1
    missing_output = [token for token in required if token not in run.stdout]
    if missing_output:
        print("[metadata-stride-contract] FAIL: L127 output missing " + repr(missing_output))
        return 1
    print("[metadata-stride-contract] PASS: three collectives consume dS; "
          "explicit compact/padded anchors and five source plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
