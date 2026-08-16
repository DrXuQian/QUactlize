#!/usr/bin/env python3
"""Pin the real C-ABI-to-kernel W4 fixed Split-K production edge."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ABI = ROOT / "quactlize/include/quactlize_ppu_device.h"
LAUNCH = ROOT / "quactlize/include/ppu_dense_w4_splitk_launch.cuh"
BACKEND = ROOT / "quactlize/csrc/device/ppu_dense_backend.cu"
L200 = ROOT / "dev/fold_derivation/l200_dense_w4_splitk_production.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l200_dense_w4_splitk_production.sh"
EXACT_WARM = ROOT / "benchmarks/dense_splitk_exact_warm_ab_tn64.cu"

KEY_FIELDS = (
    "rows", "columns", "inner", "low_bits", "high_bits", "group_size",
    "quant_semantics", "metadata_storage", "has_zero_plane",
    "artifact_layout", "artifact_tile_k", "artifact_low_fold",
    "artifact_high_fold", "artifact_b_chunk", "tactic_tile_m",
    "tactic_tile_n", "tactic_tile_k", "tactic_warp_m", "tactic_warp_n",
    "tactic_stages", "packed_a_rows", "aiu_interleaved",
)


def flat(text: str) -> str:
    return " ".join(text.split())


def extract_braced(text: str, marker: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"marker count for {marker!r} is {text.count(marker)}")
    start = text.index(marker)
    brace = text.find("{", start)
    if brace < 0:
        raise ValueError(f"marker {marker!r} has no body")
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]
    raise ValueError(f"unterminated body for {marker!r}")


def source_errors(
        abi: str, launch: str, backend: str, l200: str, runner: str,
        exact_warm: str) -> list[str]:
    bad: list[str] = []
    af, lf, bf, l2f, rf = map(flat, (abi, launch, backend, l200, runner))

    try:
        key_struct = extract_braced(
            abi, "typedef struct quactlize_ppu_dense_w4_splitk_key_v1")
        observed_fields = tuple(re.findall(r"\bint32_t\s+(\w+)\s*;", key_struct))
        if observed_fields != KEY_FIELDS:
            bad.append(
                "public key is not the ordered complete 22-field identity: " +
                repr(observed_fields))
        profile_struct = flat(extract_braced(
            abi, "typedef struct quactlize_ppu_dense_w4_splitk_profile_v1"))
        for token in (
                "uint32_t schema_version;",
                "quactlize_ppu_dense_w4_splitk_key_v1 key;",
                "int32_t selected_s;"):
            if token not in profile_struct:
                bad.append("public profile missing " + token)
    except ValueError as exc:
        bad.append(str(exc))

    abi_tokens = (
        "#define QUACTLIZE_PPU_DENSE_W4_SPLITK_PROFILE_SCHEMA_V1 1",
        "#define QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ONLY 0",
        "#define QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_FP16_PLANES 0",
        "#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_RESIDENT_XPLANE 0",
        "int64_t quactlize_ppu_dense_w4_splitk_workspace_bytes_v1(",
        "int quactlize_ppu_dense_w4_splitk_dev_v1(",
        "Dense symmetric W4A16 tensor-core route",
        "GGUF Q4_K ScaleZero/gs32",
    )
    for token in abi_tokens:
        if token not in af:
            bad.append("public ABI missing " + token)
    if abi.count("quactlize_ppu_dense_w4_splitk_workspace_bytes_v1(") != 1:
        bad.append("workspace query declaration is missing or duplicated")
    if abi.count("quactlize_ppu_dense_w4_splitk_dev_v1(") != 1:
        bad.append("async launch declaration is missing or duplicated")

    launch_tokens = (
        "using ProductionSchedule = ppu_group_schedule::FinegrainedSchedule<128>;",
        "using ProductionTile = Shape<_8, _64, _128>;",
        "using ProductionScaleTile = Shape<_64, _1>;",
        "using ProductionWarp = Shape<_8, _16, _128>;",
        "1, fpa_intb_ppu::QuantMode::FinegrainedScaleOnly, ProductionSchedule, ProductionTile, ProductionScaleTile, ProductionWarp, 2, true, cutlass::int4b_t, 64>;",
        "using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher< ProductionShipping, ProductionTile, ProductionWarp>;",
        "using TypeContract = selector::DispatchTypeContract< ProductionShipping, ProductionSplit>;",
        "typename Prepared::ShippingGemm, typename ProductionShipping::Gemm",
        "typename ProductionShipping::CollectiveMainloop, typename ProductionSplit::CollectiveMainloop",
        "ProductionShipping::MainloopPolicy::TacticTileK == 128",
        "ProductionShipping::MainloopPolicy::ArtifactTileK == 64",
        "ProductionShipping::CollectiveMainloop::DispatchPolicy:: StaticGroupSize == 128",
        "{4, 0, 128, selector::QuantSemantics::FinegrainedScaleOnly, selector::MetadataStorage::Fp16Planes, false}",
        "{selector::ArtifactLayout::ResidentXPlane, 64, 1, 1, 0}",
        "{8, 64, 128, 8, 16, 2, 1, true}",
        "inline constexpr selector::Key kProductionTypeIdentity = production_key(1, 256, 256);",
        "kProductionTypeIdentity.format.group_size == ProductionShipping::CollectiveMainloop::DispatchPolicy:: StaticGroupSize",
        "ProductionDescriptor::packed_metadata ? selector::MetadataStorage::PackedUnits : selector::MetadataStorage::Fp16Planes",
        "ppu_mixed_policy::has_zero(ProductionDescriptor::quant_mode)",
        "kProductionTypeIdentity.artifact.tile_k == ProductionShipping::MainloopPolicy::ArtifactTileK",
        "kProductionTypeIdentity.artifact.b_chunk == int(ProductionDescriptor::atom_at_a_time)",
        "kProductionTypeIdentity.tactic.tile_k == int(cute::size<2>(ProductionTile{}))",
        "kProductionTypeIdentity.tactic.warp_n == int(cute::size<1>(ProductionWarp{}))",
        "kProductionTypeIdentity.tactic.stages == ProductionDescriptor::stages",
        "kProductionTypeIdentity.tactic.packed_a_rows == ProductionShipping::MainloopPolicy::PackedARows",
        "kProductionTypeIdentity.tactic.aiu_interleaved == ProductionDescriptor::interleaved",
        "decode_profile(profile, decoded) ? &decoded : nullptr;",
        "selector::Request const request{ production_key(m, n, k), workspace_address, workspace_bytes};",
        "return selector::select(request, decoded_ptr);",
        "selector::kMeasuredWorkspaceAlignment, (std::numeric_limits<std::size_t>::max)(), profile",
        "selector::required_partial_bytes( production_key(m, n, k).problem, selected.split_k_slices())",
        "#if defined(QUACTLIZE_W4_SPLITK_SEVER_PREPARE_EDGE)",
        "m, n, k, 128, 1, workspace, workspace_bytes, stream",
        "m, n, k, 128, selected_s, workspace, workspace_bytes, stream",
    )
    for token in launch_tokens:
        if token not in lf:
            bad.append("production launch authority missing " + token)
    if launch.count("return prepared.initialize(") != 2:
        bad.append("production authority must have exactly S1 and selected-S initialize edges")
    if launch.count("scales, nullptr, out") != 2:
        bad.append("ScaleOnly production edges must both pass a literal null zero plane")
    if launch.count("bool prepare_selected(") != 1:
        bad.append("production prepare authority is missing or duplicated")

    include_island = flat("""
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED 1
#include "ppu_dense_w4_splitk_launch.cuh"
#else
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED 0
#endif
""")
    if include_island not in bf:
        bad.append("W4 production type is not scoped to default/Q4 format island")
    if backend.count(
            'extern "C" int64_t quactlize_ppu_dense_w4_splitk_workspace_bytes_v1(') != 1:
        bad.append("backend workspace query definition is missing or duplicated")
    if backend.count(
            'extern "C" int quactlize_ppu_dense_w4_splitk_dev_v1(') != 1:
        bad.append("backend async launch definition is missing or duplicated")

    try:
        query = flat(extract_braced(
            backend,
            'extern "C" int64_t quactlize_ppu_dense_w4_splitk_workspace_bytes_v1('))
        launch_entry = flat(extract_braced(
            backend, 'extern "C" int quactlize_ppu_dense_w4_splitk_dev_v1('))
        for token in (
                "#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED",
                "return ppu_dense_w4_splitk::query_workspace_bytes(m, n, k, profile);",
                "#else",
                "return -1;"):
            if token not in query:
                bad.append("workspace query missing " + token)
        entry_tokens = (
            "#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED",
            "!act || !weight_xplane || !scales || !out",
            "ppu_dense_w4_splitk::problem_is_in_fixed_abi(m, n, k)",
            "reinterpret_cast<std::uintptr_t>(workspace)",
            "ppu_dense_w4_splitk::prepare_selected<>( selected, prepared",
            "reinterpret_cast<cutlass::int4b_t const*>(weight_xplane)",
            "if (prepared.run(launch_stream) != cutlass::Status::kSuccess) return 31;",
            'ppu_gemv::rt_check_launch("dense W4 fixed Split-K enqueue")',
            "#else",
            "return 33;",
        )
        for token in entry_tokens:
            if token not in launch_entry:
                bad.append("async launch missing " + token)
        if launch_entry.count("ppu_dense_w4_splitk::prepare_selected<>") != 1:
            bad.append("backend must have exactly one real production prepare call edge")
        if launch_entry.count("prepared.run(launch_stream)") != 1:
            bad.append("backend must enqueue exactly one selected prepared kernel sequence")
    except ValueError as exc:
        bad.append(str(exc))

    # The pre-existing GGUF Q4_K ABI remains a different ScaleZero/gs32
    # contract.  Pin its validation and both compile-time macro arms.
    try:
        old_q4 = flat(extract_braced(
            backend, 'extern "C" int quactlize_ppu_dense_lowbit_config_v1('))
        if "!act || !low || !scale || !zero || !out" not in old_q4:
            bad.append("GGUF dense Q4 route no longer requires its zero plane")
        q4_case = "case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32,"
        if old_q4.count(q4_case) != 2:
            bad.append("GGUF Q4_K must retain both ScaleZero gs32 macro arms")
        if "quactlize_ppu_dense_w4_splitk" in old_q4:
            bad.append("GGUF Q4_K ABI was redirected through the new fixed-W4 symbol")
    except ValueError as exc:
        bad.append(str(exc))
    gguf_tokens = (
        "using PackedKernelTypes = fpa_intb_ppu::DensePackedAKernelTypes<1, QM::FinegrainedScaleZero",
        "bool const launched = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero",
        "m, n, k, GroupSize, 1, static_cast<char*>(workspace), workspace_bytes, stream",
    )
    for token in gguf_tokens:
        if token not in bf:
            bad.append("legacy GGUF launch binding missing " + token)

    l200_tokens = (
        "sizeof(quactlize_ppu_dense_w4_splitk_key_v1) == 22 * 4",
        "sizeof(quactlize_ppu_dense_w4_splitk_profile_v1) == 24 * 4",
        "&production::prepare_selected<>;",
        "QUACTLIZE_PPU_DENSE_W4_SPLITK_PROFILE_SCHEMA_V1",
        "4, 0, 128, QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ONLY",
        "64, 1, 1, 0, 8, 64, 128, 8, 16, 2, 1, 1",
        'check("exact-s2"', 'check("exact-s4"', 'check("exact-s8"',
        'check("null-profile"', 'check("stale-schema"',
        'check("invalid-s3"', 'check("short-workspace"',
        'check("misaligned-workspace"', 'check("malformed-enum"',
        "full_key_fields=22 profile_axis={1,2,4,8}",
    )
    for token in l200_tokens:
        if token not in l2f:
            bad.append("L200 control missing " + token)
    for field in KEY_FIELDS:
        label = {
            "quant_semantics": "quant", "metadata_storage": "metadata",
            "has_zero_plane": "zero", "artifact_layout": "artifact-layout",
            "artifact_tile_k": "artifact-tk", "artifact_low_fold": "low-fold",
            "artifact_high_fold": "high-fold", "artifact_b_chunk": "bchunk",
            "tactic_tile_m": "tile-m", "tactic_tile_n": "tile-n",
            "tactic_tile_k": "tile-k", "tactic_warp_m": "warp-m",
            "tactic_warp_n": "warp-n", "tactic_stages": "stages",
            "packed_a_rows": "packed-a", "aiu_interleaved": "interleaved",
            "low_bits": "low-bits", "high_bits": "high-bits",
            "group_size": "group-size", "rows": "rows",
            "columns": "columns", "inner": "inner",
        }[field]
        if f'"key-{label}"' not in l200:
            bad.append("L200 lacks single-field stale-key negative for " + field)

    runner_tokens = (
        "-DQUACTLIZE_DENSE_ONLY=12",
        "-DQUACTLIZE_DENSE_ONLY=10",
        "Q2 island lost fail-closed ABI symbol",
        "Q2 island instantiated or referenced the W4 production type",
        "_Static_assert(sizeof(quactlize_ppu_dense_w4_splitk_key_v1) == 88",
        "L200_PRODUCTION_PREPARE_REACHED_PREPARED_INITIALIZE",
        "-DQUACTLIZE_W4_SPLITK_SEVER_PREPARE_EDGE=1",
        "real production prepare edge did not reach Prepared::initialize exactly once",
        "profile/workspace denominator is not 35 controls",
        "controls=35/35 call-edge=instantiated/severed",
    )
    for token in runner_tokens:
        if token not in rf:
            bad.append("L200 runner missing " + token)
    if "run_exact_warm_ab<8,64,128,8,16,2,0>" not in exact_warm:
        bad.append("production tactic is no longer the committed TN64 exact-warm row")
    return bad


def replace_first(text: str, old: str, new: str, label: str,
                  expected_count: int = 1) -> str:
    count = text.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"cannot plant {label}: anchor count={count}, expected={expected_count}")
    return text.replace(old, new, 1)


def main() -> int:
    paths = (ABI, LAUNCH, BACKEND, L200, RUNNER, EXACT_WARM)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        print("[dense-w4-splitk-production] FAIL missing: " + ", ".join(missing))
        return 1

    texts = {path: path.read_text(encoding="utf-8") for path in paths}
    bad = source_errors(*(texts[path] for path in paths))
    if bad:
        print("[dense-w4-splitk-production] FAIL: " + "; ".join(bad))
        return 1

    with tempfile.TemporaryDirectory(
            prefix="quactlize-l200-ci-", dir="/workspace") as td:
        env = os.environ.copy()
        env["QUACTLIZE_L200_OUT"] = td
        positive = subprocess.run(
            ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    required_output = (
        "[l200] PASS controls=35 shipping_calls=32 parallel_calls=3 ",
        "[l200:runner] PASS abi=C-v1 production=backend-W4-ScaleOnly-gs128-",
        "call-edge=instantiated/severed",
    )
    if positive.returncode or any(
            token not in positive.stdout for token in required_output):
        print("[dense-w4-splitk-production] FAIL L200:\n" +
              positive.stdout[-14000:])
        return 1

    abi, launch, backend, l200, runner, exact_warm = (
        texts[path] for path in paths)
    plants = (
        ("backend-prepare-edge", "backend",
         "ppu_dense_w4_splitk::prepare_selected<>",
         "ppu_dense_w4_splitk::prepare_missing<>", 1),
        ("literal-s1", "launch", "m, n, k, 128, 1,",
         "m, n, k, 128, 2,", 1),
        ("selected-s", "launch",
         "m, n, k, 128,\n            selected_s,",
         "m, n, k, 128,\n            1,", 1),
        ("production-quant", "launch",
         "1, fpa_intb_ppu::QuantMode::FinegrainedScaleOnly,",
         "1, fpa_intb_ppu::QuantMode::FinegrainedScaleZero,", 1),
        ("production-tactic-tk", "launch",
         "using ProductionTile = Shape<_8, _64, _128>;",
         "using ProductionTile = Shape<_8, _64, _64>;", 1),
        ("production-artifact-tk", "launch",
         "2, true, cutlass::int4b_t, 64>;",
         "2, true, cutlass::int4b_t, 128>;", 1),
        ("profile-tactic-key", "launch",
         "{8, 64, 128, 8, 16, 2, 1, true},",
         "{8, 64, 64, 8, 16, 2, 1, true},", 1),
        ("type-key-binding", "launch",
         "kProductionTypeIdentity.tactic.tile_k ==\n            int(cute::size<2>(ProductionTile{}))",
         "kProductionTypeIdentity.tactic.tile_k == 128", 1),
        ("profile-decode", "launch",
         "decode_profile(profile, decoded) ? &decoded : nullptr;",
         "profile ? &decoded : nullptr;", 1),
        ("full-key-request", "launch",
         "selector::Request const request{\n      production_key(m, n, k), workspace_address, workspace_bytes};",
         "selector::Request const request{\n      decoded.key, workspace_address, workspace_bytes};", 1),
        ("format-island", "backend",
         "#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12\n#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED 1\n#include \"ppu_dense_w4_splitk_launch.cuh\"",
         "#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10\n#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED 1\n#include \"ppu_dense_w4_splitk_launch.cuh\"", 1),
        ("disabled-query", "backend",
         "(void)m; (void)n; (void)k; (void)profile;\n  return -1;",
         "(void)m; (void)n; (void)k; (void)profile;\n  return 0;", 1),
        ("disabled-launch", "backend",
         "(void)stream; (void)profile;\n  return 33;",
         "(void)stream; (void)profile;\n  return 0;", 1),
        ("workspace-address", "backend",
         "reinterpret_cast<std::uintptr_t>(workspace)",
         "ppu_dense_splitk_shipping::kMeasuredWorkspaceAlignment", 1),
        ("prepared-run", "backend",
         "prepared.run(launch_stream)", "cutlass::Status::kSuccess", 1),
        ("gguf-zero-plane", "backend", "!scale || !zero || !out",
         "!scale || !out", 1),
        ("gguf-q4-gs32", "backend",
         "case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32,",
         "case 12: return group_size == 128 ? dense<cutlass::int4b_t,void,32,", 2),
        ("gguf-scalezero", "backend",
         "using PackedKernelTypes = fpa_intb_ppu::DensePackedAKernelTypes<1,\n          QM::FinegrainedScaleZero,",
         "using PackedKernelTypes = fpa_intb_ppu::DensePackedAKernelTypes<1,\n          QM::FinegrainedScaleOnly,", 1),
    )
    owners = {
        "abi": abi, "launch": launch, "backend": backend,
        "l200": l200, "runner": runner, "exact": exact_warm,
    }
    red = 0
    try:
        for label, owner, old, new, count in plants:
            planted = dict(owners)
            planted[owner] = replace_first(
                planted[owner], old, new, label, count)
            errors = source_errors(
                planted["abi"], planted["launch"], planted["backend"],
                planted["l200"], planted["runner"], planted["exact"])
            if not errors:
                print(f"[dense-w4-splitk-production] FAIL source plant escaped: {label}")
                return 1
            red += 1
    except RuntimeError as exc:
        print(f"[dense-w4-splitk-production] FAIL {exc}")
        return 1

    print(
        "[dense-w4-splitk-production] PASS abi=C-v1 "
        "build-scope=q12-reachable/q10-fail-closed "
        "type=W4-ScaleOnly-gs128-TK128-xplaneTK64 "
        "profile={1,2,4,8}/missing-stale-wrong-key->literal-S1 "
        "gguf-Q4=ScaleZero-gs32-unchanged "
        f"single-field-key-negatives=22/22 source-plants={red}/{len(plants)}_RED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
