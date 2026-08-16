#!/usr/bin/env python3
"""Pin the W4 fixed-SplitK profile selector and its exact production type edge."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "quactlize/include/ppu_dense_splitk_shipping_policy.hpp"
ORACLE = ROOT / "dev/fold_derivation/l197_dense_splitk_shipping_selector.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l197_dense_splitk_shipping_selector.sh"


POLICY_PROBE = r'''
#include "ppu_dense_splitk_shipping_policy.hpp"
#include <cstdio>
namespace s = ppu_dense_splitk_shipping;

s::Key key() {
  return {{1,4096,4096},
          {4,0,128,s::QuantSemantics::FinegrainedScaleOnly,
           s::MetadataStorage::Fp16Planes,false},
          {s::ArtifactLayout::ResidentXPlane,64,1,1,0},
          {8,64,64,8,16,2,1,true}};
}

int main() {
  int errors = 0, controls = 0;
  auto expect = [&](s::Request const& request, s::ProfileRow const* profile,
                    s::DecisionReason reason, int splits, bool parallel) {
    ++controls;
    auto selected = s::select(request, profile);
    int shipping = 0, split = 0;
    int result = s::dispatch_selected(
        selected, [&]{ ++shipping; return 1; },
        [&](int value){ ++split; return 10 + value; });
    errors += selected.reason() != reason ||
              selected.split_k_slices() != splits ||
              selected.parallel_selected() != parallel ||
              shipping != !parallel || split != parallel ||
              result != (parallel ? 10 + splits : 1);
  };
  s::Request request{key(), 0x1000, 4096u * 8u * sizeof(float)};
  s::ProfileRow profile{s::kProfileSchemaVersion, key(), 8};
  expect(request, &profile, s::DecisionReason::ProfileSelectsParallel, 8, true);
  expect(request, nullptr, s::DecisionReason::NoProfile, 1, false);
  auto stale = profile; stale.key.format.group_size = 32;
  expect(request, &stale, s::DecisionReason::StaleProfileKey, 1, false);
  auto invalid = profile; invalid.selected_s = 3;
  expect(request, &invalid, s::DecisionReason::InvalidProfileSplit, 1, false);
  auto scale_zero = request;
  scale_zero.key.format.quant = s::QuantSemantics::FinegrainedScaleZero;
  auto scale_zero_profile = profile; scale_zero_profile.key = scale_zero.key;
  expect(scale_zero, &scale_zero_profile, s::DecisionReason::UnsupportedDomain, 1, false);
  auto gs32 = request; gs32.key.format.group_size = 32;
  auto gs32_profile = profile; gs32_profile.key = gs32.key;
  expect(gs32, &gs32_profile, s::DecisionReason::UnsupportedDomain, 1, false);
  auto weak = request; weak.workspace_address += 16;
  expect(weak, &profile, s::DecisionReason::InsufficientWorkspace, 1, false);
  auto shallow = request;
  shallow.key.tactic.tile_k = 256; shallow.key.tactic.stages = 12;
  auto shallow_profile = profile; shallow_profile.key = shallow.key;
  expect(shallow, &shallow_profile, s::DecisionReason::InadmissiblePartition, 1, false);
  auto s1 = profile; s1.selected_s = 1;
  auto no_workspace = request;
  no_workspace.workspace_address = 0; no_workspace.workspace_bytes = 0;
  expect(no_workspace, &s1, s::DecisionReason::ProfileSelectsS1, 1, false);
  std::printf("policy-probe controls=%d errors=%d\n", controls, errors);
  return errors == 0 ? 0 : 1;
}
'''


def source_errors(policy: str, oracle: str, runner: str) -> list[str]:
    bad: list[str] = []
    policy_tokens = (
        "inline constexpr std::uint32_t kProfileSchemaVersion = 1;",
        "int selected_s = 1;",
        "Selection() = delete;",
        "f.quant == QuantSemantics::FinegrainedScaleOnly",
        "f.low_bits == 4 && f.high_bits == 0 && f.group_size == 128",
        "a.layout == ArtifactLayout::ResidentXPlane && a.tile_k == 64",
        "a.low_fold == 1 && a.high_fold == 1 && a.b_chunk == 0",
        "t.packed_a_rows == 1 && t.aiu_interleaved",
        "if (profile == nullptr) return {1, DecisionReason::NoProfile};",
        "if (!(profile->key == request.key))",
        "if (!is_profile_split(profile->selected_s))",
        "if (!partition_is_admissible(request.key, profile->selected_s))",
        "if (!workspace_is_admissible(request, profile->selected_s))",
        "return {profile->selected_s, DecisionReason::ProfileSelectsParallel};",
        "return std::invoke(std::forward<ShippingLaunch>(shipping_launch));",
        "using S1Gemm = typename ShippingTypes::Gemm;",
        "using S1Kernel = typename ShippingTypes::GemmKernel;",
        "typename ShippingTypes::CollectiveMainloop",
        "typename SplitTypes::CollectiveMainloop",
    )
    oracle_tokens = (
        "using Shipping = fpa_intb_ppu::DensePackedAKernelTypes<",
        "fpa_intb_ppu::QuantMode::FinegrainedScaleOnly",
        "using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, TileShape, Warp>;",
        "using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<",
        "using Contract = selector::DispatchTypeContract<Shipping, Split>;",
        "typename Prepared::ShippingGemm",
        "typename Shipping::Gemm",
        "bool prepare_selected_production(",
        "a, b, scales, nullptr, d, m, n, k, 128, 1,",
        "a, b, scales, nullptr, d, m, n, k, 128, selected_s,",
        "&prepare_selected_production<>;",
        "L197_PRODUCTION_EDGE_WITNESSED == 1",
        'domain_control("scale-zero"',
        'domain_control("gguf-q4-gs32"',
        'check("stale-semantic-key"',
        'check("invalid-s3"',
        '"production_edge=%d -> %s\\n",',
    )
    runner_tokens = (
        "l197_dense_splitk_shipping_selector.cu",
        "profile S2/S4/S8 did not exclusively select fixed Split-K",
        "fallback escaped the shipping S1 edge",
        "L197_FORCE_PRODUCTION_EDGE=1",
        "L197_SEVER_PRODUCTION_EDGE=1",
        "L197_PRODUCTION_EDGE_WITNESSED=1",
        "L197_PREPARED_INITIALIZE_EDGE_INSTANTIATED",
        "profile-parallel=3/3 fallback-s1=21/21",
    )
    for token in policy_tokens:
        if token not in policy:
            bad.append("policy missing " + token)
    for token in oracle_tokens:
        if token not in oracle:
            bad.append("oracle missing " + token)
    for token in runner_tokens:
        if token not in runner:
            bad.append("runner missing " + token)
    if policy.count("selected_s = 8"):
        bad.append("policy hard-codes S8 instead of taking a profile row")
    if oracle.count("return prepared.initialize(") != 2:
        bad.append("production binding must have exactly one S1 and one profiled-S initialize edge")
    if oracle.count("bool prepare_selected_production(") != 1:
        bad.append("production binding is missing or duplicated")
    if policy.count("using S1Gemm = typename ShippingTypes::Gemm;") != 1:
        bad.append("S1 exact Gemm authority is missing or duplicated")
    return bad


def compile_policy_probe(policy_text: str) -> tuple[int, int, str]:
    with tempfile.TemporaryDirectory(
            prefix="quactlize-l197-policy-", dir="/workspace") as td_raw:
        td = Path(td_raw)
        (td / POLICY.name).write_text(policy_text)
        source = td / "probe.cpp"
        source.write_text(POLICY_PROBE)
        binary = td / "probe"
        build = subprocess.run(
            ["c++", "-std=c++17", f"-I{td}",
             f"-I{ROOT / 'quactlize/include'}", str(source), "-o", str(binary)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60)
        if build.returncode:
            return build.returncode, -1, build.stdout
        run = subprocess.run([str(binary)], cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=30)
        return 0, run.returncode, run.stdout


def replace_one(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"cannot plant {label}: anchor count={text.count(old)}")
    return text.replace(old, new, 1)


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in (POLICY, ORACLE, RUNNER)
               if not path.is_file()]
    if missing:
        print("[dense-splitk-shipping-selector] FAIL missing: " + ", ".join(missing))
        return 1

    policy = POLICY.read_text()
    oracle = ORACLE.read_text()
    runner = RUNNER.read_text()
    bad = source_errors(policy, oracle, runner)
    if bad:
        print("[dense-splitk-shipping-selector] FAIL: " + "; ".join(bad))
        return 1

    with tempfile.TemporaryDirectory(
            prefix="quactlize-l197-run-", dir="/workspace") as td:
        env = os.environ.copy()
        env["QUACTLIZE_L197_OUT"] = td
        positive = subprocess.run(
            ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    required = (
        "[l197] PASS controls=24 shipping_calls=21 parallel_calls=3 ",
        "[l197:runner] PASS type=production-W4-ScaleOnly-gs128-xplane64 ",
    )
    if positive.returncode or any(token not in positive.stdout for token in required):
        print("[dense-splitk-shipping-selector] FAIL L197:\n" + positive.stdout[-12000:])
        return 1

    build_rc, run_rc, output = compile_policy_probe(policy)
    if build_rc or run_rc or "policy-probe controls=9 errors=0" not in output:
        print("[dense-splitk-shipping-selector] FAIL policy probe:\n" + output[-6000:])
        return 1

    runtime_plants = (
        ("absent-profile",
         "if (profile == nullptr) return {1, DecisionReason::NoProfile};",
         "if (profile == nullptr) return {8, DecisionReason::ProfileSelectsParallel};"),
        ("stale-key",
         "if (!(profile->key == request.key)) {",
         "if (false) {"),
        ("invalid-s",
         "split_k_slices == 4 || split_k_slices == 8;",
         "split_k_slices == 3 || split_k_slices == 4 || split_k_slices == 8;"),
        ("scalezero-domain",
         "f.quant == QuantSemantics::FinegrainedScaleOnly &&",
         "true &&"),
        ("gs32-domain",
         "f.low_bits == 4 && f.high_bits == 0 && f.group_size == 128 &&",
         "f.low_bits == 4 && f.high_bits == 0 &&"),
        ("workspace-alignment",
         "kMeasuredWorkspaceAlignment = 128;",
         "kMeasuredWorkspaceAlignment = 16;"),
        ("partition-depth",
         "if (!partition_is_admissible(request.key, profile->selected_s)) {",
         "if (false) {"),
        ("workspace-size",
         "if (!workspace_is_admissible(request, profile->selected_s)) {",
         "if (false) {"),
        ("s1-dispatch-edge",
         "return std::invoke(std::forward<ShippingLaunch>(shipping_launch));",
         "return std::invoke(std::forward<ParallelLaunch>(parallel_launch), 2);"),
        ("profile-s1",
         "if (profile->selected_s == 1) {",
         "if (false) {"),
    )
    runtime_red = 0
    try:
        for label, old, new in runtime_plants:
            planted = replace_one(policy, old, new, label)
            build_rc, run_rc, output = compile_policy_probe(planted)
            if build_rc:
                print(f"[dense-splitk-shipping-selector] FAIL plant {label} did not compile:\n" +
                      output[-4000:])
                return 1
            if run_rc == 0:
                print(f"[dense-splitk-shipping-selector] FAIL plant escaped: {label}")
                return 1
            runtime_red += 1
    except RuntimeError as exc:
        print(f"[dense-splitk-shipping-selector] FAIL {exc}")
        return 1

    source_plants = (
        ("remove-s1-production-edge", "oracle",
         '''        return prepared.initialize(
            a, b, scales, nullptr, d, m, n, k, 128, 1,
            workspace, workspace_bytes, stream);''',
         "        return false;"),
        ("s1-production-count", "oracle",
         "a, b, scales, nullptr, d, m, n, k, 128, 1,",
         "a, b, scales, nullptr, d, m, n, k, 128, 2,"),
        ("parallel-production-count", "oracle",
         "a, b, scales, nullptr, d, m, n, k, 128, selected_s,",
         "a, b, scales, nullptr, d, m, n, k, 128, 1,"),
        ("s1-type-authority", "policy",
         "using S1Gemm = typename ShippingTypes::Gemm;",
         "using S1Gemm = typename SplitTypes::Gemm;"),
        ("mainloop-identity", "policy",
         "typename ShippingTypes::CollectiveMainloop,\n                               typename SplitTypes::CollectiveMainloop",
         "typename SplitTypes::CollectiveMainloop,\n                               typename SplitTypes::CollectiveMainloop"),
        ("forgeable-selection", "policy",
         "Selection() = delete;", "Selection() = default;"),
    )
    source_red = 0
    try:
        for label, owner, old, new in source_plants:
            planted_policy, planted_oracle = policy, oracle
            if owner == "policy":
                planted_policy = replace_one(policy, old, new, label)
            else:
                planted_oracle = replace_one(oracle, old, new, label)
            if not source_errors(planted_policy, planted_oracle, runner):
                print(f"[dense-splitk-shipping-selector] FAIL source plant escaped: {label}")
                return 1
            source_red += 1
    except RuntimeError as exc:
        print(f"[dense-splitk-shipping-selector] FAIL {exc}")
        return 1

    print(
        "[dense-splitk-shipping-selector] PASS production-type=L197 "
        "profile={1,2,4,8} default=S1 exact-domain=M1/W4/ScaleOnly/gs128/xplane64 "
        f"runtime-plants={runtime_red}/{len(runtime_plants)}_RED "
        f"source-plants={source_red}/{len(source_plants)}_RED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
