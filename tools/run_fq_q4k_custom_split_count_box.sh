#!/usr/bin/env bash
# One-box single-contract closure for the rare Q4_K m8 Split-K operand-delivery
# failure. The default closure runs the three remaining orthogonal seams beside
# one uncontaminated baseline; numeric failures never truncate the bundle.
#
# Every binary contains the exact same generated AP0/AP1 tactic pair and uses
# GemmUniversalMixedInputSplitKParallel for S=1/2/4. The previous eight-arm
# factorial excluded the earlier source-level scheduling seams. Each selected
# candidate changes one variable only. `asm-memory-contract` keeps
# the same x4 swizzle load and adds compiler-visible shared-memory effects to
# the actual fp16-A producer, Q4-B producer, scalar packed-A/metadata producer,
# commit/wait, and common consumer asm statements.
# `logical-x2-scalar` keeps the baseline producer/wait contract and replaces
# only the physical x4 consumer with the two exact semantic shared loads.
# `async-shared-fence` is the actual issuer/wait publication test: it leaves
# issuer ownership unchanged and adds one PPU async-proxy fence after each
# wait, before the existing CTA publication barrier.  PPU_PACKED_SPLIT_GROUPS
# is deliberately absent because it changes metadata decode cadence, not A/B
# issuer or waiter ownership.
# `single-aiu-issuer` restores the opaque PPU0010 AIU copy atom's one-logical-
# thread contract at the two mixed-input helper overloads. It changes no CuTe
# coordinate, descriptor, packed-A scalar copy, wait/barrier, MMA or store.
# `exact-metadata-publication` is the causal closure: its baseline reconstructs
# the old all-thread modulo replay with one compile-time must-red define, while
# the candidate is the production exact-owner path.  Both use the same source,
# generated tactics, custom kernel, inputs and S=1/2/4 denominator.
# `stable-k-tile-shape` is a lifetime-hygiene diagnostic, not a causal repair
# candidate for this incident. CuTe's iterator retains Shape const& while the
# baseline binds a temporary returned by shape<2>(gA), but this dense kernel's
# exact K coordinate is rank-1: ForwardCoordIterator increments it without
# reading the retained shape. A clean result therefore means cadence/codegen
# sensitivity, not that the dangling reference supplied a wrong K coordinate.
#
# `FQ_A_STAGE_CANDIDATE=repeat-state` instead builds only the baseline binary
# and runs reuse/control-poison/target-poison for S2 and S4. Every repeat checks
# every FP32 partial plane; the disjoint control has the same memset/sync
# cadence as the live-workspace poison arm.
set -uo pipefail

fail() {
  printf '[fq-a-stage-root] FAIL: %s\n' "$*" >&2
  return 2
}

main() {
  local root workspace_root sha short stamp out jobs repeats attempts candidate
  local full generated arm defs build_dir build_log binary log rc attempt
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-a-stage-root-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) fail "OUT must resolve below /workspace: $out"; return $? ;;
  esac
  [ ! -e "$out" ] || { fail "refusing to overwrite $out"; return $?; }
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the arm-set identity'
    return $?
  fi
  jobs="${JOBS:-16}"
  repeats="${PROBE_REPEATS:-32768}"
  attempts="${PROBE_ATTEMPTS:-2}"
  candidate="${FQ_A_STAGE_CANDIDATE:-remaining-three}"
  case "$candidate" in
    asm-memory-contract|logical-x2-scalar|async-shared-fence|single-aiu-issuer|stable-k-tile-shape|exact-metadata-publication|remaining-two|remaining-three|repeat-state) ;;
    *)
      fail "FQ_A_STAGE_CANDIDATE must be asm-memory-contract, logical-x2-scalar, async-shared-fence, single-aiu-issuer, stable-k-tile-shape, exact-metadata-publication, remaining-two, remaining-three, or repeat-state"
      return $?
      ;;
  esac
  case "$jobs:$repeats:$attempts" in
    *[!0-9:]*|0:*|*:0:*|*:*:0)
      fail 'JOBS, PROBE_REPEATS and PROBE_ATTEMPTS must be positive integers'
      return $?
      ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" \
    "$out/results" || return 2

  python3 -B "$root/ci/check_fq_splitk_partial_path.py" || return 2

  # Fail closed if a diagnostic macro escapes its intended source seam, or if
  # the host route starts changing a device type instead of selecting the
  # already-instantiated custom type.
  python3 -B - "$root" <<'PY' || return 2
import pathlib, sys
root = pathlib.Path(sys.argv[1])
bench = (root / "benchmarks/fully_quantized_splitk_producer_bench.hpp").read_text()
main = (root / "benchmarks/test_fully_quantized_internal_sweep.cu").read_text()
partition = (root / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
    "ppu_fixed_splitk_partition.hpp")).read_text()
kernel = (root / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
    "ppu_aiu_gemm_mixed_input_splitk_parallel.hpp")).read_text()
partial_layout = (root / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/"
    "ppu_splitk_partial_layout.hpp")).read_text()
schedule = (root / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "ppu_mixed_a_schedule.hpp")).read_text()
pipeline = (root / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "ppu_mixed_pipeline.hpp")).read_text()
collective = (root / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "quactlize_mma_mixed_input.hpp")).read_text()
m8_copy = (root / (
    "third_party/actlize/include/cute/arch/"
    "copy_ppu0010_aiu.hpp")).read_text()
copy_async = (root / (
    "third_party/actlize/include/cute/arch/"
    "copy_ppu.hpp")).read_text()
packed_metadata_async = copy_async.split(
    "struct PPU_CP_ASYNC_CACHEGLOBAL", 1)[1].split(
        "struct PPU_CP_ASYNC_CACHEALWAYS_ZFILL", 1)[0]
copy_aiu = (root / (
    "third_party/actlize/include/cute/algorithm/"
    "ppu_copy.hpp")).read_text()
copy_aiu_traits = (root / (
    "third_party/actlize/include/cute/atom/"
    "copy_traits_ppu0010_aiu.hpp")).read_text()
tensor_impl = (root / (
    "third_party/actlize/include/cute/tensor_impl.hpp")).read_text()
coord_iterator = (root / (
    "third_party/actlize/include/cute/stride.hpp")).read_text()
splitk_coord_iterator = (root / (
    "third_party/actlize/include/cute/ppu_stride.hpp")).read_text()
tensor_shape_mode = tensor_impl.split(
    "// Return the shape of a mode", 1)[1].split(
        "// Return the stride of a mode", 1)[0]
forward_coord_type = coord_iterator.split(
    "struct ForwardCoordIterator\n{", 1)[1].split(
        "// A forward iterator for a coordinate that starts", 1)[0]
splitk_coord_type = splitk_coord_iterator.split(
    "struct SplitkCoordIterator\n{", 1)[1].split(
        "template <class Shape>", 1)[0]
nontrans_x4_impl = m8_copy.split(
    "struct PPU0010_TSM_LD_SWZL_IMPL<Element, false>", 1)[1].split(
        "struct PPU0010_TSM_LD_SWZL_IMPL<Element, true>", 1)[0]
q4_bulk_producer = m8_copy.split(
    "sizeof_bits<Element>::value == 4", 1)[1].split(
        "sizeof_bits<Element>::value == 2", 1)[0]
m8_wrapper = m8_copy.split("struct PPU0010_TSM_LD_SWZL_M8", 1)[1]
aiu_overloads = [section.split("\ntemplate <", 1)[0]
                 for section in copy_aiu.split("\ncopy_aiu(\n")[1:]]
if len(aiu_overloads) != 3:
    raise SystemExit("copy_aiu overload denominator changed")
pair_aiu, fa_lvalue_aiu, single_rvalue_aiu = aiu_overloads
hggc_aiu = pair_aiu.split(
    "#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100", 1)[1].split(
        "#else\n  if constexpr (SplitAIU)", 1)[0]
alternate_aiu = pair_aiu.split(
    "#else\n  if constexpr (SplitAIU)", 1)[1].rsplit("#endif", 1)[0]
aiu_load_traits = copy_aiu_traits.split(
    "struct Copy_Traits<PPU0010_AIU_LOAD", 1)[1].split(
        "template <typename Element", 1)[0]
packed_decode = collective.split(
    "packed_decode_stage(Storage& storage", 1)[1].split(
        "private:", 1)[0]
checks = {
    "one host branch": bench.count(
        "if (splits == 1 && !options.force_custom_splitk_s1)") == 1,
    "route/scope uses only": bench.count(
        "splits == 1 && !options.force_custom_splitk_s1") == 2,
    "one diagnostic workspace branch": bench.count(
        "options.force_custom_splitk_s1 && splits == 1") == 1,
    "no device macro": "FORCE_CUSTOM_SPLITK_S1" not in bench + main,
    "cli exact": main.count("--force-custom-splitk-s1") == 2,
    "runtime split descriptor": "args.split_k_slices" in kernel,
    "exact stable K-tile shape lifetime seam":
        kernel.count("PPU_SPLITK_STABLE_K_TILE_SHAPE") == 2 and
        kernel.count("auto const k_tile_shape = shape<2>(gA);") == 1 and
        kernel.count("rank(decltype(k_tile_shape){}) == 1") == 1 and
        kernel.count("idx2crd(work.k_begin, k_tile_shape), k_tile_shape") == 1 and
        kernel.count("idx2crd(work.k_begin, shape<2>(gA)), shape<2>(gA)") == 1,
    "CuTe tensor shape is a value retained by iterator reference":
        "auto\nshape(" in tensor_shape_mode and
        "decltype(auto)\nshape(" not in tensor_shape_mode and
        forward_coord_type.count("Shape const& shape;") == 1 and
        splitk_coord_type.count("Shape const& shape;") == 1,
    "S1 admitted by descriptor":
        "splits == 1 || splits == 2 || splits == 4 || splits == 8" in partition,
    "singleton plane keeps physical stride":
        "make_compact_fp32_partial_stride<PartialStride>(in.m, in.n)" in bench and
        "cute::get<2>(stride) = rows * columns;" in partial_layout,
    "partial oracle has one definition plus exact ordinary/every-repeat calls":
        bench.count("inspect_failed_partials(") == 3,
    "repeat-state diagnostic is host-only":
        main.count("--check-partials-each-repeat") == 2 and
        main.count("--repeat-state=") == 2 and
        bench.count("RepeatState::TargetPoison") == 1 and
        bench.count("RepeatState::ControlPoison") == 2 and
        "repeat_state" not in kernel + collective,
    "prior factorial retained":
        schedule.count("PPU_MIXED_A_PREPARE_AFTER_CONSUME") == 4 and
        collective.count("PPU_MIXED_A_EXPLICIT_STAGE_VIEW") == 4 and
        collective.count("PPU_PACKED_A_COMPILER_MEMORY_FENCE") == 12 and
        collective.count("PPU_PACKED_A_SYNCHRONOUS_STORE") == 2 and
        collective.count("PPU_PACKED_A_BEFORE_B") == 2 and
        collective.count("PPU_PACKED_A_SEPARATE_ASYNC_GROUP") == 2 and
        m8_copy.count("PPU_M8_DIRECT_X4_PROJECTION") == 2,
    "reserved logical-x2 scalar seam":
        m8_copy.count("PPU_M8_LOGICAL_X2_SCALAR_LOAD") == 2 and
        m8_copy.count("ppu.ld.shared.u32 %0, [%2]") == 1 and
        m8_copy.count("ppu.ld.shared.u32 %1, [%3]") == 1,
    "exact asm memory contract seam":
        m8_copy.count("PPU_PACKED_A_ASM_MEMORY_CONTRACT") == 6 and
        copy_async.count("PPU_PACKED_A_ASM_MEMORY_CONTRACT") == 8 and
        packed_metadata_async.count("PPU_PACKED_A_ASM_MEMORY_CONTRACT") == 2 and
        packed_metadata_async.count(': "memory"') == 1,
    "exact async shared visibility seam":
        pipeline.count("PPU_MIXED_ASYNC_SHARED_FENCE") == 2 and
        pipeline.count("mixed_async_shared_visibility_fence();") == 2 and
        pipeline.count("cutlass::arch::fence_view_async_shared();") == 1,
    "PPU0010 ordinary A/B have one common issuer":
        hggc_aiu.count("if (warp_idx == 0)") == 1 and
        hggc_aiu.count("copy(copy_policy_a, src_a, dst_a);") == 1 and
        hggc_aiu.count("copy(copy_policy_b, src_b, dst_b);") == 1 and
        "warp_idx == 1" not in hggc_aiu,
    "warp-split A/B is alternate architecture only":
        alternate_aiu.count("warp_idx == 0") == 2 and
        alternate_aiu.count("warp_idx == 1") == 1,
    "opaque AIU atom declares one logical issuer":
        aiu_load_traits.count("using ThrID   = Layout<_1>;") == 1 and
        "issued by ONE" in aiu_load_traits,
    "single physical AIU issuer seam is exact":
        copy_aiu.count("PPU_AIU_SINGLE_LOGICAL_ISSUER") == 4 and
        pair_aiu.count("PPU_AIU_SINGLE_LOGICAL_ISSUER") == 2 and
        single_rvalue_aiu.count("PPU_AIU_SINGLE_LOGICAL_ISSUER") == 2 and
        "PPU_AIU_SINGLE_LOGICAL_ISSUER" not in fa_lvalue_aiu and
        pair_aiu.count(
            "if (warp_idx == 0 && int(threadIdx.x) == 0)") == 1 and
        single_rvalue_aiu.count(
            "if (warp_idx == 0 && int(threadIdx.x) == 0)") == 1,
    "split-groups is metadata-decode-only":
        collective.count("PPU_PACKED_SPLIT_GROUPS") == 3 and
        packed_decode.count("PPU_PACKED_SPLIT_GROUPS") == 2 and
        "copy_aiu(" not in packed_decode,
    "contract covers Q4 producer and original nontrans x4 consumer":
        nontrans_x4_impl.count("PPU_PACKED_A_ASM_MEMORY_CONTRACT") == 2 and
        nontrans_x4_impl.count("m8n8.x4.swzl.shared.b16") == 2 and
        nontrans_x4_impl.count(': "memory"') == 1 and
        q4_bulk_producer.count("PPU_PACKED_A_ASM_MEMORY_CONTRACT") == 2 and
        q4_bulk_producer.count(': "memory"') == 2 and
        "PPU_PACKED_A_ASM_MEMORY_CONTRACT" not in m8_wrapper,
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    raise SystemExit("custom-S1 source seam changed: " + repr(bad))
print("[fq-a-stage-root:source] PASS CuTe tensor shape<I>() returns by value "
      "while both K iterators retain Shape const&; custom baseline binds a "
      "temporary and stable-k-tile-shape names the same dynamic value, but "
      "the exact dense K coordinate is rank-1 so that lifetime seam is not "
      "causal by itself; "
      "PPU0010 single-issuer, asm-memory, logical-x2, async-shared-fence and "
      "exact metadata-publication seams remain exact; S is runtime data")
PY

  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --tile-m-filter 8 \
    --per-unit 1 --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"

  # Materialize exactly the frozen AP0/AP1 pair.  The unit files still come
  # from the shipping generator; this script creates no second tactic source.
  python3 -B - "$full" "$generated" <<'PY' || return 2
import json, pathlib, shutil, sys
source = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
manifest_path = source / "manifest.json"
manifest = json.loads(manifest_path.read_text())
expected = {
  "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
  "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
}
identity = manifest.get("identity", {})
if (identity.get("qtype"), identity.get("artifact_tile_k"),
        identity.get("bchunk")) != (12, 64, 0):
    raise SystemExit("generated identity is not q12/A64/bc0")
rows = manifest.get("typed_rows", [])
units = manifest.get("units", [])
if len(rows) != manifest.get("denominator", {}).get("typed_rows"):
    raise SystemExit("typed denominator differs")
if len(units) != len(rows):
    raise SystemExit("selector requires --per-unit=1 authority")
selected = [(i, row) for i, row in enumerate(rows)
            if row.get("symbol") in expected]
got = {row["symbol"] for _, row in selected}
if got != expected or len(selected) != 2:
    raise SystemExit(
        f"exact row denominator changed missing={sorted(expected-got)} "
        f"extra={sorted(got-expected)}")
selected.sort(key=lambda item: item[1]["symbol"])
output.mkdir(parents=True, exist_ok=True)
unit_dir = output / "units"
unit_dir.mkdir(parents=True, exist_ok=True)
copied, selected_rows = [], []
for ordinal, (index, row) in enumerate(selected):
    axes = (row.get("tile_m"), row.get("tile_n"),
            row.get("tactic_tile_k"), row.get("warp_m"),
            row.get("warp_n"), row.get("stages"))
    provider = row.get("a_provider")
    if axes != (8, 64, 256, 8, 16, 2) or provider not in {
            "standard-aiu", "packed-row"}:
        raise SystemExit(f"symbol/axis contradiction: {row}")
    src = pathlib.Path(units[index]).resolve()
    if not src.is_file() or src.is_symlink() or source not in src.parents:
        raise SystemExit(f"generated unit escaped source authority: {src}")
    dst = unit_dir / f"fq_custom_split_count_{ordinal:02d}.cu"
    shutil.copy2(src, dst)
    copied.append(str(dst.resolve()))
    selected_rows.append(row)
lines = ["#define FQ_TC_REGISTRY_ROWS(X) \\"]
for index, row in enumerate(selected_rows):
    ap = 1 if row["a_provider"] == "packed-row" else 0
    suffix = " \\" if index + 1 < len(selected_rows) else ""
    lines.append(
        f"  X({row['symbol']},{row['qtype']},{row['artifact_tile_k']},"
        f"{row['tile_m']},{row['tile_n']},{row['tactic_tile_k']},"
        f"{row['warp_m']},{row['warp_n']},{row['stages']},"
        f"{row['bchunk']},{ap}){suffix}")
registry = (
    "// GENERATED -- exact custom Split-K runtime-count closure.\n"
    "#define FQ_TC_GENERATED_QTYPE 12\n"
    "#define FQ_TC_GENERATED_ARTIFACT_TK 64\n"
    "#define FQ_TC_GENERATED_BCHUNK 0\n"
    "#define FQ_TC_GENERATED_RAW_ROWS 2\n"
    "#define FQ_TC_GENERATED_TYPED_ROWS 2\n" + "\n".join(lines) + "\n")
(output / "fq_tc_registry.inc").write_text(registry)
(output / "units.cmake").write_text(
    "# GENERATED -- exact custom Split-K runtime-count closure.\n"
    "set(FQ_TC_GENERATED_UNIT_SOURCES\n" +
    "".join(f'  "{path}"\n' for path in copied) + ")\n" +
    f'set(FQ_TC_GENERATED_REGISTRY "{(output / "fq_tc_registry.inc").resolve()}")\n' +
    f'set(FQ_TC_GENERATED_MANIFEST "{(output / "manifest.json").resolve()}")\n')
closure = {
    "schema": "quactlize.fq-a-stage-root-closure.v1",
    "source_manifest": str(manifest_path),
    "source_typed_denominator": len(rows),
    "selection_denominator": 2,
    "identity": identity,
    "typed_rows": selected_rows,
    "units": copied,
}
(output / "manifest.json").write_text(
    json.dumps(closure, indent=2, sort_keys=True) + "\n")
print(f"[fq-a-stage-root:select] PASS source_typed={len(rows)} selected=2")
PY

  : >"$out/results/infrastructure.tsv"
  local selected_arms
  if [ "$candidate" = repeat-state ]; then
    selected_arms="baseline"
  elif [ "$candidate" = remaining-three ]; then
    selected_arms="baseline asm-memory-contract logical-x2-scalar async-shared-fence"
  elif [ "$candidate" = remaining-two ]; then
    selected_arms="baseline logical-x2-scalar async-shared-fence"
  else
    selected_arms="baseline $candidate"
  fi
  for arm in $selected_arms; do
    case "$arm" in
      baseline)
        if [ "$candidate" = exact-metadata-publication ]; then
          defs='PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS=1'
        else
          defs=""
        fi
        ;;
      asm-memory-contract) defs='PPU_PACKED_A_ASM_MEMORY_CONTRACT=1' ;;
      logical-x2-scalar) defs='PPU_M8_LOGICAL_X2_SCALAR_LOAD=1' ;;
      async-shared-fence) defs='PPU_MIXED_ASYNC_SHARED_FENCE=1' ;;
      single-aiu-issuer) defs='PPU_AIU_SINGLE_LOGICAL_ISSUER=1' ;;
      stable-k-tile-shape) defs='PPU_SPLITK_STABLE_K_TILE_SHAPE=1' ;;
      exact-metadata-publication) defs='' ;;
      *) fail "internal arm table error: $arm"; return $? ;;
    esac
    build_dir="$out/build-$arm"
    build_log="$out/results/$arm-build.log"
    mkdir -p "$build_dir" || return 2
    (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 \
      JOBS="$jobs" PPU_DEFS="$defs" \
      TARGET=test_fully_quantized_internal_sweep \
      FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
      FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 \
      FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) >"$build_log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      tail -160 "$build_log" >&2
      printf '%s\tbuild\t%d\n' "$arm" "$rc" \
        >>"$out/results/infrastructure.tsv"
      continue
    fi
    if [ -n "$defs" ]; then
      if ! grep -Fq -- "-D$defs" "$build_log"; then
        printf '%s\tdefine-missing\t2\n' "$arm" \
          >>"$out/results/infrastructure.tsv"
        continue
      fi
    elif grep -Eq -- '-DPPU_(PACKED_A_ASM_MEMORY_CONTRACT|M8_LOGICAL_X2_SCALAR_LOAD|MIXED_ASYNC_SHARED_FENCE|AIU_SINGLE_LOGICAL_ISSUER|SPLITK_STABLE_K_TILE_SHAPE|MIXED_LEGACY_MODULO_METADATA_PUBLISHERS)=1' \
        "$build_log"; then
      printf '%s\tbaseline-contaminated\t2\n' "$arm" \
        >>"$out/results/infrastructure.tsv"
      continue
    fi
    binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
    [ -x "$binary" ] && [ ! -L "$binary" ] || {
      printf '%s\tbinary-missing\t2\n' "$arm" \
        >>"$out/results/infrastructure.tsv"
      continue
    }
    printf 'FQ_A_STAGE_ROOT_BINARY arm=%s defs=%s sha256=%s attempts=%s repeats=%s\n' \
      "$arm" "${defs:-NONE}" "$(sha256sum "$binary" | awk '{print $1}')" \
      "$attempts" "$repeats"
    if [ "$candidate" = repeat-state ]; then
      local mode split
      for mode in reuse control-poison target-poison; do
        for split in 2 4; do
          attempt=1
          while [ "$attempt" -le "$attempts" ]; do
            log="$out/results/repeat-state-$mode-s$split-attempt$attempt.log"
            "$binary" --shape=1x1024x5120 --iterations=1 \
              --correctness-repeats="$repeats" --tm8-max-m=8 \
              --bc-mode=skip --only-split="$split" \
              --force-custom-splitk-s1 --correctness-only \
              --check-partials-each-repeat --repeat-state="$mode" \
              >"$log" 2>&1
            rc=$?
            # A numeric mismatch is the evidence sought by this diagnostic.
            # Keep all six arms running and reserve rc=2 for infrastructure.
            if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
              tail -100 "$log" >&2
              printf '%s-s%d\trun-attempt-%d\t%d\n' \
                "$mode" "$split" "$attempt" "$rc" \
                >>"$out/results/infrastructure.tsv"
            fi
            printf 'FQ_REPEAT_STATE_EXECUTION mode=%s S=%d attempt=%d rc=%d\n' \
              "$mode" "$split" "$attempt" "$rc"
            attempt=$((attempt + 1))
          done
        done
      done
    else
      attempt=1
      while [ "$attempt" -le "$attempts" ]; do
        log="$out/results/$arm-attempt$attempt.log"
        "$binary" --shape=1x1024x5120 --iterations=1 \
          --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
          --force-custom-splitk-s1 >"$log" 2>&1
        rc=$?
        # Numeric rc=1 is evidence, not a runner failure. Keep sampling every
        # arm/attempt; only record infrastructure here and adjudicate once all
        # arms have had their chance to run.
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
          tail -100 "$log" >&2
          printf '%s\trun-attempt-%d\t%d\n' "$arm" "$attempt" "$rc" \
            >>"$out/results/infrastructure.tsv"
        fi
        printf 'FQ_A_STAGE_ROOT_EXECUTION arm=%s attempt=%d rc=%d\n' \
          "$arm" "$attempt" "$rc"
        if [ "$arm" = baseline ]; then
          local shipping_log shipping_rc
          shipping_log="$out/results/shipping-s1-attempt$attempt.log"
          "$binary" --shape=1x1024x5120 --iterations=1 \
            --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
            --only-split=1 --correctness-only >"$shipping_log" 2>&1
          shipping_rc=$?
          if [ "$shipping_rc" -ne 0 ] && [ "$shipping_rc" -ne 1 ]; then
            tail -100 "$shipping_log" >&2
            printf 'shipping-s1\trun-attempt-%d\t%d\n' \
              "$attempt" "$shipping_rc" \
              >>"$out/results/infrastructure.tsv"
          fi
          printf 'FQ_A_STAGE_ROOT_SHIPPING_EXECUTION attempt=%d rc=%d\n' \
            "$attempt" "$shipping_rc"
        fi
        attempt=$((attempt + 1))
      done
    fi
  done

  if [ "$candidate" = repeat-state ]; then
    python3 -B - "$out/results" "$attempts" "$repeats" <<'PY' \
        | tee "$out/results/verdict.log"
import pathlib, shlex, sys

result_dir = pathlib.Path(sys.argv[1])
attempts = int(sys.argv[2])
repeats = int(sys.argv[3])
modes = ("reuse", "control-poison", "target-poison")
providers = {
    "standard-aiu": "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "packed-row": "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
}

def fields(line):
    row = {}
    for token in shlex.split(line.removeprefix("FQ_CUSTOM_SPLIT_COUNT_CELL ")):
        if "=" not in token:
            raise ValueError(f"malformed token {token!r}")
        key, value = token.split("=", 1)
        if key in row:
            raise ValueError(f"duplicate field {key}")
        row[key] = value
    return row

def parse(path, mode, split):
    text = path.read_text()
    marker = (f"FQ_REPEAT_STATE mode={mode} "
              "partial_check_each_repeat=1 measure=0")
    if text.count(marker) != 1:
        raise ValueError(f"{path}: repeat-state marker denominator differs")
    rows = [fields(line) for line in text.splitlines()
            if line.startswith("FQ_CUSTOM_SPLIT_COUNT_CELL ")]
    expected = {(provider, s) for provider in providers for s in (1, 2, 4)}
    keyed = {(row.get("provider"), int(row.get("S", "0"))): row
             for row in rows}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError(f"{path}: custom cell denominator differs")
    selected = {}
    for provider, symbol in providers.items():
        row = keyed[(provider, split)]
        if row.get("symbol") != symbol:
            raise ValueError(f"{path}: symbol/provider contradiction")
        if row.get("kernel") != "GemmUniversalMixedInputSplitKParallel":
            raise ValueError(f"{path}: custom producer route changed")
        if row.get("partial_probe") != "COMPLETE":
            raise ValueError(f"{path}: every-repeat partial oracle did not complete")
        selected[provider] = row
    return selected

def is_clean(row):
    return (row["state"] == "MEASURED" and row["failure_step"] == "NONE" and
            int(row["raw_bad"]) == 0 and
            int(row["partial_value_raw_bad"]) == 0 and
            int(row["failure_repeat"]) == -1)

def is_numeric(row):
    return (row["state"] in {"RAW_FP16_MISMATCH", "PARTIAL_FP32_MISMATCH"} and
            (int(row["raw_bad"]) > 0 or
             int(row["partial_value_raw_bad"]) > 0) and
            int(row["failure_repeat"]) >= 0)

try:
    infrastructure = (result_dir / "infrastructure.tsv").read_text().strip()
    if infrastructure:
        raise ValueError("infrastructure failures: " +
                         infrastructure.replace("\n", ";"))
    runs = {}
    for mode in modes:
        for split in (2, 4):
            for attempt in range(1, attempts + 1):
                path = result_dir / (
                    f"repeat-state-{mode}-s{split}-attempt{attempt}.log")
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"missing regular log: {path}")
                runs[(mode, split, attempt)] = parse(path, mode, split)

    stats = {}
    for mode in modes:
        failures = 0
        exposure = 0
        output_bad = 0
        latent_partial_bad = 0
        cells = 0
        for split in (2, 4):
            for attempt in range(1, attempts + 1):
                for provider, row in runs[(mode, split, attempt)].items():
                    cells += 1
                    if is_clean(row):
                        exposure += repeats
                    elif is_numeric(row):
                        failures += 1
                        exposure += int(row["failure_repeat"]) + 1
                        output_bad += int(row["raw_bad"]) > 0
                        latent_partial_bad += (
                            int(row["raw_bad"]) == 0 and
                            int(row["partial_value_raw_bad"]) > 0)
                        print(
                            "FQ_REPEAT_STATE_FAILURE "
                            f"mode={mode} provider={provider} S={split} "
                            f"attempt={attempt} repeat={row['failure_repeat']} "
                            f"state={row['state']} raw_bad={row['raw_bad']} "
                            f"partial_value_raw_bad={row['partial_value_raw_bad']} "
                            f"plane_mask={row['partial_bad_plane_mask']} "
                            f"partial_first_index={row['partial_first_bad_index']} "
                            f"partial_first_want={row['partial_first_bad_want']} "
                            f"partial_first_got={row['partial_first_bad_got']}")
                    else:
                        raise ValueError(
                            f"{mode}/{provider}/S{split}: unexpected row {row}")
        rate = failures / exposure if exposure else 0.0
        stats[mode] = (failures, exposure, rate)
        print(
            "FQ_REPEAT_STATE_ARM "
            f"mode={mode} cells={cells} failures={failures} "
            f"exposure={exposure} mle_per_repeat={rate:.12g} "
            f"output_bad={output_bad} latent_partial_bad={latent_partial_bad}")

    reuse_bad = stats["reuse"][0] > 0
    control_bad = stats["control-poison"][0] > 0
    target_bad = stats["target-poison"][0] > 0
    if not reuse_bad:
        verdict = "BASELINE_NONREPRODUCTION"
        interpretation = (
            "fresh-process reuse did not reproduce; no state or cadence cause "
            "can be assigned")
    elif control_bad and not target_bad:
        verdict = "PARTIAL_WORKSPACE_CARRYOVER_CAUSAL"
        interpretation = (
            "equal memset/synchronize cadence stayed dirty only when the live "
            "partial workspace was not overwritten")
    elif control_bad and target_bad:
        verdict = "PARTIAL_WORKSPACE_REUSE_EXCLUDED"
        interpretation = (
            "both equal-cadence arms remained dirty; prior partial workspace "
            "contents are not required for the failure")
    elif not control_bad and not target_bad:
        verdict = "MEMSET_CADENCE_MASKS_FAILURE"
        interpretation = (
            "both equal-cadence arms became clean; clearing the target cannot be "
            "assigned causally because the disjoint control also masked the race")
    else:
        verdict = "TARGET_POISON_ONLY_DIRTY_UNADJUDICATED"
        interpretation = (
            "only poisoning the live workspace remained dirty; poison exposure or "
            "missing producer stores needs a dedicated follow-up")
    print(
        "FQ_REPEAT_STATE_VERDICT "
        f"verdict={verdict} attempts={attempts} repeats={repeats} "
        "same_binary=1 fresh_process_per_arm=1 every_partial_checked=1")
    print("FQ_REPEAT_STATE_INTERPRETATION " + interpretation)
except (KeyError, OSError, ValueError) as error:
    print(f"[fq-repeat-state] FAIL: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
    rc=${PIPESTATUS[0]}
    find "$out/results" -maxdepth 1 -type f \
        ! -name authority.sha256 -print0 | sort -z | \
      xargs -0 sha256sum >"$out/results/authority.sha256" || return 2
    if [ "$rc" -ne 0 ]; then
      fail "repeat-state analysis failed artifacts=$out"
      return "$rc"
    fi
    printf '[fq-repeat-state] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' \
      "$sha" "$out"
    return 0
  fi

  python3 -B - "$out/results" "$attempts" "$repeats" "$candidate" <<'PY' \
      | tee "$out/results/verdict.log"
import pathlib, shlex, sys

result_dir = pathlib.Path(sys.argv[1])
attempts = int(sys.argv[2])
repeats = int(sys.argv[3])
selection = sys.argv[4]
if selection not in {"asm-memory-contract", "logical-x2-scalar",
                     "async-shared-fence", "single-aiu-issuer",
                     "stable-k-tile-shape", "exact-metadata-publication", "remaining-two",
                     "remaining-three"}:
    raise SystemExit("invalid candidate identity")
candidates = (("asm-memory-contract", "logical-x2-scalar",
               "async-shared-fence")
              if selection == "remaining-three" else
              (("logical-x2-scalar", "async-shared-fence")
               if selection == "remaining-two" else (selection,)))
arms = ("baseline",) + candidates
expected_symbols = {
    "standard-aiu": "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "packed-row": "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
}
expected_keys = {(provider, split)
                 for provider in expected_symbols for split in (1, 2, 4)}

def parse(path):
    text = pathlib.Path(path).read_text()
    marker = "FQ_CUSTOM_SPLIT_COUNT_PROBE "
    if text.count(marker) != 1:
        raise ValueError(f"{path}: route marker denominator differs")
    rows = {}
    for line in text.splitlines():
        if not line.startswith("FQ_CUSTOM_SPLIT_COUNT_CELL "):
            continue
        fields = {}
        for token in shlex.split(line.removeprefix(
                "FQ_CUSTOM_SPLIT_COUNT_CELL ")):
            if "=" not in token:
                raise ValueError(f"{path}: malformed token {token!r}")
            key, value = token.split("=", 1)
            if key in fields:
                raise ValueError(f"{path}: duplicate field {key}")
            fields[key] = value
        key = (fields.get("provider"), int(fields.get("S", "0")))
        if key in rows:
            raise ValueError(f"{path}: duplicate cell {key}")
        rows[key] = fields
    if set(rows) != expected_keys:
        raise ValueError(
            f"{path}: cell denominator differs "
            f"missing={sorted(expected_keys-set(rows))} "
            f"extra={sorted(set(rows)-expected_keys)}")
    for (provider, split), fields in rows.items():
        if fields.get("symbol") != expected_symbols[provider]:
            raise ValueError(f"{path}: symbol/provider contradiction")
        if fields.get("kernel") != "GemmUniversalMixedInputSplitKParallel":
            raise ValueError(f"{path}: S={split} did not use custom kernel")
        if int(fields.get("partial_bytes", "-1")) != split * 1024 * 4:
            raise ValueError(f"{path}: S={split} workspace differs")
    return rows

def parse_shipping(path):
    text = pathlib.Path(path).read_text()
    if "FQ_CUSTOM_SPLIT_COUNT_PROBE " in text:
        raise ValueError(f"{path}: shipping control used the custom route")
    rows = {}
    for line in text.splitlines():
        if not line.startswith("FQ_TC_CELL "):
            continue
        fields = {}
        for token in shlex.split(line.removeprefix("FQ_TC_CELL ")):
            if "=" not in token:
                raise ValueError(f"{path}: malformed shipping token {token!r}")
            key, value = token.split("=", 1)
            if key in fields:
                raise ValueError(f"{path}: duplicate shipping field {key}")
            fields[key] = value
        if fields.get("S") != "1":
            continue
        provider = fields.get("provider")
        if provider in rows:
            raise ValueError(f"{path}: duplicate shipping provider {provider}")
        rows[provider] = fields
    if set(rows) != set(expected_symbols):
        raise ValueError(
            f"{path}: shipping provider denominator differs "
            f"missing={sorted(set(expected_symbols)-set(rows))} "
            f"extra={sorted(set(rows)-set(expected_symbols))}")
    for provider, fields in rows.items():
        if fields.get("symbol") != expected_symbols[provider] or \
                fields.get("scope") != "FULL_OUTPUT" or \
                int(fields.get("partial_bytes", "-1")) != 0:
            raise ValueError(f"{path}: shipping S1 identity differs for {provider}")
    return rows

def clean(row):
    return (row["state"] == "MEASURED" and int(row["raw_bad"]) == 0 and
            row["failure_step"] == "NONE")

def corrupt(row):
    return (row["state"] == "RAW_FP16_MISMATCH" and
            int(row["raw_bad"]) > 0 and
            "RAW_FP16_MISMATCH" in row["failure_step"])

def s1_inadmissible(row, split):
    return (split == 1 and row["state"] == "REAL_CAN_IMPLEMENT" and
            int(row["raw_bad"]) == 0 and row["failure_step"] == "NONE")

def localization(row):
    if not corrupt(row):
        return "NOT_APPLICABLE"
    if row.get("partial_probe") != "COMPLETE":
        return "FAILURE_NOT_SNAPSHOTTED"
    if int(row.get("partial_value_raw_bad", "-1")) > 0:
        return "PRODUCER_PARTIAL_VALUE_BAD"
    if int(row.get("reducer_replay_raw_bad", "-1")) == 0:
        return "SAME_STREAM_PUBLICATION_GAP"
    return "REDUCER_REPLAY_STILL_BAD"

def producer_partial_bad(row):
    return (corrupt(row) and row.get("partial_probe") == "COMPLETE" and
            int(row.get("partial_value_raw_bad", "-1")) > 0 and
            int(row.get("reducer_replay_raw_bad", "-1")) > 0)

def frozen_ap1_s4_incident(row):
    # ca01dc6: output fp16 26 -> 31, caused by producer plane 2 carrying
    # FP32 6.0 instead of 1.0 across one 32-output stripe.  The stripe can
    # land on any physical N tile, so its aligned index is evidence to print,
    # not part of the mechanism identity.  Symbol/provider/S and fixture are
    # already frozen by the surrounding denominator.
    output_index = int(row.get("first_bad", "-1"))
    partial_index = int(row.get("partial_first_bad_index", "-2"))
    return (producer_partial_bad(row) and
            row.get("provider") == "packed-row" and row.get("S") == "4" and
            int(row.get("raw_bad", "-1")) == 32 and
            output_index == partial_index and
            0 <= output_index < 1024 and output_index % 32 == 0 and
            row.get("first_want") == "0x4e80" and
            row.get("first_got") == "0x4fc0" and
            int(row.get("partial_bad_plane_mask", "-1"), 0) == 0x4 and
            int(row.get("partial_first_bad_plane", "-1")) == 2 and
            row.get("partial_first_bad_want") == "0x3f800000" and
            row.get("partial_first_bad_got") == "0x40c00000")

try:
    infrastructure = (result_dir / "infrastructure.tsv").read_text().strip()
    if infrastructure:
        raise ValueError("one or more arm infrastructure failures: " +
                         infrastructure.replace("\n", ";"))

    runs = {}
    for arm in arms:
        for attempt in range(1, attempts + 1):
            path = result_dir / f"{arm}-attempt{attempt}.log"
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing regular arm log: {path}")
            runs[(arm, attempt)] = parse(path)

    shipping_runs = {}
    for attempt in range(1, attempts + 1):
        path = result_dir / f"shipping-s1-attempt{attempt}.log"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular shipping control log: {path}")
        shipping_runs[attempt] = parse_shipping(path)

    for (arm, attempt), rows in runs.items():
        bad_states = [
            f"{provider}:S{split}:{row['state']}"
            for (provider, split), row in rows.items()
            if not (clean(row) or corrupt(row) or
                    s1_inadmissible(row, split))]
        if bad_states:
            raise ValueError(f"{arm}/attempt{attempt} contains infrastructure/non-numeric "
                             f"states: {','.join(sorted(bad_states))}")
        for provider in expected_symbols:
            for split in (1, 2, 4):
                row = rows[(provider, split)]
                if corrupt(row):
                    print(
                        "FQ_A_STAGE_ROOT_FAILURE "
                        f"arm={arm} attempt={attempt} provider={provider} S={split} "
                        f"failure_repeat={row['failure_repeat']} "
                        f"raw_bad={row['raw_bad']} first_bad={row['first_bad']} "
                        f"first_want={row['first_want']} first_got={row['first_got']} "
                        f"partial_value_raw_bad={row.get('partial_value_raw_bad', 'MISSING')} "
                        f"partial_bad_plane_mask={row.get('partial_bad_plane_mask', 'MISSING')} "
                        f"partial_first_bad_plane={row.get('partial_first_bad_plane', 'MISSING')} "
                        f"partial_first_bad_index={row.get('partial_first_bad_index', 'MISSING')} "
                        f"partial_first_bad_want={row.get('partial_first_bad_want', 'MISSING')} "
                        f"partial_first_bad_got={row.get('partial_first_bad_got', 'MISSING')} "
                        f"frozen_incident={int(frozen_ap1_s4_incident(row))} "
                        f"localization={localization(row)}")

    candidate_clean = {}
    denominator = attempts * len(expected_symbols) * 2
    s1_denominator = attempts * len(expected_symbols)
    for arm in arms:
        s1_cells = [runs[(arm, attempt)][(provider, 1)]
                    for attempt in range(1, attempts + 1)
                    for provider in expected_symbols]
        s1_clean_count = sum(clean(row) for row in s1_cells)
        s1_inadmissible_count = sum(s1_inadmissible(row, 1)
                                    for row in s1_cells)
        s1_corrupt_count = sum(corrupt(row) for row in s1_cells)
        if (s1_clean_count + s1_inadmissible_count + s1_corrupt_count !=
                s1_denominator):
            raise ValueError(f"{arm}: custom-S1 census is not closed")
        print(
            "FQ_A_STAGE_ROOT_S1 "
            f"arm={arm} attempts={attempts} repeats={repeats} "
            f"clean={s1_clean_count}/{s1_denominator} "
            f"inadmissible={s1_inadmissible_count} "
            f"corrupt={s1_corrupt_count}")

        cells = [runs[(arm, attempt)][(provider, split)]
                 for attempt in range(1, attempts + 1)
                 for provider in expected_symbols for split in (2, 4)]
        clean_count = sum(clean(row) for row in cells)
        producer_bad_count = sum(producer_partial_bad(row) for row in cells)
        other_bad_count = sum(corrupt(row) and not producer_partial_bad(row)
                              for row in cells)
        # A candidate is a closure only if it fixes every stressed S2/S4 cell
        # and keeps the same custom producer/reducer route clean at S1. This
        # prevents an S-dependent cadence change from merely moving the bug.
        is_clean = (clean_count == denominator and
                    s1_clean_count == s1_denominator)
        if arm in candidates:
            candidate_clean[arm] = is_clean
        print(
            "FQ_A_STAGE_ROOT_ARM "
            f"arm={arm} attempts={attempts} repeats={repeats} "
            f"s_gt1_clean={clean_count}/{denominator} "
            f"producer_partial_bad={producer_bad_count} "
            f"other_bad={other_bad_count} "
            f"all_s_clean={int(is_clean)} clean={int(is_clean)}")

    shipping_cells = [shipping_runs[attempt][provider]
                      for attempt in range(1, attempts + 1)
                      for provider in expected_symbols]
    shipping_clean_count = sum(clean(row) for row in shipping_cells)
    shipping_corrupt_count = sum(corrupt(row) for row in shipping_cells)
    if shipping_clean_count + shipping_corrupt_count != len(shipping_cells):
        raise ValueError("shipping S1 control contains infrastructure/non-numeric states")
    print(
        "FQ_A_STAGE_ROOT_SHIPPING_S1 "
        f"attempts={attempts} repeats={repeats} "
        f"clean={shipping_clean_count}/{len(shipping_cells)} "
        f"corrupt={shipping_corrupt_count} "
        "kernel=GemmUniversal-SplitKSerialScheduler")

    baseline_exact_fingerprint = any(
        frozen_ap1_s4_incident(
            runs[("baseline", attempt)][("packed-row", 4)])
        for attempt in range(1, attempts + 1))
    # Causality is assigned to the incident family, not to one historical
    # output index/value pair. Require all four independent S>1 cells to
    # reproduce producer-partial corruption at least once; the exact old
    # AP1/S4 1.0->6.0 fingerprint remains a diagnostic field only.
    baseline_target_cells = sum(
        any(producer_partial_bad(
                runs[("baseline", attempt)][(provider, split)])
            for attempt in range(1, attempts + 1))
        for provider in expected_symbols for split in (2, 4))
    baseline_failure_events = sum(
        producer_partial_bad(
            runs[("baseline", attempt)][(provider, split)])
        for attempt in range(1, attempts + 1)
        for provider in expected_symbols for split in (2, 4))
    baseline_failure_attempts = sum(
        any(producer_partial_bad(
                runs[("baseline", attempt)][(provider, split)])
            for provider in expected_symbols for split in (2, 4))
        for attempt in range(1, attempts + 1))
    # Earlier multi-seam closures preregistered all four AP0/AP1 x S2/S4
    # cells. The repeat-state audit subsequently froze the reduced incident:
    # the corrupt cell migrates with code layout while every baseline attempt
    # still contains a producer-partial failure. One-variable issuer/fence/
    # lifetime arms therefore use that attempt denominator; combined legacy
    # factorials retain the historical four-cell rule.
    reduced_incident_selection = candidates in {
        ("async-shared-fence",), ("single-aiu-issuer",),
        ("stable-k-tile-shape",), ("exact-metadata-publication",)}
    baseline_target = (
        baseline_failure_attempts == attempts and
        baseline_failure_events >= attempts
        if reduced_incident_selection else baseline_target_cells == 4)
    baseline_s1 = [runs[("baseline", attempt)][(provider, 1)]
                   for attempt in range(1, attempts + 1)
                   for provider in expected_symbols]
    baseline_s1_clean = all(clean(row) for row in baseline_s1)
    baseline_s1_corrupt = any(corrupt(row) for row in baseline_s1)
    baseline_s1_inadmissible = all(s1_inadmissible(row, 1)
                                   for row in baseline_s1)
    shipping_all_clean = shipping_clean_count == len(shipping_cells)
    shipping_any_corrupt = shipping_corrupt_count != 0
    if baseline_s1_corrupt and shipping_all_clean:
        failure_scope = "CUSTOM_WRAPPER_OR_DIRECT_PARTIAL_NOT_SHIPPING_S1"
    elif baseline_s1_corrupt and shipping_any_corrupt:
        failure_scope = "CUSTOM_AND_SHIPPING_S1_COMMON_MAINLOOP"
    elif baseline_s1_corrupt:
        failure_scope = "CUSTOM_S1_WITH_INCOMPLETE_SHIPPING_CONTROL"
    elif baseline_s1_clean:
        failure_scope = "S_GT1_ONLY_IN_THIS_DENOMINATOR"
    elif baseline_s1_inadmissible:
        failure_scope = "S1_UNEXECUTED"
    else:
        failure_scope = "S1_MIXED_OR_INCOMPLETE"
    clean_candidates = [arm for arm in candidates if candidate_clean[arm]]
    if not baseline_target:
        if reduced_incident_selection:
            verdict = "BASELINE_EVERY_ATTEMPT_NONREPRODUCTION"
            interpretation = (
                "at least one independent baseline attempt did not reproduce "
                "producer-partial corruption in any AP0/AP1 S2/S4 cell; a clean "
                "single-issuer candidate cannot be assigned causally")
        else:
            verdict = "BASELINE_FOUR_CELL_NONREPRODUCTION"
            interpretation = (
                "the complete AP0/AP1 x S2/S4 baseline denominator did not "
                "reproduce producer-partial corruption; a clean candidate cannot "
                "be assigned causally")
        diagnostic_rc = 1
    elif not clean_candidates:
        if candidates == ("asm-memory-contract",):
            verdict = "INLINE_ASM_SHARED_MEMORY_CONTRACT_REMAINS_DIRTY"
            interpretation = (
                "the frozen specialization reproduced producer-partial corruption, "
                "but the x4-preserving exact asm memory contract did not close every "
                "stressed S1/S2/S4 cell; the contract defect is real but is not a "
                "complete device root")
        elif candidates == ("logical-x2-scalar",):
            verdict = "M8_LOGICAL_X2_SCALAR_REMAINS_DIRTY"
            interpretation = (
                "the frozen specialization reproduced producer-partial corruption, "
                "but removing the physical x4 swizzle consumer did not close every "
                "stressed S1/S2/S4 cell; the x4 opcode is not a complete device root")
        elif candidates == ("async-shared-fence",):
            verdict = "ASYNC_SHARED_VISIBILITY_FENCE_REMAINS_DIRTY"
            interpretation = (
                "the baseline reproduced, but an explicit async-shared proxy "
                "fence before packed decode/CTA publication did not close every "
                "stressed S1/S2/S4 cell; missing proxy visibility is not the root")
        elif candidates == ("single-aiu-issuer",):
            verdict = "AIU_SINGLE_LOGICAL_ISSUER_REMAINS_DIRTY"
            interpretation = (
                "the baseline reproduced producer-partial corruption in every "
                "attempt, but reducing each opaque AIU copy from 32 identical "
                "warp0 callers to its one declared logical issuer did not close "
                "every stressed S1/S2/S4 cell")
        elif candidates == ("stable-k-tile-shape",):
            verdict = "K_TILE_ITERATOR_SHAPE_LIFETIME_REMAINS_DIRTY"
            interpretation = (
                "the baseline reproduced producer-partial corruption in every "
                "attempt, but extending the dynamic K-tile shape lifetime across "
                "the reference-holding CuTe iterator did not close every stressed "
                "S1/S2/S4 cell")
        elif candidates == ("exact-metadata-publication",):
            verdict = "EXACT_METADATA_PUBLICATION_REMAINS_DIRTY"
            interpretation = (
                "the legacy modulo-publisher negative reproduced producer-partial "
                "corruption in every attempt, but exact scale/zero/raw ownership "
                "plus the one-time packed initialization edge did not close every "
                "stressed S1/S2/S4 cell")
        else:
            verdict = ("ALL_REMAINING_COMMON_SEAMS_DIRTY"
                       if candidates == ("asm-memory-contract",
                                         "logical-x2-scalar",
                                         "async-shared-fence")
                       else "BOTH_REMAINING_COMMON_SEAMS_DIRTY")
            interpretation = (
                "the baseline reproduced, but neither removing the physical m8 "
                "x4 reader, completing the Q4 operand asm-memory contract, nor "
                "adding the exact async-shared visibility fence closed every "
                "stressed S1/S2/S4 cell")
        diagnostic_rc = 1
    elif len(clean_candidates) > 1:
        verdict = "MULTIPLE_ORTHOGONAL_CLOSURES_UNADJUDICATED"
        interpretation = (
            "more than one orthogonal arm closed; code-layout/cadence sensitivity "
            "prevents assigning a unique source cause")
        diagnostic_rc = 1
    else:
        winner = clean_candidates[0]
        if winner == "asm-memory-contract":
            verdict = "INLINE_ASM_SHARED_MEMORY_CONTRACT_CAUSAL"
            interpretation = (
                "the baseline reproduced and the physical x4 opcode, packed bytes, "
                "stage geometry, schedule and MMA order remained unchanged; "
                "declaring the fp16-A, Q4-B and scalar async producers plus "
                "commit/wait and ldmatrix shared-memory effects closed every "
                "independent S1/S2/S4 cell")
        elif winner == "logical-x2-scalar":
            verdict = "M8_X4_SWIZZLE_OPCODE_OR_SCOREBOARD_CAUSAL"
            interpretation = (
                "the baseline reproduced while the packed bytes, stage geometry, "
                "producer/wait contract, logical fragment, schedule and MMA order "
                "remained unchanged; replacing only the physical x4 consumer with "
                "its two exact semantic shared loads closed every S1/S2/S4 cell")
        elif winner == "single-aiu-issuer":
            verdict = "AIU_DUPLICATE_PHYSICAL_ISSUE_CAUSAL"
            interpretation = (
                "the baseline reproduced producer-partial corruption in every "
                "attempt while CuTe coordinates, AIU descriptors, packed-A scalar "
                "copies, waits, barriers, MMA and stores remained unchanged; "
                "matching the PPU0010 opaque copy atom's Layout<_1> contract by "
                "reducing 32 identical warp0 callers to one physical issuer closed "
                "every S1/S2/S4 cell")
        elif winner == "stable-k-tile-shape":
            verdict = "K_TILE_ITERATOR_RANK1_CADENCE_SENSITIVE"
            interpretation = (
                "the baseline reproduced producer-partial corruption in every "
                "attempt and extending the K-shape lifetime changed the failure "
                "rate, but this exact K iterator is rank-1 and its in-range "
                "increment does not read Shape; the clean arm is a code-layout "
                "sensitivity result and cannot assign the producer root")
            diagnostic_rc = 1
        elif winner == "exact-metadata-publication":
            verdict = "PACKED_METADATA_CLEAR_DECODE_RACE_CLOSED"
            interpretation = (
                "the same source and tactic pair reproduced with the historical "
                "all-thread modulo publishers and missing initialization edge, "
                "while production exact ownership plus the one-time pre-prefetch "
                "CTA edge closed every custom S1/S2/S4 cell; the repaired contract "
                "covers scale/zero initialization, ordinary metadata and packed "
                "raw copies")
        else:
            verdict = "ASYNC_SHARED_VISIBILITY_FENCE_CAUSAL"
            interpretation = (
                "the baseline reproduced while issuer sets, copy groups, stage "
                "geometry, CTA barrier and MMA order remained unchanged; an "
                "explicit async-shared proxy fence alone closed every S1/S2/S4 cell")
        if winner != "stable-k-tile-shape":
            diagnostic_rc = 0

    print("FQ_A_STAGE_ROOT_VERDICT "
          f"verdict={verdict} baseline_target={int(baseline_target)} "
          f"baseline_target_cells={baseline_target_cells}/4 "
          f"baseline_failure_events={baseline_failure_events} "
          f"baseline_failure_attempts={baseline_failure_attempts}/{attempts} "
          f"baseline_exact_fingerprint={int(baseline_exact_fingerprint)} "
          f"selection={selection} candidates={','.join(candidates)} "
          f"clean_candidates={','.join(clean_candidates) or 'NONE'} "
          f"failure_scope={failure_scope} "
          f"shipping_s1_clean={shipping_clean_count}/{len(shipping_cells)}")
    print("FQ_A_STAGE_ROOT_INTERPRETATION " + interpretation)
    raise SystemExit(diagnostic_rc)
except (KeyError, OSError, ValueError) as error:
    print(f"[fq-a-stage-root] FAIL: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
  rc=${PIPESTATUS[0]}

  find "$out/results" -maxdepth 1 -type f \
      ! -name authority.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >"$out/results/authority.sha256" || return 2
  if [ "$rc" -ne 0 ]; then
    fail "unadjudicated device result artifacts=$out"
    return "$rc"
  fi
  printf '[fq-a-stage-root] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' "$sha" "$out"
}

main "$@"
