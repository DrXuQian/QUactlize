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
    asm-memory-contract|logical-x2-scalar|async-shared-fence|remaining-two|remaining-three) ;;
    *)
      fail "FQ_A_STAGE_CANDIDATE must be asm-memory-contract, logical-x2-scalar, async-shared-fence, remaining-two, or remaining-three"
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
nontrans_x4_impl = m8_copy.split(
    "struct PPU0010_TSM_LD_SWZL_IMPL<Element, false>", 1)[1].split(
        "struct PPU0010_TSM_LD_SWZL_IMPL<Element, true>", 1)[0]
q4_bulk_producer = m8_copy.split(
    "sizeof_bits<Element>::value == 4", 1)[1].split(
        "sizeof_bits<Element>::value == 2", 1)[0]
m8_wrapper = m8_copy.split("struct PPU0010_TSM_LD_SWZL_M8", 1)[1]
hggc_aiu = copy_aiu.split(
    "#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100", 1)[1].split(
        "#else", 1)[0]
alternate_aiu = copy_aiu.split(
    "#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100", 1)[1].split(
        "#else", 1)[1].split("#endif", 1)[0]
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
    "S1 admitted by descriptor":
        "splits == 1 || splits == 2 || splits == 4 || splits == 8" in partition,
    "singleton plane keeps physical stride":
        "make_compact_fp32_partial_stride<PartialStride>(in.m, in.n)" in bench and
        "cute::get<2>(stride) = rows * columns;" in partial_layout,
    "failure-only partial oracle":
        bench.count("inspect_failed_partials(") == 2,
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
    "split-groups is metadata-decode-only":
        collective.count("PPU_PACKED_SPLIT_GROUPS") == 4 and
        packed_decode.count("PPU_PACKED_SPLIT_GROUPS") == 3 and
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
print("[fq-a-stage-root:source] PASS PPU0010 warp0 issues ordinary A+B; "
      "split-groups is metadata-only; exact asm-memory, logical-x2 and "
      "async-shared-fence seams; S is runtime data")
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
  if [ "$candidate" = remaining-three ]; then
    selected_arms="baseline asm-memory-contract logical-x2-scalar async-shared-fence"
  elif [ "$candidate" = remaining-two ]; then
    selected_arms="baseline logical-x2-scalar async-shared-fence"
  else
    selected_arms="baseline $candidate"
  fi
  for arm in $selected_arms; do
    case "$arm" in
      baseline) defs="" ;;
      asm-memory-contract) defs='PPU_PACKED_A_ASM_MEMORY_CONTRACT=1' ;;
      logical-x2-scalar) defs='PPU_M8_LOGICAL_X2_SCALAR_LOAD=1' ;;
      async-shared-fence) defs='PPU_MIXED_ASYNC_SHARED_FENCE=1' ;;
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
    elif grep -Eq -- '-DPPU_(PACKED_A_ASM_MEMORY_CONTRACT|M8_LOGICAL_X2_SCALAR_LOAD|MIXED_ASYNC_SHARED_FENCE)=1' \
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
      attempt=$((attempt + 1))
    done
  done

  python3 -B - "$out/results" "$attempts" "$repeats" "$candidate" <<'PY' \
      | tee "$out/results/verdict.log"
import pathlib, shlex, sys

result_dir = pathlib.Path(sys.argv[1])
attempts = int(sys.argv[2])
repeats = int(sys.argv[3])
selection = sys.argv[4]
if selection not in {"asm-memory-contract", "logical-x2-scalar",
                     "async-shared-fence", "remaining-two",
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
    baseline_target = baseline_target_cells == 4
    baseline_s1 = [runs[("baseline", attempt)][(provider, 1)]
                   for attempt in range(1, attempts + 1)
                   for provider in expected_symbols]
    baseline_s1_clean = all(clean(row) for row in baseline_s1)
    baseline_s1_corrupt = any(corrupt(row) for row in baseline_s1)
    baseline_s1_inadmissible = all(s1_inadmissible(row, 1)
                                   for row in baseline_s1)
    if baseline_s1_corrupt:
        failure_scope = "CUSTOM_KERNEL_COMMON_S1_S2_S4"
    elif baseline_s1_clean:
        failure_scope = "S_GT1_ONLY_IN_THIS_DENOMINATOR"
    elif baseline_s1_inadmissible:
        failure_scope = "S1_UNEXECUTED"
    else:
        failure_scope = "S1_MIXED_OR_INCOMPLETE"
    clean_candidates = [arm for arm in candidates if candidate_clean[arm]]
    if not baseline_target:
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
        else:
            verdict = "ASYNC_SHARED_VISIBILITY_FENCE_CAUSAL"
            interpretation = (
                "the baseline reproduced while issuer sets, copy groups, stage "
                "geometry, CTA barrier and MMA order remained unchanged; an "
                "explicit async-shared proxy fence alone closed every S1/S2/S4 cell")
        diagnostic_rc = 0

    print("FQ_A_STAGE_ROOT_VERDICT "
          f"verdict={verdict} baseline_target={int(baseline_target)} "
          f"baseline_target_cells={baseline_target_cells}/4 "
          f"baseline_exact_fingerprint={int(baseline_exact_fingerprint)} "
          f"selection={selection} candidates={','.join(candidates)} "
          f"clean_candidates={','.join(clean_candidates) or 'NONE'} "
          f"failure_scope={failure_scope}")
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
