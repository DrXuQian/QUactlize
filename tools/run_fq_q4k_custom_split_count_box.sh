#!/usr/bin/env bash
# One-box causal control for the Q4_K fixed Split-K partial epilogue.
#
# The generated AP0/AP1 symbols are identical across runtime S=1/2/4.  The
# benchmark's diagnostic flag bypasses ShippingGemm at S=1 and submits that
# cell through the same GemmUniversalMixedInputSplitKParallel type as S=2/4.
# Two binaries differ only by the retained legacy shared partial epilogue.
set -uo pipefail

fail() {
  printf '[fq-custom-split-count] FAIL: %s\n' "$*" >&2
  return 2
}

main() {
  local root workspace_root sha short stamp out jobs repeats
  local full generated arm defs build_dir build_log binary log rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-custom-split-count-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) fail "OUT must resolve below /workspace: $out"; return $? ;;
  esac
  [ ! -e "$out" ] || { fail "refusing to overwrite $out"; return $?; }
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the two-arm identity'
    return $?
  fi
  jobs="${JOBS:-16}"
  repeats="${PROBE_REPEATS:-8192}"
  case "$jobs:$repeats" in
    *[!0-9:]*|0:*|*:0)
      fail 'JOBS and PROBE_REPEATS must be positive integers'
      return $?
      ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" \
    "$out/results" || return 2

  # Fail closed if the diagnostic flag starts changing a device type instead
  # of only selecting the already-instantiated custom type on the host.
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
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    raise SystemExit("custom-S1 source seam changed: " + repr(bad))
print("[fq-custom-split-count:source] PASS host-route-only; S is runtime data")
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
    "schema": "quactlize.fq-custom-split-count-closure.v1",
    "source_manifest": str(manifest_path),
    "source_typed_denominator": len(rows),
    "selection_denominator": 2,
    "identity": identity,
    "typed_rows": selected_rows,
    "units": copied,
}
(output / "manifest.json").write_text(
    json.dumps(closure, indent=2, sort_keys=True) + "\n")
print(f"[fq-custom-split-count:select] PASS source_typed={len(rows)} selected=2")
PY

  for arm in direct legacy-shared; do
    defs=""
    if [ "$arm" = legacy-shared ]; then
      defs='PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1'
    fi
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
      fail "$arm build rc=$rc artifacts=$out"
      return "$rc"
    fi
    if [ "$arm" = legacy-shared ]; then
      grep -Fq -- '-DPPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1' \
        "$build_log" || {
          fail 'legacy define did not reach the compiled target'
          return $?
        }
    elif grep -Fq 'PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE' "$build_log"; then
      fail 'direct arm unexpectedly contains the legacy define'
      return $?
    fi
    binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
    [ -x "$binary" ] && [ ! -L "$binary" ] || {
      fail "$arm binary missing: $binary"
      return $?
    }
    log="$out/results/$arm.log"
    "$binary" --shape=1x1024x5120 --iterations=1 \
      --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
      --force-custom-splitk-s1 >"$log" 2>&1
    rc=$?
    # The benchmark returns 1 when one or more selected cells fail numeric
    # correctness.  That is evidence for this diagnostic, not a runner
    # failure: retain both arms and let the closed denominator below classify
    # it.  Only an rc outside {0,1} is an infrastructure failure.
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
      tail -100 "$log" >&2
      fail "$arm infrastructure rc=$rc artifacts=$out"
      return $?
    fi
    printf 'FQ_CUSTOM_SPLIT_COUNT_BINARY arm=%s sha256=%s rc=%d repeats=%s\n' \
      "$arm" "$(sha256sum "$binary" | awk '{print $1}')" "$rc" "$repeats"
  done

  python3 -B - "$out/results/direct.log" \
      "$out/results/legacy-shared.log" <<'PY' \
      | tee "$out/results/verdict.log"
import pathlib, shlex, sys

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

try:
    direct = parse(sys.argv[1])
    legacy = parse(sys.argv[2])
    for arm, rows in (("direct", direct), ("legacy-shared", legacy)):
        bad_states = [
            f"{provider}:S{split}:{row['state']}"
            for (provider, split), row in rows.items()
            if not (clean(row) or corrupt(row) or
                    s1_inadmissible(row, split))]
        if bad_states:
            raise ValueError(f"{arm} arm contains infrastructure/non-numeric "
                             f"states: {','.join(sorted(bad_states))}")
        for provider in expected_symbols:
            for split in (1, 2, 4):
                row = rows[(provider, split)]
                print(
                    "FQ_CUSTOM_SPLIT_COUNT_RESULT "
                    f"arm={arm} provider={provider} S={split} "
                    f"state={row['state']} raw_bad={row['raw_bad']} "
                    f"failure_repeat={row['failure_repeat']} "
                    f"first_bad={row['first_bad']} "
                    f"first_want={row['first_want']} "
                    f"first_got={row['first_got']} "
                    f"partial_probe={row.get('partial_probe', 'MISSING')} "
                    f"partial_value_raw_bad="
                    f"{row.get('partial_value_raw_bad', 'MISSING')} "
                    f"partial_bad_plane_mask="
                    f"{row.get('partial_bad_plane_mask', 'MISSING')} "
                    f"reducer_replay_raw_bad="
                    f"{row.get('reducer_replay_raw_bad', 'MISSING')} "
                    f"localization={localization(row)}")

    direct_s_gt1_bad = [
        f"{provider}:S{split}"
        for provider in expected_symbols for split in (2, 4)
        if corrupt(direct[(provider, split)])]
    direct_s1_bad = [provider for provider in expected_symbols
                     if corrupt(direct[(provider, 1)])]
    direct_s1_inadmissible = [provider for provider in expected_symbols
        if s1_inadmissible(direct[(provider, 1)], 1)]
    legacy_s_gt1_clean = [
        f"{provider}:S{split}"
        for provider in expected_symbols for split in (2, 4)
        if clean(legacy[(provider, split)])]

    # Fail closed, but only after publishing all twelve cells and a semantic
    # verdict.  A dirty production-direct S>1 cell invalidates the shipping
    # repair.  A dirty production-direct custom S1 cell invalidates this S1
    # control specifically; neither outcome may be hidden as a missing log.
    diagnostic_rc = 0
    if direct_s_gt1_bad:
        verdict = "PRODUCTION_DIRECT_S_GT1_REGRESSION"
        loci = sorted({localization(direct[(provider, split)])
                       for provider in expected_symbols for split in (2, 4)
                       if corrupt(direct[(provider, split)])})
        interpretation = (
            "the production direct-store S2/S4 arm is not raw-bit exact; "
            "the shipping repair must be re-opened before this causal test; "
            "failure localization=" + ",".join(loci))
        detail = ",".join(direct_s_gt1_bad) + ";locus=" + ",".join(loci)
        diagnostic_rc = 1
    elif direct_s1_inadmissible:
        verdict = "CUSTOM_S1_CONTROL_INADMISSIBLE"
        interpretation = (
            "the custom-kernel S1 route was rejected before launch; runtime "
            "split count remains unadjudicated")
        detail = ",".join(direct_s1_inadmissible)
        diagnostic_rc = 1
    elif direct_s1_bad:
        verdict = "CUSTOM_S1_CONTROL_INVALID"
        interpretation = (
            "the custom-kernel S1 direct-store route is itself not raw-bit "
            "exact, so legacy custom-S1 cannot adjudicate whether runtime "
            "K partitioning is necessary")
        detail = ",".join(direct_s1_bad)
        diagnostic_rc = 1
    elif legacy_s_gt1_clean:
        verdict = "LEGACY_NEGATIVE_NONREPRODUCTION"
        interpretation = (
            "one or more legacy S2/S4 cells stayed clean; the historical "
            "negative denominator did not reproduce")
        detail = ",".join(legacy_s_gt1_clean)
        diagnostic_rc = 1
    else:
        legacy_s1_inadmissible = [provider for provider in expected_symbols
            if s1_inadmissible(legacy[(provider, 1)], 1)]
        if legacy_s1_inadmissible:
            print("FQ_CUSTOM_SPLIT_COUNT_VERDICT "
                  "verdict=CUSTOM_S1_CONTROL_INADMISSIBLE "
                  f"detail={','.join(legacy_s1_inadmissible)}")
            print("FQ_CUSTOM_SPLIT_COUNT_INTERPRETATION the legacy custom-S1 "
                  "route was rejected before launch")
            raise SystemExit(1)
        s1_bad = [provider for provider in expected_symbols
                  if corrupt(legacy[(provider, 1)])]
        if not s1_bad:
            verdict = "RUNTIME_SPLIT_DECOMPOSITION_NECESSARY"
            interpretation = (
                "same custom kernel and FP32 epilogue are clean at S1; "
                "S>1 K partition/work-grid context is required")
        elif len(s1_bad) == len(expected_symbols):
            verdict = "CUSTOM_KERNEL_CONTEXT_SUFFICIENT"
            interpretation = (
                "runtime split count is not necessary; shipping S1 stays "
                "clean because it is a different kernel/output epilogue")
        else:
            verdict = "PROVIDER_DEPENDENT_CUSTOM_S1"
            interpretation = (
                "custom S1 sensitivity depends on the A provider; runtime "
                "split is not a universal explanation")
        detail = ",".join(s1_bad) if s1_bad else "NONE"

    print(f"FQ_CUSTOM_SPLIT_COUNT_VERDICT verdict={verdict} detail={detail}")
    print("FQ_CUSTOM_SPLIT_COUNT_INTERPRETATION " + interpretation)
    raise SystemExit(diagnostic_rc)
except (KeyError, OSError, ValueError) as error:
    print(f"[fq-custom-split-count] FAIL: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
  rc=${PIPESTATUS[0]}

  sha256sum "$generated/manifest.json" "$out/results/direct.log" \
    "$out/results/legacy-shared.log" "$out/results/verdict.log" \
    >"$out/results/authority.sha256" || return 2
  if [ "$rc" -ne 0 ]; then
    fail "unadjudicated device result artifacts=$out"
    return "$rc"
  fi
  printf '[fq-custom-split-count] PASS sha=%s artifacts=%s\n' "$sha" "$out"
}

main "$@"
