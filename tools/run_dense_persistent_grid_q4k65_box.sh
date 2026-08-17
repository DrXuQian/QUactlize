#!/usr/bin/env bash
# Measure one exact Q4K65 collective with the ordinary launch and with a
# literal persistent-DP grid axis.  This runner intentionally does not involve
# Stream-K: every persistent CTA receives whole output tiles only.
#
# The grid is an absolute CTA count, not blocks/CU.  The default axis contains
# CU-multiple grids through the exact occupancy ceiling, plus G=512 as the
# power-of-two load-balance control.  Override it with, e.g.
#
#   PERSISTENT_GRID_CTAS_LIST="432 504 512 576" bash tools/run_dense_persistent_grid_q4k65_box.sh
#
# No identity probe is used.  Provenance is the repository SHA, actlize SHA,
# one built executable path, and one executable hash recorded in the bundle.
set -uo pipefail

main() {
  local root target sha actlize_sha short stamp out build_root
  local build_log list_log control_log control_meta default_log default_meta
  local manifest summary
  local grid_spec iterations jobs binary binary_sha rc tee_rc failures grid
  local -a bins grids pipeline_rc common
  local -A seen_grid=()

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  target=test_lowbit_dense_streamk_q4k65_ab
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  actlize_sha="$(git -C "$root/third_party/actlize" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-dense-persistent-grid-q4k65-${short}-${stamp}}"
  out="$(realpath -m -- "$out")" || return 2
  case "$out" in
    /workspace/*) ;;
    *)
      printf '[q4k65-persistent-grid] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2
      return 2
      ;;
  esac
  if [ -e "$out" ]; then
    printf '[q4k65-persistent-grid] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi

  grid_spec="${PERSISTENT_GRID_CTAS_LIST:-72 144 216 288 360 432 504 512 576}"
  iterations="${ITERATIONS:-20}"
  jobs="${JOBS:-16}"
  read -r -a grids <<<"$grid_spec"
  case "$iterations" in
    ''|*[!0-9]*)
      printf '[q4k65-persistent-grid] FAIL: ITERATIONS must be a positive integer\n' >&2
      return 2
      ;;
  esac
  case "$jobs" in
    ''|*[!0-9]*)
      printf '[q4k65-persistent-grid] FAIL: JOBS must be a positive integer\n' >&2
      return 2
      ;;
  esac
  if [ "$iterations" -lt 1 ] || [ "$jobs" -lt 1 ] || [ "${#grids[@]}" -eq 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: require ITERATIONS>=1, JOBS>=1 and a nonempty grid list\n' >&2
    return 2
  fi
  for grid in "${grids[@]}"; do
    case "$grid" in
      ''|*[!0-9]*)
        printf '[q4k65-persistent-grid] FAIL: every grid value must be a positive integer: %s\n' "$grid" >&2
        return 2
        ;;
    esac
    if [ "$grid" -lt 1 ]; then
      printf '[q4k65-persistent-grid] FAIL: grid must be positive: %s\n' "$grid" >&2
      return 2
    fi
    if [ -n "${seen_grid[$grid]+x}" ]; then
      printf '[q4k65-persistent-grid] FAIL: duplicate grid value: %s\n' "$grid" >&2
      return 2
    fi
    seen_grid[$grid]=1
  done

  build_root="$out/build"
  build_log="$out/build.log"
  list_log="$out/configs.log"
  control_log="$out/non-persistent.log"
  control_meta="$out/non-persistent.json"
  default_log="$out/persistent-default.log"
  default_meta="$out/persistent-default.json"
  manifest="$out/manifest.txt"
  summary="$out/results.tsv"
  mkdir -p "$out" "$build_root" || return 2

  {
    printf 'root_sha=%s\n' "$sha"
    printf 'actlize_sha=%s\n' "$actlize_sha"
    printf 'target=%s\n' "$target"
    printf 'shape=M2048,N4096,K4096,L1,gs32,ScaleOnly\n'
    printf 'config=64x64:64_w64x32_s3_bc0->0\n'
    printf 'fixture=q4k65-exact,order-independent+fp16-exact\n'
    printf 'persistent_default_request=0\n'
    printf 'persistent_grid_authority=persistent-default-arm\n'
    printf 'persistent_grid_ctas_values=%s\n' "${grids[*]}"
    printf 'iterations=%s\njobs=%s\n' "$iterations" "$jobs"
    printf 'utc=%s\n' "$stamp"
  } >"$manifest"
  printf 'role\tgrid_ctas\tstatus\trc\tcu\toccupancy_api\tq\tq_per_grid\tmax_work\tlong_workers\tlong_work\tshort_workers\tshort_work\tworker_rounds\tworker_empty\tworker_overhead_pct\tworker_fill_pct\tworker_minmax_pct\tresident_capacity\tresident_grid_fraction_pct\tbaseline_capacity_rounds\tbaseline_capacity_empty\tbaseline_capacity_overhead_pct\teffective_capacity_empty\teffective_capacity_overhead_pct\tmedian_us\tlog\n' >"$summary"

  printf '[q4k65-persistent-grid] sha=%s actlize=%s out=%s iterations=%s grids=%s\n' \
    "$sha" "$actlize_sha" "$out" "$iterations" "${grids[*]}"
  printf '[q4k65-persistent-grid] build target=%s (one binary for control and every grid cell)\n' "$target"
  env PPU_BUILD_DIR="$build_root" PPU_ARCHS=ppu0010 TARGET="$target" \
    QUANT=int4 BENCH_GS=32 JOBS="$jobs" "$root/build.sh" 2>&1 | tee "$build_log"
  pipeline_rc=("${PIPESTATUS[@]}")
  rc=${pipeline_rc[0]}
  tee_rc=${pipeline_rc[1]}
  if [ "$tee_rc" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: tee could not persist build log, rc=%d\n' "$tee_rc" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 74
  fi
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: isolated target build returned rc=%d\n' "$rc" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi

  mapfile -t bins < <(find "$build_root" -type f -name "$target" -perm -u+x -print)
  if [ "${#bins[@]}" -ne 1 ]; then
    printf '[q4k65-persistent-grid] FAIL: expected one executable, found %d\n' "${#bins[@]}" >&2
    printf '  %s\n' "${bins[@]:-<none>}" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 2
  fi
  binary="${bins[0]}"
  binary_sha="$(sha256sum "$binary" | awk '{print $1}')" || return 2
  {
    printf 'binary=%s\n' "$binary"
    printf 'binary_sha256=%s\n' "$binary_sha"
  } >>"$manifest"
  printf '[q4k65-persistent-grid] binary=%s sha256=%s\n' "$binary" "$binary_sha"

  if ! "$binary" --list_configs >"$list_log" 2>&1; then
    printf '[q4k65-persistent-grid] FAIL: --list_configs returned nonzero\n' >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 1
  fi
  if [ "$(grep -Ec '^  .*tile 64x64x64  warp 64x32  stages 3  instruction=m16$' "$list_log")" -ne 1 ] ||
     [ "$(grep -Ec '^  .*tile [0-9]+x[0-9]+x[0-9]+  warp [0-9]+x[0-9]+  stages [0-9]+  instruction=m(8|16)$' "$list_log")" -ne 1 ]; then
    printf '[q4k65-persistent-grid] FAIL: binary is not the unique 64x64x64/w64x32/s3 row\n' >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 1
  fi

  common=(--m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1
          --alpha=1 --beta=0 '--config=64x64x64:64x32:s3:bc0->0'
          "--iterations=$iterations" --streamk_exact_fixture)

  printf '\n== exact non-persistent control ==\n'
  "$binary" "${common[@]}" 2>&1 | tee "$control_log"
  pipeline_rc=("${PIPESTATUS[@]}")
  rc=${pipeline_rc[0]}
  tee_rc=${pipeline_rc[1]}
  if [ "$tee_rc" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: tee could not persist control log, rc=%d\n' "$tee_rc" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 74
  fi
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: non-persistent control returned rc=%d\n' "$rc" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  if ! adjudicate_log "$control_log" "$control_meta" non-persistent 2048 "$iterations" 0 0; then
    printf '[q4k65-persistent-grid] FAIL: non-persistent control evidence did not close\n' >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 1
  fi

  local control_cu control_occupancy control_us
  read -r control_cu control_occupancy control_us < <(
    python3 - "$control_meta" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(d["cu"], d["occupancy_api"], d["median_us"])
PY
  )
  printf '[q4k65-persistent-grid] CONTROL scheduler=non-persistent Q=2048 grid=2048 cu=%s occupancy_api=%s median=%s_us exact=RAW-BIT/PASS binary_sha256=%s\n' \
    "$control_cu" "$control_occupancy" "$control_us" "$binary_sha"
  printf 'non-persistent\t2048\tPASS\t0\t%s\t%s\t2048\t1.000000\t1\t0\t1\t2048\t1\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\t%s\n' \
    "$control_cu" "$control_occupancy" "$control_us" "$(basename "$control_log")" >>"$summary"

  printf '\n== exact persistent default grid (request=0) ==\n'
  "$binary" "${common[@]}" --persistent 2>&1 | tee "$default_log"
  pipeline_rc=("${PIPESTATUS[@]}")
  rc=${pipeline_rc[0]}
  tee_rc=${pipeline_rc[1]}
  if [ "$tee_rc" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: tee could not persist default-persistent log, rc=%d\n' "$tee_rc" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 74
  fi
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: default-persistent admission returned rc=%d\n' "$rc" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  if ! adjudicate_log "$default_log" "$default_meta" persistent 0 "$iterations" \
      "$control_cu" 0; then
    printf '[q4k65-persistent-grid] FAIL: default-persistent authority did not close\n' >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 1
  fi

  local persistent_cu persistent_occupancy persistent_default_grid persistent_default_us
  read -r persistent_cu persistent_occupancy persistent_default_grid persistent_default_us < <(
    python3 - "$default_meta" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(d["cu"], d["occupancy_api"], d["grid_ctas"], d["median_us"])
PY
  )
  {
    printf 'persistent_default_cu=%s\n' "$persistent_cu"
    printf 'persistent_default_occupancy_api=%s\n' "$persistent_occupancy"
    printf 'persistent_default_resolved_grid=%s\n' "$persistent_default_grid"
    printf 'persistent_default_median_us=%s\n' "$persistent_default_us"
  } >>"$manifest"
  printf '[q4k65-persistent-grid] DEFAULT scheduler=persistent requested=0 resolved=%s cu=%s occupancy_api=%s median=%s_us exact=RAW-BIT/PASS authority=explicit-grid-cells\n' \
    "$persistent_default_grid" "$persistent_cu" "$persistent_occupancy" \
    "$persistent_default_us"
  append_persistent_result "$default_meta" "$summary" \
    "$(basename "$default_log")" persistent-default DEFAULT_RESULT

  failures=0
  for grid in "${grids[@]}"; do
    local log meta
    log="$out/persistent-g${grid}.log"
    meta="$out/persistent-g${grid}.json"
    printf '\n== exact persistent pure-DP grid=%s ==\n' "$grid"
    "$binary" "${common[@]}" --persistent "--persistent-grid-ctas=$grid" \
      2>&1 | tee "$log"
    pipeline_rc=("${PIPESTATUS[@]}")
    rc=${pipeline_rc[0]}
    tee_rc=${pipeline_rc[1]}
    if [ "$tee_rc" -ne 0 ]; then
      printf '[q4k65-persistent-grid] GRID_RESULT G=%s status=FAIL stage=tee rc=%d\n' "$grid" "$tee_rc" >&2
      printf 'persistent\t%s\tFAIL\t%s\tNA\tNA\t2048\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\n' \
        "$grid" "$tee_rc" "$(basename "$log")" >>"$summary"
      failures=$((failures + 1))
      continue
    fi
    if [ "$rc" -ne 0 ]; then
      printf '[q4k65-persistent-grid] GRID_RESULT G=%s status=FAIL stage=run rc=%d\n' "$grid" "$rc" >&2
      printf 'persistent\t%s\tFAIL\t%s\tNA\tNA\t2048\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\n' \
        "$grid" "$rc" "$(basename "$log")" >>"$summary"
      failures=$((failures + 1))
      continue
    fi
    if ! adjudicate_log "$log" "$meta" persistent "$grid" "$iterations" \
        "$persistent_cu" "$persistent_occupancy"; then
      printf '[q4k65-persistent-grid] GRID_RESULT G=%s status=FAIL stage=adjudication rc=1\n' "$grid" >&2
      printf 'persistent\t%s\tFAIL\t1\tNA\tNA\t2048\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\n' \
        "$grid" "$(basename "$log")" >>"$summary"
      failures=$((failures + 1))
      continue
    fi
    append_persistent_result "$meta" "$summary" "$(basename "$log")" \
      persistent GRID_RESULT
  done

  find "$out" -type f ! -path "$build_root/*" ! -name bundle-files.sha256 -print0 | \
    sort -z | xargs -0 sha256sum >"$out/bundle-files.sha256"
  if [ "$failures" -ne 0 ]; then
    printf '[q4k65-persistent-grid] FAIL: %d grid cell(s) failed; all independent cells were attempted\n' "$failures" >&2
    printf '[q4k65-persistent-grid] artifacts: %s\n' "$out" >&2
    return 1
  fi
  printf '[q4k65-persistent-grid] PASS: control + default authority + %d explicit persistent grids; artifacts: %s\n' \
    "${#grids[@]}" "$out"
  return 0
}

append_persistent_result() {
  local meta="$1" summary="$2" log_name="$3" role="$4" label="$5"
  python3 - "$meta" "$summary" "$log_name" "$role" "$label" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
line = (
    "{role}\t{grid_ctas}\tPASS\t0\t{cu}\t{occupancy_api}\t{q}\t"
    "{q_per_grid:.6f}\t{max_work}\t{long_workers}\t{long_work}\t"
    "{short_workers}\t{short_work}\t{worker_rounds}\t{worker_empty}\t"
    "{worker_overhead_pct:.6f}\t{worker_fill_pct:.6f}\t"
    "{worker_minmax_pct:.6f}\t{resident_capacity}\t"
    "{resident_grid_fraction_pct:.6f}\t{baseline_capacity_rounds}\t"
    "{baseline_capacity_empty}\t{baseline_capacity_overhead_pct:.6f}\t"
    "{effective_capacity_empty}\t{effective_capacity_overhead_pct:.6f}\t"
    "{median_us:.6f}\t{log}\n"
).format(**d, role=sys.argv[4], log=sys.argv[3])
with pathlib.Path(sys.argv[2]).open("a") as f:
    f.write(line)
print(
    f"[q4k65-persistent-grid] {sys.argv[5]} "
    f"requested={d['requested_grid_ctas']} G={d['grid_ctas']} "
    f"status=PASS Q={d['q']} Q/G={d['q_per_grid']:.6f} "
    f"max_work={d['max_work']} "
    f"long={d['long_workers']}x{d['long_work']} "
    f"short={d['short_workers']}x{d['short_work']} "
    f"worker_rounds={d['worker_rounds']} worker_empty={d['worker_empty']} "
    f"worker_overhead={d['worker_overhead_pct']:.3f}% "
    f"worker_fill={d['worker_fill_pct']:.3f}% "
    f"worker_minmax={d['worker_minmax_pct']:.3f}% "
    f"resident_capacity={d['resident_capacity']} "
    f"resident_grid_fraction={d['resident_grid_fraction_pct']:.3f}% "
    f"baseline_capacity_rounds={d['baseline_capacity_rounds']} "
    f"baseline_capacity_empty={d['baseline_capacity_empty']} "
    f"baseline_capacity_overhead={d['baseline_capacity_overhead_pct']:.3f}% "
    f"effective_capacity_empty={d['effective_capacity_empty']} "
    f"effective_capacity_overhead={d['effective_capacity_overhead_pct']:.3f}% "
    f"median={d['median_us']:.6f}_us exact=RAW-BIT/PASS"
)
PY
}

adjudicate_log() {
  local log="$1" meta="$2" scheduler="$3" requested_grid="$4"
  local iterations="$5" expected_cu="$6" expected_occupancy="$7"
  python3 - "$log" "$meta" "$scheduler" "$requested_grid" "$iterations" \
    "$expected_cu" "$expected_occupancy" <<'PY'
import json
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
meta_path = pathlib.Path(sys.argv[2])
scheduler = sys.argv[3]
requested_grid = int(sys.argv[4])
iterations = int(sys.argv[5])
expected_cu = int(sys.argv[6])
expected_occupancy = int(sys.argv[7])
text = log_path.read_text()

exact = re.findall(
    r"\[streamk fixture exactness\] fixture=q4k65-exact "
    r"shape=2048x4096x4096 .* -> ORDER-INDEPENDENT\+FP16-EXACT",
    text,
)
bucket = re.findall(
    r"\[dense verify bucket=DP\] tiles=2048 outputs=8388608 "
    r"mismatches=0 max_abs=0 max_rel_sym=0 max_half_ulp=0 nonfinite=0",
    text,
)
fingerprints = re.findall(
    r"^\s*\[dense verify fingerprint\].*\braw_bitdiff=(\d+)\b.*$",
    text,
    re.M,
)
dispositions = re.findall(r"^\s*Disposition: (.+)$", text, re.M)
spans = re.findall(
    rf"\[dense kernel-span-upper\] n={iterations} median=([0-9.]+) us",
    text,
)
rows = re.findall(
    rf"\[CUTLASS w4 gs=32 cfg=64x64:64 w64x32 s3 bc0->0 "
    rf"scheduler={re.escape(scheduler)}\] M=2048\s+([0-9.]+) us",
    text,
)
grids = re.findall(
    rf"\[dense scheduler={re.escape(scheduler)}\] logical_cta=2048 cu=(\d+) "
    r"occupancy_api=(\d+) grid=\((\d+),(\d+),(\d+)\) "
    r"physical_cta=(\d+)",
    text,
)
persistent_markers = re.findall(
    r"\[dense persistent grid\] requested=(\d+) resolved=(\d+) default=(\d+) "
    r"resident_capacity=(\d+) Q=(\d+) short_work=(\d+) long_work=(\d+) "
    r"short_workers=(\d+) long_workers=(\d+) worker_empty=(\d+) "
    r"worker_balance_overhead=([0-9.]+)% resident_grid_fraction=([0-9.]+)% "
    r"grid_ctas_per_cu=(\d+)\+(\d+)/(\d+) ceil=(\d+) "
    r"grid_warps_per_cu_avg=([0-9.]+) baseline_capacity_rounds=(\d+) "
    r"baseline_capacity_empty=(\d+) baseline_wave_overhead=([0-9.]+)% "
    r"effective_capacity_empty=(\d+) effective_capacity_overhead=([0-9.]+)% "
    r"coverage=grid-stride-exact-once",
    text,
)

errors = []
if len(exact) != 1:
    errors.append(f"exactness witnesses={len(exact)}, expected 1")
if len(bucket) != 1:
    errors.append(f"exact DP bucket witnesses={len(bucket)}, expected 1")
if fingerprints != ["0"]:
    errors.append(f"raw fingerprints={fingerprints}, expected exactly [0]")
if not dispositions or any("Failed" in item for item in dispositions) or not any(
    item.startswith("Passed") for item in dispositions
):
    errors.append(f"numerical dispositions are not an unambiguous PASS: {dispositions}")
if len(spans) != 1 or len(rows) != 1:
    errors.append(f"timing witnesses span={spans}, row={rows}")
elif abs(float(spans[0]) - float(rows[0])) > 0.01:
    errors.append(f"timing witnesses disagree span={spans[0]} row={rows[0]}")
if len(grids) != 1:
    errors.append(f"scheduler grid witnesses={grids}, expected 1")
if scheduler == "non-persistent" and persistent_markers:
    errors.append("non-persistent control printed a persistent-grid marker")
if scheduler == "persistent" and len(persistent_markers) != 1:
    errors.append(
        f"persistent requested/resolved markers={persistent_markers}, expected 1"
    )

if errors:
    for error in errors:
        print(f"[q4k65-persistent-grid] adjudication FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)

cu, occupancy, gx, gy, gz, physical = map(int, grids[0])
if gx * gy * gz != physical:
    errors.append(f"grid product {gx}*{gy}*{gz} != physical_cta={physical}")
if scheduler == "non-persistent":
    if physical != 2048:
        errors.append(f"non-persistent physical grid={physical}, expected Q=2048")
else:
    resolved_expected_grid = (
        min(2048, cu * occupancy) if requested_grid == 0 else requested_grid
    )
    if physical != resolved_expected_grid:
        errors.append(
            f"persistent physical grid={physical}, requested={requested_grid} "
            f"resolved={resolved_expected_grid}"
        )
    if not (gx == resolved_expected_grid and gy == 1 and gz == 1):
        errors.append(
            "persistent grid must be "
            f"({resolved_expected_grid},1,1), got {(gx, gy, gz)}"
        )
    if expected_cu > 0 and cu != expected_cu:
        errors.append(f"CU changed from control {expected_cu} to {cu}")
    if expected_occupancy > 0 and occupancy != expected_occupancy:
        errors.append(
            f"occupancy changed from control {expected_occupancy} to {occupancy}"
        )
    if requested_grid > cu * occupancy:
        errors.append(
            f"requested grid {requested_grid} exceeds exact resident ceiling {cu*occupancy}"
        )
if errors:
    for error in errors:
        print(f"[q4k65-persistent-grid] adjudication FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)

q = 2048
grid = physical
base, remainder = divmod(q, grid)
max_work = base + int(remainder != 0)
long_workers = remainder
short_workers = grid - remainder
long_work = max_work
short_work = base
worker_fill = q / (grid * max_work) * 100.0
worker_minmax = short_work / max_work * 100.0
worker_rounds = max_work
worker_empty = worker_rounds * grid - q
worker_overhead = worker_empty / q * 100.0
resident_capacity = cu * occupancy
resident_grid_fraction = grid / resident_capacity * 100.0
baseline_capacity_rounds = (q + resident_capacity - 1) // resident_capacity
baseline_capacity_empty = baseline_capacity_rounds * resident_capacity - q
baseline_capacity_overhead = baseline_capacity_empty / q * 100.0
effective_capacity_empty = worker_rounds * resident_capacity - q
effective_capacity_overhead = effective_capacity_empty / q * 100.0

if scheduler == "persistent":
    (requested, resolved, marker_default, marker_capacity, marker_q,
     marker_short_work, marker_long_work, marker_short_workers,
     marker_long_workers, marker_worker_empty, marker_worker_overhead,
     marker_resident_fraction, marker_grid_floor, marker_grid_remainder,
     marker_grid_denominator, marker_grid_ceil, marker_grid_warps_avg,
     marker_capacity_rounds, marker_capacity_empty, marker_capacity_overhead,
     marker_effective_empty, marker_effective_overhead) = persistent_markers[0]
    ints = list(map(int, (
        requested, resolved, marker_default, marker_capacity, marker_q,
        marker_short_work, marker_long_work, marker_short_workers,
        marker_long_workers, marker_worker_empty, marker_grid_floor,
        marker_grid_remainder, marker_grid_denominator, marker_grid_ceil,
        marker_capacity_rounds, marker_capacity_empty, marker_effective_empty,
    )))
    grid_floor, grid_remainder = divmod(grid, cu)
    grid_ceil = grid_floor + int(grid_remainder != 0)
    expected_ints = [
        requested_grid, grid, min(q, resident_capacity), resident_capacity, q,
        short_work, long_work, short_workers, long_workers, worker_empty,
        grid_floor, grid_remainder, cu, grid_ceil,
        baseline_capacity_rounds, baseline_capacity_empty,
        effective_capacity_empty,
    ]
    if ints != expected_ints:
        raise SystemExit(
            "[q4k65-persistent-grid] adjudication FAIL: persistent marker "
            f"requested/resolved/census drifted got={ints} expected={expected_ints}"
        )
    floats = list(map(float, (
        marker_worker_overhead, marker_resident_fraction,
        marker_grid_warps_avg, marker_capacity_overhead,
        marker_effective_overhead,
    )))
    expected_floats = [
        worker_overhead, resident_grid_fraction, grid * 2.0 / cu,
        baseline_capacity_overhead, effective_capacity_overhead,
    ]
    if any(abs(got - want) > 1e-5 for got, want in zip(floats, expected_floats)):
        raise SystemExit(
            "[q4k65-persistent-grid] adjudication FAIL: persistent marker "
            f"overhead/fraction drifted got={floats} expected={expected_floats}"
        )
data = {
    "scheduler": scheduler,
    "requested_grid_ctas": requested_grid,
    "cu": cu,
    "occupancy_api": occupancy,
    "q": q,
    "grid_ctas": grid,
    "q_per_grid": q / grid,
    "max_work": max_work,
    "long_workers": long_workers,
    "long_work": long_work,
    "short_workers": short_workers,
    "short_work": short_work,
    "worker_rounds": worker_rounds,
    "worker_empty": worker_empty,
    "worker_overhead_pct": worker_overhead,
    "worker_fill_pct": worker_fill,
    "worker_minmax_pct": worker_minmax,
    "resident_capacity": resident_capacity,
    "resident_grid_fraction_pct": resident_grid_fraction,
    "baseline_capacity_rounds": baseline_capacity_rounds,
    "baseline_capacity_empty": baseline_capacity_empty,
    "baseline_capacity_overhead_pct": baseline_capacity_overhead,
    "effective_capacity_empty": effective_capacity_empty,
    "effective_capacity_overhead_pct": effective_capacity_overhead,
    "median_us": float(spans[0]),
}
meta_path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

main "$@"
