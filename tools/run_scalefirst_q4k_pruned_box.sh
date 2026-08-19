#!/usr/bin/env bash
# Conservative Q4_K ScaleFirst pruning pilot on the historical 65.7% MFU
# shape/config anchor.  This is intentionally separate from the exhaustive
# all-model runner: every legal compiled type is screened, but only an audited
# shortlist expands the scheduler and seven-sample confirmation boards.
set -uo pipefail

main() {
  if [ "$#" -ne 0 ]; then
    printf '[q4k-prune] FAIL: no positional arguments are accepted\n' >&2
    return 2
  fi
  local root workspace_root sha short stamp out jobs per_unit
  local policy policy_copy generated manifest build_dir binary build_log
  local shape screen_iterations screen_repeats scheduler_iterations
  local scheduler_repeats confirm_iterations confirm_repeats seed rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || {
    printf '[q4k-prune] FAIL: /workspace is unavailable\n' >&2
    return 2
  }
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-q4k-pruned-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[q4k-prune] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[q4k-prune] FAIL: refusing existing OUT=%s\n' "$out" >&2
    return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/raw" "$out/results" \
    "$out/inputs" || return 2

  jobs="${JOBS:-16}"
  per_unit="${SCALEFIRST_CONFIGS_PER_UNIT:-32}"
  case "$jobs:$per_unit" in
    *[!0-9:]*|0:*|*:0)
      printf '[q4k-prune] FAIL: JOBS/SCALEFIRST_CONFIGS_PER_UNIT must be positive integers\n' >&2
      return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[q4k-prune] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the compiled graph\n' >&2
    return 2
  fi

  policy="$root/benchmarks/scalefirst_q4k_pruned_policy.json"
  policy_copy="$out/inputs/scalefirst_q4k_pruned_policy.json"
  cp -- "$policy" "$policy_copy" || return 2
  python3 -B - "$root" "$out/inputs/source-authority.json" "$sha" <<'PY' || return 2
import hashlib,json,os,pathlib,subprocess,sys
root,out,commit=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
paths=[
 "benchmarks/scalefirst_internal_sweep_bench.hpp",
 "benchmarks/test_scalefirst_internal_sweep.cu",
 "benchmarks/scalefirst_internal_sweep_unit.inc",
 "benchmarks/scalefirst_q4k_pruned_policy.json",
 "quactlize/csrc/scalefirst_internal_sweep.cmake.in",
 "tools/gen_scalefirst_internal_units.py",
 "tools/prune_scalefirst_q4k_pilot.py",
 "tools/run_scalefirst_q4k_pruned_box.sh",
 "tools/scalefirst_internal_matrix.py",
 "build.sh"]
for rel in paths:
 if not (root/rel).is_file(): raise SystemExit(f"source authority lacks {rel}")
status=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--"]+paths,text=True)
if status: raise SystemExit("pilot source authority is dirty:\n"+status)
doc={"schema":"quactlize.scalefirst_q4k_pruned_source.v1","git_sha":commit,
     "files":{rel:hashlib.sha256((root/rel).read_bytes()).hexdigest()
              for rel in paths}}
t=out.with_name(f".{out.name}.current.{os.getpid()}")
with t.open("w") as f:
 json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(t,out)
PY
  read -r shape screen_iterations screen_repeats scheduler_iterations \
    scheduler_repeats confirm_iterations confirm_repeats < <(
      python3 -B - "$policy_copy" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(
 "x".join(map(str,p["shape"])),
 p["screen"]["iterations"],p["screen"]["correctness_repeats"],
 p["scheduler"]["iterations"],p["scheduler"]["correctness_repeats"],
 p["confirm"]["iterations"],p["confirm"]["correctness_repeats"])
PY
    ) || return 2
  if [ -z "$shape" ] || [ -z "$confirm_repeats" ]; then
    printf '[q4k-prune] FAIL: policy phase tuple is incomplete\n' >&2
    return 2
  fi
  seed="${SCALEFIRST_SCHEDULE_SEED:-0x6a09e667f3bcc909}"

  python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" self-test || return 2
  generated="$out/generated/q12-a64-bc0"
  python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --per-unit "$per_unit" \
    --out-dir "$generated" || return 2
  manifest="$generated/manifest.json"
  python3 -B - "$manifest" "$policy_copy" <<'PY' || return 2
import json,sys
m,p=map(lambda x:json.load(open(x)),sys.argv[1:])
if m["identity"] != {"qtype":12,"format":"Q4_K","artifact_tile_k":64,"bchunk":0}:
 raise SystemExit("generated identity is not Q4_K/A64/bc0")
if m["denominator"]["typed_rows"] != 1824:
 raise SystemExit(f"typed denominator drifted: {m['denominator']['typed_rows']}/1824")
anchor=p["anchor_symbol"]
if sum(row["symbol"]==anchor for row in m["typed_rows"]) != 1:
 raise SystemExit("historical anchor is not present exactly once")
print(f"[q4k-prune] generated typed={m['denominator']['typed_rows']} anchor={anchor}")
PY

  build_dir="$out/build/q12-a64-bc0"
  binary="$build_dir/ppu_targets/test_scalefirst_internal_sweep"
  build_log="$out/build/q12-a64-bc0.log"
  printf '[q4k-prune] build Q4_K/A64/bc0 typed=1824\n'
  (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 \
    JOBS="$jobs" TARGET=test_scalefirst_internal_sweep \
    SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
    SCALEFIRST_SWEEP_QTYPE=12 SCALEFIRST_SWEEP_ARTIFACT_TK=64 \
    SCALEFIRST_SWEEP_BCHUNK=0 ./build.sh) >"$build_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k-prune] FAIL: exact Q4_K target build rc=%d\n' "$rc" >&2
    tail -100 "$build_log" >&2
    return "$rc"
  fi
  if [ ! -x "$binary" ]; then
    printf '[q4k-prune] FAIL: build returned success without exact binary: %s\n' \
      "$binary" >&2
    tail -100 "$build_log" >&2
    return 2
  fi
  printf '[q4k-prune] binary=%s sha256=%s\n' "$binary" \
    "$(sha256sum "$binary" | awk '{print $1}')"

  printf '[q4k-prune] phase=screen shape=%s algorithm=nonpersistent iterations=%s\n' \
    "$shape" "$screen_iterations"
  "$binary" --shape="$shape" --algorithm=nonpersistent \
    --iterations="$screen_iterations" --correctness-repeats="$screen_repeats" \
    --schedule-seed="$seed" >"$out/raw/screen.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k-prune] FAIL: screen rc=%d\n' "$rc" >&2
    tail -100 "$out/raw/screen.log" >&2
    return "$rc"
  fi
  python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" screen \
    --policy "$policy_copy" --manifest "$manifest" \
    --log "$out/raw/screen.log" --output "$out/results/screen.json" \
    --symbols-output "$out/results/screen-shortlist.txt" || return 2

  printf '[q4k-prune] phase=scheduler algorithms=NP+P+S2+S4+S8 iterations=%s\n' \
    "$scheduler_iterations"
  "$binary" --shape="$shape" --algorithm=all \
    --symbol-file="$out/results/screen-shortlist.txt" \
    --iterations="$scheduler_iterations" \
    --correctness-repeats="$scheduler_repeats" \
    --schedule-seed="$((seed + 1))" >"$out/raw/scheduler.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k-prune] FAIL: scheduler expansion rc=%d\n' "$rc" >&2
    tail -100 "$out/raw/scheduler.log" >&2
    return "$rc"
  fi
  python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" scheduler \
    --policy "$policy_copy" --manifest "$manifest" \
    --log "$out/raw/scheduler.log" \
    --expected-symbols "$out/results/screen-shortlist.txt" \
    --output "$out/results/scheduler.json" \
    --symbols-output "$out/results/confirm-shortlist.txt" || return 2

  printf '[q4k-prune] phase=confirm iterations=%s correctness_repeats=%s\n' \
    "$confirm_iterations" "$confirm_repeats"
  "$binary" --shape="$shape" --algorithm=all \
    --symbol-file="$out/results/confirm-shortlist.txt" \
    --iterations="$confirm_iterations" \
    --correctness-repeats="$confirm_repeats" \
    --schedule-seed="$((seed + 2))" >"$out/raw/confirm.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k-prune] FAIL: confirm rc=%d\n' "$rc" >&2
    tail -100 "$out/raw/confirm.log" >&2
    return "$rc"
  fi
  python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" confirm \
    --policy "$policy_copy" --manifest "$manifest" \
    --log "$out/raw/confirm.log" \
    --expected-symbols "$out/results/confirm-shortlist.txt" \
    --output "$out/results/summary.json" | tee "$out/results/winners.txt" || return 2

  python3 -B - "$root" "$out" "$sha" "$binary" "$manifest" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
root,out,commit,binary,manifest=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3],pathlib.Path(sys.argv[4]),pathlib.Path(sys.argv[5])
files=[out/"inputs/scalefirst_q4k_pruned_policy.json",
       out/"inputs/source-authority.json",manifest,binary,
       out/"raw/screen.log",out/"raw/scheduler.log",out/"raw/confirm.log",
       out/"results/screen.json",out/"results/scheduler.json",
       out/"results/summary.json",out/"results/screen-shortlist.txt",
       out/"results/confirm-shortlist.txt"]
doc={"schema":"quactlize.scalefirst_q4k_pruned_bundle.v1","git_sha":commit,
     "files":{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}}
p=out/"bundle.json"; t=out/f".bundle.json.current.{os.getpid()}"
with t.open("w") as f:
 json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
  printf '[q4k-prune] PASS sha=%s artifacts=%s\n' "$sha" "$out"
  return 0
}

main "$@"
