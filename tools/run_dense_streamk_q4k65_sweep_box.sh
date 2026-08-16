#!/usr/bin/env bash
# Build and sweep the complete prefill (TM>=16) int4/gs32 dense Stream-K tactic
# registry at the Q4_K65 prefill shape.  Decode-only TM8 is intentionally out
# of scope.  This is a scheduler-specific binary: every generated wrapper is
# StreamKGemm, while --streamk is retained as explicit operator intent rather
# than as a runtime selector.
#
# Durable evidence lives below /workspace.  The 209.30 us exact-normal result
# is printed as context only; it is not an admission threshold for this sweep.
set -uo pipefail

main() {
  local root target sha short stamp out build_root build_log list_log
  local sweep_log samples analysis_log analysis_json command_file manifest
  local binary rc source_rows eligible_rows filtered_rows reps iterations jobs
  local -a _bins sweep

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  target=test_lowbit_dense_streamk_sweep
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-dense-streamk-q4k65-sweep-${short}-${stamp}}"
  out="$(realpath -m -- "$out")" || return 2
  case "$out" in
    /workspace/*) ;;
    *) printf '[q4k65-streamk-sweep] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[q4k65-streamk-sweep] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi

  reps="${BENCH_REPS:-2}"
  iterations="${ITERATIONS:-7}"
  jobs="${JOBS:-16}"
  case "$reps" in ''|*[!0-9]*) printf '[q4k65-streamk-sweep] FAIL: BENCH_REPS must be an integer >=2\n' >&2; return 2 ;; esac
  case "$iterations" in ''|*[!0-9]*) printf '[q4k65-streamk-sweep] FAIL: ITERATIONS must be a positive integer\n' >&2; return 2 ;; esac
  case "$jobs" in ''|*[!0-9]*) printf '[q4k65-streamk-sweep] FAIL: JOBS must be a positive integer\n' >&2; return 2 ;; esac
  if [ "$reps" -lt 2 ] || [ "$iterations" -lt 1 ] || [ "$jobs" -lt 1 ]; then
    printf '[q4k65-streamk-sweep] FAIL: require BENCH_REPS>=2, ITERATIONS>=1, JOBS>=1\n' >&2
    return 2
  fi
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[q4k65-streamk-sweep] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS would change the binary\n' >&2
    return 2
  fi
  if [ -n "$(git -C "$root" status --porcelain=v1)" ]; then
    printf '[q4k65-streamk-sweep] FAIL: parent worktree is dirty; result would not bind to sha=%s\n' "$sha" >&2
    return 2
  fi
  if [ -n "$(git -C "$root/third_party/actlize" status --porcelain=v1)" ]; then
    printf '[q4k65-streamk-sweep] FAIL: third_party/actlize is dirty\n' >&2
    return 2
  fi

  mkdir -p "$out" || return 2
  build_root="$out/build"
  build_log="$out/build.log"
  list_log="$out/configs.log"
  sweep_log="$out/sweep.log"
  samples="$out/samples.jsonl"
  analysis_log="$out/analysis.txt"
  analysis_json="$out/analysis.json"
  command_file="$out/command.sh"
  manifest="$out/manifest.txt"

  {
    printf 'root_sha=%s\n' "$sha"
    printf 'actlize_sha=%s\n' "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'target=%s\n' "$target"
    printf 'shape=M2048,N4096,K4096,L1,gs32,ScaleOnly\n'
    printf 'sweep_scope=prefill\nprefill_tm_min=16\n'
    printf 'fixture=q4k65-exact,order-independent+fp16-exact\n'
    printf 'scheduler=streamk,build-time-direct-wrapper\n'
    printf 'bench_reps=%s\niterations=%s\njobs=%s\n' "$reps" "$iterations" "$jobs"
    printf 'normal_exact_context_us=209.30\nnormal_exact_context_is_gate=0\n'
  } >"$manifest"
  {
    printf 'utc=%s\n' "$stamp"
    printf 'uname='; uname -a 2>&1 || true
    printf 'hgcc='; (hgcc --version 2>&1 || true) | sed -n '1p'
    printf 'ppu_clang='; (ppu_clang++ --version 2>&1 || true) | sed -n '1p'
    printf 'visible_devices=%s\n' "${HGG_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-<unset>}}"
    printf 'lspci_begin\n'; lspci -Dnn 2>&1 || true; printf 'lspci_end\n'
    printf 'driver_modules_begin\n'; (lsmod 2>/dev/null | grep -Ei 'hgg|ppu|metax' || true); printf 'driver_modules_end\n'
  } >"$out/device.txt"

  printf '[q4k65-streamk-sweep] sha=%s out=%s reps=%s iterations=%s\n' \
    "$sha" "$out" "$reps" "$iterations"
  printf '[q4k65-streamk-sweep] context-only normal-exact=209.30 us (not an admission gate)\n'

  env PPU_BUILD_DIR="$build_root" PPU_ARCHS=ppu0010 TARGET="$target" \
    QUANT=int4 BENCH_GS=32 JOBS="$jobs" "$root/build.sh" 2>&1 | tee "$build_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-streamk-sweep] FAIL: isolated target build failed; tail follows\n' >&2
    tail -100 "$build_log" >&2
    printf '[q4k65-streamk-sweep] artifacts: %s\n' "$out" >&2
    return 1
  fi
  mapfile -t _bins < <(find "$build_root" -type f -name "$target" -perm -u+x -print)
  if [ "${#_bins[@]}" -ne 1 ]; then
    printf '[q4k65-streamk-sweep] FAIL: expected one executable, found %d\n' "${#_bins[@]}" >&2
    printf '  %s\n' "${_bins[@]:-<none>}" >&2
    return 2
  fi
  binary="${_bins[0]}"
  {
    printf 'binary=%s\n' "$binary"
    printf 'binary_sha256=%s\n' "$(sha256sum "$binary" | awk '{print $1}')"
  } >>"$manifest"
  printf '[q4k65-streamk-sweep] binary=%s sha256=%s\n' \
    "$binary" "$(sha256sum "$binary" | awk '{print $1}')"

  if ! "$binary" --list_configs >"$list_log" 2>&1; then
    printf '[q4k65-streamk-sweep] FAIL: --list_configs returned nonzero\n' >&2
    return 1
  fi
  python3 - "$list_log" "$out/config-census.json" \
    "$root/benchmarks/lowbit_dense_configs.inc" <<'PY' || return 1
import collections
import json
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
provenance = re.findall(
    r"^\[dense-table\] scheduler=streamk .* rows=(\d+) source_rows=(\d+) "
    r"eligible_rows=(\d+) filtered_rows=(\d+) gs=32 .*"
    r"shape_scope=prefill-TM>=16 .*"
    r"cohort_capability=exact-cta-threads-64-or-128 .*"
    r"startup_capability=Stages-1<=8 .*?$", text, re.M)
if provenance != [("577", "1772", "577", "1195")]:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: bad/duplicate provenance: {provenance}")
row_re = re.compile(
    r"^  (?P<name>.+?)\s{2,}tile (?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)  "
    r"warp (?P<wm>\d+)x(?P<wn>\d+)  stages (?P<st>\d+)  instruction=m(?P<im>8|16)$",
    re.M)
rows = []
for m in row_re.finditer(text):
    d = {k: (v.strip() if k == "name" else int(v)) for k, v in m.groupdict().items()}
    if d["tm"] % d["wm"] or d["tn"] % d["wn"]:
        raise SystemExit(f"[q4k65-streamk-sweep] FAIL: non-integral topology: {d}")
    d["threads"] = 32 * (d["tm"] // d["wm"]) * (d["tn"] // d["wn"])
    rows.append(d)
if len(rows) != 577:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: list has {len(rows)} rows, expected 577")
if len({r["name"] for r in rows}) != 577:
    raise SystemExit("[q4k65-streamk-sweep] FAIL: config names are not unique")
name_re = re.compile(
    r"^(\d+)x(\d+):(\d+) w(\d+)x(\d+) s(\d+) bc(\d+)->(\d+)$")
listed = set()
for row in rows:
    m = name_re.fullmatch(row["name"])
    if not m:
        raise SystemExit(f"[q4k65-streamk-sweep] FAIL: non-canonical config name {row['name']!r}")
    tm, tn, tk, wm, wn, st, bc, bc_eff = map(int, m.groups())
    if (tm, tn, tk, wm, wn, st) != tuple(row[k] for k in ("tm", "tn", "tk", "wm", "wn", "st")):
        raise SystemExit(f"[q4k65-streamk-sweep] FAIL: name/geometry disagreement: {row}")
    if bc_eff != 0:
        raise SystemExit(f"[q4k65-streamk-sweep] FAIL: int4 row unexpectedly grants B-chunk: {row['name']}")
    listed.add((tm, tn, tk, wm, wn, st, bc))
if any(row["tm"] < 16 for row in rows):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: decode-only TM8 entered the prefill registry")
if not any(row["tm"] == 16 for row in rows):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: prefill registry lost its TM16 control")
authority_text = pathlib.Path(sys.argv[3]).read_text()
authority = [tuple(map(int, m.groups())) for m in re.finditer(
    r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),B\)",
    authority_text, re.M)]
if len(authority) != 1772:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: source authority has {len(authority)} rows")
expected = {
    r for r in authority
    if r[0] >= 16 and
       32 * (r[0] // r[3]) * (r[1] // r[4]) in (64, 128) and
       r[5] - 1 <= 8
}
if len(expected) != 577 or listed != expected:
    raise SystemExit(
        f"[q4k65-streamk-sweep] FAIL: listed registry is not the exact independently filtered authority "
        f"missing={len(expected-listed)} extra={len(listed-expected)}")
thread_counts = collections.Counter(r["threads"] for r in rows)
stage_counts = collections.Counter(r["st"] for r in rows)
tile_m_counts = collections.Counter(r["tm"] for r in rows)
if thread_counts != {64: 248, 128: 329}:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: CTA cohort census {dict(thread_counts)}")
if stage_counts != {2: 132, 3: 131, 4: 119, 6: 108, 8: 87}:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: stage census {dict(stage_counts)}")
if tile_m_counts != {16: 88, 32: 175, 64: 199, 128: 97, 256: 18}:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: prefill TileM census {dict(tile_m_counts)}")
out = {
    "source_rows": 1772, "eligible_rows": 577, "filtered_rows": 1195,
    "cta_threads": dict(sorted(thread_counts.items())),
    "stages": dict(sorted(stage_counts.items())),
    "tile_m": dict(sorted(tile_m_counts.items())), "rows": rows,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
print("[q4k65-streamk-sweep] config census PASS: source=1772 eligible=577 "
      "filtered=1195 threads=64:248,128:329 stages=2:132,3:131,4:119,6:108,8:87 "
      "tile_m=16:88,32:175,64:199,128:97,256:18")
PY
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-streamk-sweep] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  source_rows=1772; eligible_rows=577; filtered_rows=1195

  sweep=("$binary" --search_configs --streamk --streamk_exact_fixture
    --m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1
    --alpha=1 --beta=0 "--iterations=$iterations")
  {
    printf 'BENCH_REPS=%q BENCH_JSONL=%q ' "$reps" "$samples"
    printf '%q ' "${sweep[@]}"
    printf '\n'
  } >"$command_file"

  BENCH_REPS="$reps" BENCH_JSONL="$samples" \
    "${sweep[@]}" 2>&1 | tee "$sweep_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-streamk-sweep] FAIL: sweep returned rc=%d\n' "$rc" >&2
    printf '[q4k65-streamk-sweep] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  if [ ! -s "$samples" ]; then
    printf '[q4k65-streamk-sweep] FAIL: sweep emitted no durable samples\n' >&2
    return 1
  fi
  if grep -Fq 'Disposition: Failed' "$sweep_log"; then
    printf '[q4k65-streamk-sweep] FAIL: an exact-fixture candidate printed a numerical failure\n' >&2
    return 1
  fi
  python3 - "$samples" "$reps" "$eligible_rows" "$out/sample-closure.json" \
    "$sweep_log" <<'PY' || return 1
import collections
import json
import pathlib
import re
import sys

path, expected_reps, expected_rows, outpath = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
runs = [x for x in records if x.get("rec") == "run"]
if (len(runs) != 1 or runs[0].get("reps") != expected_reps or
        runs[0].get("bench") != "cutlass_w4a16_streamk" or
        "scheduler=streamk" not in runs[0].get("build", "")):
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: run header is not one scheduler=streamk/reps={expected_reps} record: {runs}")
required = ("schema", "tm", "tn", "tk", "wm", "wn", "st", "bc", "fixture", "dist", "n", "k", "gs", "experts", "rows", "mmax", "pass")
def key(x): return tuple(x.get(k, 0 if k == "warp_k" else None) for k in required + ("warp_k",))
attempts = [x for x in records if x.get("rec") == "a"]
samples = [x for x in records if x.get("rec") == "s"]
excluded = [x for x in records if x.get("rec") == "x"]
unknown = [x for x in records if x.get("rec") not in {"run", "a", "s", "x"}]
if unknown:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: unknown JSONL records: {unknown[:2]}")
for kind, group, extra in (("attempt", attempts, ()), ("sample", samples, ("us",)), ("exclusion", excluded, ("why",))):
    for x in group:
        missing = [k for k in required + extra if k not in x]
        if missing:
            raise SystemExit(f"[q4k65-streamk-sweep] FAIL: {kind} missing {missing}: {x}")
if len(attempts) != expected_reps * expected_rows:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: attempts={len(attempts)}, expected {expected_reps*expected_rows}")
if len({key(x) for x in attempts}) != len(attempts):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: duplicate attempt identity")
if any(x.get("tm", 0) < 16 for x in attempts):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: decode-only TM8 was attempted by the prefill sweep")
if not any(x.get("tm") == 16 for x in attempts):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: prefill sweep made no TM16 control attempt")
done_s, done_x = {key(x) for x in samples}, {key(x) for x in excluded}
if done_s & done_x:
    raise SystemExit("[q4k65-streamk-sweep] FAIL: one attempt is both sampled and excluded")
unfinished = {key(x) for x in attempts} - done_s - done_x
extra = (done_s | done_x) - {key(x) for x in attempts}
if unfinished or extra:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: unfinished={len(unfinished)} orphan_outcomes={len(extra)}")
by_pass = collections.defaultdict(set)
sampled_by_pass = collections.defaultdict(set)
excluded_by_pass = collections.defaultdict(set)
def config(x): return tuple(x.get(k, 0 if k == "warp_k" else None) for k in ("schema", "tm", "tn", "tk", "wm", "wn", "warp_k", "st", "bc"))
for x in attempts: by_pass[x["pass"]].add(config(x))
for x in samples: sampled_by_pass[x["pass"]].add(config(x))
for x in excluded: excluded_by_pass[x["pass"]].add(config(x))
if set(by_pass) != set(range(expected_reps)) or any(len(v) != expected_rows for v in by_pass.values()):
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: per-pass attempt census {[len(by_pass[p]) for p in sorted(by_pass)]}")
if any(by_pass[p] != by_pass[0] for p in range(expected_reps)):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: compiled candidate set changed between passes")
if any(sampled_by_pass[p] != sampled_by_pass[0] or excluded_by_pass[p] != excluded_by_pass[0] for p in range(expected_reps)):
    raise SystemExit("[q4k65-streamk-sweep] FAIL: sampled/excluded status changed between passes")
bad_exclusions = [x for x in excluded if "Stream-K" not in x.get("why", "") or "NOT EXERCISED" not in x.get("why", "")]
if bad_exclusions:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: exclusion is not an explicit no-seam outcome: {bad_exclusions[:2]}")
for x in samples:
    if (x.get("schema"), x.get("n"), x.get("k"), x.get("gs"), x.get("experts"), x.get("rows"), x.get("mmax")) != ("i4", 4096, 4096, 32, 0, 2048, 2048):
        raise SystemExit(f"[q4k65-streamk-sweep] FAIL: sample escaped fixed fixture: {x}")
    if x.get("dist") != "dense-streamk-v1" or not (isinstance(x.get("us"), (int, float)) and x["us"] > 0):
        raise SystemExit(f"[q4k65-streamk-sweep] FAIL: sample lost Stream-K distribution/positive timing: {x}")
log = pathlib.Path(sys.argv[5]).read_text()
actual_kinds = collections.Counter(re.findall(
    r"\[dense streamk decomposition\] actual=(StreamK|DataParallel|SplitK)\b", log))
actual_streamk = actual_kinds["StreamK"]
actual_dp = actual_kinds["DataParallel"]
exercised = len(re.findall(r"\[dense streamk split gate\] EXERCISED\b", log))
not_exercised = len(re.findall(r"\[dense streamk split gate\] NOT EXERCISED\b", log))
exact_fixture = len(re.findall(
    r"\[streamk fixture exactness\] fixture=streamk-sweep-gs32-exact "
    r"shape=2048x4096x4096 .* -> ORDER-INDEPENDENT\+FP16-EXACT", log))
# Search launches every candidate once per pass and launches the selected
# leader once more after ranking.  A durable sample is legal only if its own
# lowered scheduler contained a real peer seam.
if sum(actual_kinds.values()) != len(attempts) + 1 or actual_kinds["SplitK"] != 0:
    raise SystemExit(
        f"[q4k65-streamk-sweep] FAIL: decomposition witnesses={dict(actual_kinds)}, "
        f"expected {len(attempts)+1} StreamK/DataParallel and zero fixed-SplitK")
# Sampled rows plus the final leader must be genuine StreamK.  An excluded
# no-seam row may lower either to a StreamK whole-K assignment or to pure DP;
# both are recorded as NOT EXERCISED rather than being mislabeled as samples.
if actual_streamk < len(samples) + 1 or actual_dp > len(excluded):
    raise SystemExit(
        f"[q4k65-streamk-sweep] FAIL: decomposition/outcome binding disagrees "
        f"StreamK={actual_streamk}>={len(samples)+1} DataParallel={actual_dp}<={len(excluded)}")
if exact_fixture != len(attempts) + 1:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: exact-fixture witnesses={exact_fixture}, expected {len(attempts)+1}")
if exercised != len(samples) + 1 or not_exercised != len(excluded):
    raise SystemExit(
        f"[q4k65-streamk-sweep] FAIL: split-gate/sample binding disagrees "
        f"exercised={exercised}/{len(samples)+1} not_exercised={not_exercised}/{len(excluded)}")
summary = {"reps": expected_reps, "attempts": len(attempts), "samples": len(samples), "exclusions": len(excluded), "successful_configs": len(sampled_by_pass[0]), "excluded_configs": len(excluded_by_pass[0])}
outpath.write_text(json.dumps(summary, indent=2) + "\n")
print("[q4k65-streamk-sweep] sample closure PASS: " + " ".join(f"{k}={v}" for k, v in summary.items()))
PY
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-streamk-sweep] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi

  if ! python3 "$root/benchmarks/analyse.py" "$samples" >"$analysis_log" 2>&1; then
    cat "$analysis_log" >&2
    printf '[q4k65-streamk-sweep] FAIL: common analyser refused the durable sample set\n' >&2
    return 1
  fi
  if ! python3 "$root/benchmarks/analyse.py" "$samples" --json >"$analysis_json" 2>&1; then
    cat "$analysis_json" >&2
    printf '[q4k65-streamk-sweep] FAIL: common JSON analyser refused the durable sample set\n' >&2
    return 1
  fi
  cat "$analysis_log"
  python3 - "$analysis_json" <<'PY' || return 1
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
if len(rows) != 1:
    raise SystemExit(f"[q4k65-streamk-sweep] FAIL: analyser returned {len(rows)} fixtures")
r = rows[0]
if not r.get("ranked") or r.get("passes", 0) < 2:
    raise SystemExit("[q4k65-streamk-sweep] FAIL: analyser did not establish a repeated ranking")
state = "SEPARATED" if not r.get("ties") else "UNRESOLVED"
delta = float(r["median"]) / 209.30 - 1.0
print(f"Q4K65_STREAMK_SWEEP {state} leader={r['leader']} median={r['median']:.6f}_us "
      f"band=[{r['band'][0]:.6f},{r['band'][1]:.6f}]_us ties={len(r.get('ties', []))} "
      f"current_exact_normal_context=209.300000_us context_delta={delta:+.3%} context_is_gate=0")
PY
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4k65-streamk-sweep] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi

  {
    printf 'source_rows=%s\neligible_rows=%s\nfiltered_rows=%s\n' "$source_rows" "$eligible_rows" "$filtered_rows"
    printf 'samples_sha256=%s\n' "$(sha256sum "$samples" | awk '{print $1}')"
    printf 'sweep_log_sha256=%s\n' "$(sha256sum "$sweep_log" | awk '{print $1}')"
    printf 'analysis_json_sha256=%s\n' "$(sha256sum "$analysis_json" | awk '{print $1}')"
  } >>"$manifest"
  find "$out" -type f ! -path "$out/build/*" ! -name bundle-files.sha256 -print0 | \
    sort -z | xargs -0 sha256sum >"$out/bundle-files.sha256"
  printf '[q4k65-streamk-sweep] PASS; artifacts: %s\n' "$out"
}

main "$@"
