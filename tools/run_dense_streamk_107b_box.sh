#!/usr/bin/env bash
# #107b's dense-only PPU gate.  This does not run or update any grouped/MoE
# number: it proves the mixed-input Stream-K wiring, its absolute-K seam, and
# the 128-thread fixup cohort before a future grouped scheduler is attempted.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_streamk_ab
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-dense-streamk-107b.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/build.log"
LIST_LOG="$ARTIFACT_ROOT/config.log"
GATE_LOG="$ARTIFACT_ROOT/seam-gate.log"
NP_LOG="$ARTIFACT_ROOT/a0-non-persistent.log"
P_LOG="$ARTIFACT_ROOT/a0-persistent.log"
SK_LOG="$ARTIFACT_ROOT/a0-streamk.log"
SPLIT_REPEAT_LOG="$ARTIFACT_ROOT/a0-split-repeat.log"

fail() {
  printf '[107b] FAIL: %s\n' "$*" >&2
  printf '[107b] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[107b] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

run_case() {
  local label="$1" log="$2"
  shift 2
  printf '\n== %s ==\n' "$label"
  if ! "$BIN" "$@" 2>&1 | tee "$log"; then
    fail "$label returned nonzero"
  fi
  grep -q '^  Disposition: Passed$' "$log" \
    || fail "$label did not report a passed numerical check"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 ' "$log" \
    || fail "$label did not report 20 distinct event-pair kernel spans"
}

require_verify_buckets() {
  local label="$1" log="$2"
  local bucket
  for bucket in DP SK-whole SK-split; do
    [ "$(grep -Ec "^  \\[dense verify bucket=${bucket}\\] tiles=[0-9]+ outputs=[0-9]+ mismatches=[0-9]+ max_abs=[^ ]+ max_rel_sym=[^ ]+ max_half_ulp=[0-9]+ nonfinite=[0-9]+$" "$log")" -eq 1 ] \
      || fail "$label did not report exactly one complete ${bucket} error bucket"
  done
}

# A0 is deliberately diagnostic: the device result that opened this item was
# Failed, and accepting only rc=0 would delete the evidence we are here to
# classify.  Accept only the program's two documented verdict exits, require a
# complete verdict plus all three buckets, and reject crashes or partial logs.
run_diagnostic_case() {
  local label="$1" log="$2"
  shift 2
  printf '\n== %s ==\n' "$label"
  set +e
  "$BIN" "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    fail "$label exited rc=$rc instead of a numerical Passed/Failed verdict"
  fi
  [ "$(grep -Ec '^  Disposition: (Passed|Failed)$' "$log")" -eq 1 ] \
    || fail "$label did not report exactly one numerical disposition"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 ' "$log" \
    || fail "$label did not report 20 distinct event-pair kernel spans"
  require_verify_buckets "$label" "$log"
}

# A split-path gate has three outcomes: numerical PASS/FAIL after exercising a
# real peer seam, or rc=2/NOT EXERCISED.  The last outcome is useful while
# searching runtime-dependent shapes, but it is never accepted as evidence.
run_split_probe() {
  local label="$1" log="$2"
  shift 2
  printf '\n== %s ==\n' "$label"
  set +e
  "$BIN" "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 2 ]; then
    python3 - "$log" <<'PY' || fail "$label malformed its NOT EXERCISED result"
import pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
part = re.findall(
    r"\[dense verify partition\] DP=(\d+) SK-whole=(\d+) SK-split=(\d+) "
    r"peer_excess=(\d+) valid_fixup_elements=(\d+) qk_cells=(\d+) coverage=exact-once",
    text)
gate = re.findall(
    r"\[dense streamk split gate\] NOT EXERCISED real_cu=(\d+) "
    r"ctas_per_cu=(\d+) workers=(\d+) logical_cta=(\d+) "
    r"logical_cta%workers=(\d+)%(\d+)=(\d+) SK-split=(\d+) "
    r"peer_excess=(\d+) reason=([a-z-]+)", text)
if len(part) != 1 or len(gate) != 1:
    raise SystemExit("missing or duplicate partition/NOT EXERCISED record")
dp, whole, split, peers, valid, cells = map(int, part[0])
cu, ctas, workers, tiles, lhs, rhs, rem, gate_split, gate_peers, reason = gate[0]
cu, ctas, workers, tiles, lhs, rhs, rem, gate_split, gate_peers = map(
    int, (cu, ctas, workers, tiles, lhs, rhs, rem, gate_split, gate_peers))
if not (workers == cu * ctas and tiles == lhs and workers == rhs and
        rem == tiles % workers and split == peers == gate_split == gate_peers == 0 and
        valid == 0):
    raise SystemExit(f"inconsistent empty seam: part={part[0]} gate={gate[0]}")
if rem == 0 and reason != "complete-worker-waves":
    raise SystemExit(f"divisible tile count has wrong reason {reason}")
if text.count("  Disposition: NOT EXERCISED") != 1 or re.search(
        r"^  Disposition: (Passed|Failed)", text, re.M):
    raise SystemExit("NOT EXERCISED was collapsed into a numerical disposition")
print(f"[107b][split-search] rejected empty seam tiles={tiles} workers={workers} "
      f"remainder={rem}")
PY
    return 2
  fi
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    fail "$label exited rc=$rc instead of Passed/Failed/NOT EXERCISED"
  fi
  [ "$(grep -Ec '^  Disposition: (Passed \(StreamK same-order partial replay bit-exact; any ordinary-reference differences are reassociation\)|Failed \(StreamK same-order partial replay did not close\))$' "$log")" -eq 1 ] \
    || fail "$label did not report exactly one exercised numerical disposition"
  if grep -q '^  Disposition: NOT EXERCISED$' "$log"; then
    fail "$label returned a numerical rc but printed NOT EXERCISED"
  fi
  require_verify_buckets "$label" "$log"
  [ "$(grep -Ec '^  \[dense verify fingerprint\] comparator_positions=[0-9]+ position_fnv1a=[0-9a-f]{16} value_fnv1a=[0-9a-f]{16} .*' "$log")" -eq 1 ] \
    || fail "$label did not report exactly one mismatch-position fingerprint"
  python3 - "$log" "$rc" <<'PY' || fail "$label did not prove a real split path"
import pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
rc = int(sys.argv[2])
part = re.findall(
    r"\[dense verify partition\] DP=(\d+) SK-whole=(\d+) SK-split=(\d+) "
    r"peer_excess=(\d+) valid_fixup_elements=(\d+) qk_cells=(\d+) coverage=exact-once",
    text)
gate = re.findall(
    r"\[dense streamk split gate\] EXERCISED real_cu=(\d+) ctas_per_cu=(\d+) "
    r"workers=(\d+) logical_cta=(\d+) logical_cta%workers=(\d+)%(\d+)=(\d+) "
    r"SK-split=(\d+) peer_excess=(\d+)", text)
if len(part) != 1 or len(gate) != 1:
    raise SystemExit("missing or duplicate exercised partition/gate record")
dp, whole, split, peers, valid, cells = map(int, part[0])
cu, ctas, workers, tiles, lhs, rhs, rem, gate_split, gate_peers = map(int, gate[0])
if not (workers == cu * ctas and tiles == lhs and workers == rhs and
        rem == tiles % workers and split == gate_split > 0 and
        peers == gate_peers > 0 and valid > 0):
    raise SystemExit(f"split witness did not close: part={part[0]} gate={gate[0]}")
replay = re.findall(
    r"\[streamk same-order replay\] split_tiles=(\d+) peers=(\d+) "
    r"split_outputs=(\d+) capture_scalars=(\d+) capture_holes=(\d+) "
    r"bad_slot_visits=(\d+) bad_k_counts=(\d+) "
    r"capture_vs_normal_bitdiff=(\d+) device_replay_bitdiff=(\d+) "
    r"non_split_reference_mismatches=(\d+) non_split_reference_bitdiff=(\d+) "
    r"reference_raw_bitdiff=(\d+) "
    r"triangle=(CLOSED|OPEN) (BIT-EXACT/PASS|MISMATCH/FAIL)", text)
if len(replay) != 1:
    raise SystemExit("missing or duplicate same-order replay record")
passed_disposition = "Disposition: Passed (StreamK same-order partial replay bit-exact; " in text
failed_disposition = "Disposition: Failed (StreamK same-order partial replay did not close)" in text
*counts, triangle, replay_verdict = replay[0]
counts = list(map(int, counts))
if (rc == 0) != passed_disposition or (rc == 1) != failed_disposition:
    raise SystemExit(
        f"process status/disposition disagree: rc={rc} passed={passed_disposition} "
        f"failed={failed_disposition}")
if rc == 0 and not (
        counts[0] == split and counts[1] == split + peers and counts[2] > 0 and
        counts[3] > 0 and counts[4:11] == [0, 0, 0, 0, 0, 0, 0] and
        triangle == "CLOSED" and replay_verdict == "BIT-EXACT/PASS"):
    raise SystemExit(f"Passed disposition is not backed by a closed replay: {replay[0]}")
if rc == 1 and replay_verdict != "MISMATCH/FAIL":
    raise SystemExit(f"Failed disposition lacks a failed replay: {replay[0]}")
print(f"[107b][split-path] EXERCISED tiles={tiles} workers={workers} "
      f"split_tiles={split} peer_excess={peers} replay={replay_verdict}")
PY
  # rc=1 is evidence that the exercised kernel failed, not another shape to
  # search past.  Only rc=2/NOT EXERCISED may advance the adaptive search.
  return "$rc"
}

printf '[107b] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[107b] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[107b] artifacts=%s\n' "$ARTIFACT_ROOT"

printf '\n== build isolated ppu001 four-warp target ==\n'
if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    QUANT=int4 BENCH_GS=128 "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'ppu001 dense Stream-K build failed'
fi

BUILD_MAKE="$(find_one 'Stream-K build.make' "$BUILD_ROOT" -type f \
    -path "*${TARGET}.dir/build.make")"
mapfile -t ARCHS < <(grep -oE -- '-arch=ppu_[0-9]+' "$BUILD_MAKE" | sort -u)
printf '[107b][arch] build.make=%s\n' "$BUILD_MAKE"
printf '[107b][arch] unique hgcc arch flags:'
printf ' %s' "${ARCHS[@]:-<none>}"
printf '\n'
if [ "${#ARCHS[@]}" -ne 1 ] || [ "${ARCHS[0]}" != '-arch=ppu_10' ]; then
  fail 'target must contain only -arch=ppu_10'
fi

BIN="$(find_one 'dense Stream-K binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
if ! "$BIN" --list_configs 2>&1 | tee "$LIST_LOG"; then
  fail '--list_configs failed'
fi
grep -Eq '^  .*tile 64x128x64  warp 64x32  stages 2$' "$LIST_LOG" \
  || fail 'isolated target does not expose exactly the reviewed tile geometry'
if [ "$(grep -Ec '^  .*tile [0-9]+x[0-9]+x[0-9]+  warp [0-9]+x[0-9]+  stages [0-9]+$' "$LIST_LOG")" -ne 1 ]; then
  fail 'isolated target exposes more or fewer than one tactic row'
fi

# This shape has one output tile and 68 K tiles.  With the scheduler's minimum
# eight-tile slice, forced Stream-K must make 8 ordered contributors.  The
# seams after tiles 9 and 27 are inside gs128 groups (TileK=64), so restarting
# metadata at local K=0 cannot pass this gate.
run_case 'exact absolute-K/fixup seam gate plus repeated-launch lock gate' "$GATE_LOG" \
  --m=64 --n=128 --k=4352 --l=1 --g=128 --mode=1 \
  --alpha=.75 --beta=.5 --iterations=20 --streamk_gate

python3 - "$GATE_LOG" <<'PY' || fail 'exact seam/decomposition audit failed'
import pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
d = re.search(
    r"\[dense streamk decomposition\] actual=(\w+) real_cu=(\d+) ctas_per_cu=(\d+) "
    r"workers=(\d+) scheduler_workers=(\d+) sk_tiles=(\d+) sk_units=(\d+) "
    r"dp_units=(\d+) units=(\d+) splits=(\d+) separate=(\d+)", text)
if not d:
    raise SystemExit("missing decomposition witness")
actual, cu, ctas, workers, sched, sk_tiles, sk_units, dp, units, splits, separate = d.groups()
nums = list(map(int, (cu, ctas, workers, sched, sk_tiles, sk_units, dp, units, splits, separate)))
cu, ctas, workers, sched, sk_tiles, sk_units, dp, units, splits, separate = nums
if not (actual == "StreamK" and workers == cu * ctas == sched and
        (sk_tiles, sk_units, dp, units, splits, separate) == (1, 8, 0, 8, 1, 0)):
    raise SystemExit(f"wrong decomposition: {d.group(0)}")

g = re.search(
    r"\[dense scheduler=streamk\].*grid=\((\d+),(\d+),(\d+)\) physical_cta=(\d+) "
    r"block_threads=(\d+)", text)
if not g:
    raise SystemExit("missing physical-grid witness")
gx, gy, gz, physical, threads = map(int, g.groups())
if gx * gy * gz != physical or physical != workers or threads != 128:
    raise SystemExit(f"grid/barrier mismatch: {g.group(0)} workers={workers}")

required = (
    "[streamk witness] fixup_work=8 epilogue_work=1 separate_reduction_work=0",
    "[streamk sequential CPU-FP32 fixture] order=k-ascending dyadic=1 outputs=8192 bad=0 bitdiff=0",
    "[streamk same-order replay] split_tiles=1 peers=8 split_outputs=8192",
    "capture_holes=0 bad_slot_visits=0 bad_k_counts=0 capture_vs_normal_bitdiff=0 device_replay_bitdiff=0",
    "non_split_reference_mismatches=0 non_split_reference_bitdiff=0",
    "triangle=CLOSED BIT-EXACT/PASS",
    "BIT-EXACT",
    "distinct-event-pairs=20 warmup-event-pairs=1 includes-launch-idle=1 lock-reset-before-start=1",
    "StreamK-C valid_elements=",
    "MODEL-ONLY/not-a-DRAM-counter",
)
missing = [x for x in required if x not in text]
if missing:
    raise SystemExit("missing exact gate marker(s): " + ", ".join(missing))
print(f"[107b][gate] PASS workers={workers} decomposition=1x8 witness=8/1/0 grid={gx}x{gy}x{gz}")
PY

# Same binary, geometry, event protocol, and 20-launch median for all three
# dense arms.  Fixed A0 remains a performance comparison.  It is deliberately
# not the split-path correctness gate: on a worker count that divides its 1024
# output tiles, its Stream-K arm contains no peer seam.
COMMON_SHAPE=(--m=2048 --n=4096 --k=4096 --l=1 --g=128 --mode=1)
COMMON=("${COMMON_SHAPE[@]}" --iterations=20)
run_case 'A0 non-persistent control' "$NP_LOG" "${COMMON[@]}"
run_case 'A0 serial-persistent control' "$P_LOG" "${COMMON[@]}" --persistent
run_diagnostic_case 'A0 Stream-K performance diagnostic (not split gate)' \
  "$SK_LOG" "${COMMON[@]}" --streamk

require_verify_buckets 'A0 non-persistent control' "$NP_LOG"
require_verify_buckets 'A0 serial-persistent control' "$P_LOG"
grep -Eq '^  \[dense verify partition\] DP=[0-9]+ SK-whole=[0-9]+ SK-split=[0-9]+ peer_excess=[0-9]+ valid_fixup_elements=[0-9]+ qk_cells=[0-9]+ coverage=exact-once$' "$SK_LOG" \
  || fail 'A0 Stream-K did not prove an exact scheduler-derived DP/SK partition'

grep -q 'lock-reset-before-start=0' "$NP_LOG" || fail 'non-persistent timing identity drifted'
grep -q 'lock-reset-before-start=0' "$P_LOG" || fail 'persistent timing identity drifted'
grep -q 'lock-reset-before-start=1' "$SK_LOG" || fail 'Stream-K did not reset locks outside every event'
grep -q 'actual=StreamK' "$SK_LOG" || fail 'A0 Stream-K silently fell back to another decomposition'
grep -q 'StreamK-C valid_elements=' "$SK_LOG" \
  || fail 'A0 Stream-K did not surface its per-q partial-C model'
grep -q 'MODEL-ONLY/not-a-DRAM-counter' "$SK_LOG" \
  || fail 'A0 Stream-K mislabeled logical partial-C accesses as measured DRAM traffic'

# Choose a correctness shape from the runtime worker count, then let the
# lowered scheduler be the final oracle.  Filtering out exact worker-wave
# multiples avoids the known empty case; rc=2 advances to the next N if a
# future MinIters/grouping policy still produces no peer seam.
WORKERS="$(python3 - "$SK_LOG" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
hits = re.findall(r"\[dense streamk decomposition\].* workers=(\d+) ", text)
if len(hits) != 1 or int(hits[0]) <= 0:
    raise SystemExit("cannot recover one positive runtime worker count")
print(hits[0])
PY
)" || fail 'could not derive runtime workers for the adaptive split gate'

mapfile -t SPLIT_CANDIDATES < <(python3 - "$WORKERS" <<'PY'
import sys
workers = int(sys.argv[1])
tiles_m, tile_n = 32, 128       # reviewed 64x128 A0 tactic, M=2048
for n_tiles in range(32, 32 + 64):
    tiles = tiles_m * n_tiles
    if tiles > workers and tiles % workers:
        print(n_tiles * tile_n)
PY
)
[ "${#SPLIT_CANDIDATES[@]}" -gt 0 ] \
  || fail "adaptive split search generated no candidate for workers=$WORKERS"

SPLIT_N=''
SPLIT_LOG=''
for candidate_n in "${SPLIT_CANDIDATES[@]}"; do
  candidate_log="$ARTIFACT_ROOT/a0-split-n${candidate_n}.log"
  if run_split_probe "adaptive split-path probe N=${candidate_n}" "$candidate_log" \
      --m=2048 --n="$candidate_n" --k=4096 --l=1 --g=128 --mode=1 \
      --iterations=0 --streamk_split_gate; then
    SPLIT_N="$candidate_n"
    SPLIT_LOG="$candidate_log"
    break
  else
    probe_rc=$?
    [ "$probe_rc" -eq 2 ] \
      || fail "adaptive split-path probe N=${candidate_n} failed unexpectedly rc=$probe_rc"
  fi
done
[ -n "$SPLIT_N" ] && [ -n "$SPLIT_LOG" ] \
  || fail "no tested N exercised a split path for workers=$WORKERS"

if run_split_probe "adaptive split-path repeat N=${SPLIT_N}" "$SPLIT_REPEAT_LOG" \
    --m=2048 --n="$SPLIT_N" --k=4096 --l=1 --g=128 --mode=1 \
    --iterations=0 --streamk_split_gate; then
  :
else
  repeat_rc=$?
  fail "selected split shape N=${SPLIT_N} was not repeatably exercised rc=$repeat_rc"
fi

python3 - "$SPLIT_LOG" "$SPLIT_REPEAT_LOG" <<'PY' || fail 'split-path mismatch fingerprint comparison failed'
import pathlib, re, sys

pat = re.compile(
    r"\[dense verify fingerprint\] comparator_positions=(\d+) "
    r"position_fnv1a=([0-9a-f]{16}) value_fnv1a=([0-9a-f]{16})")
rows = []
for path in sys.argv[1:]:
    hits = pat.findall(pathlib.Path(path).read_text())
    if len(hits) != 1:
        raise SystemExit(f"expected one fingerprint in {path}, got {len(hits)}")
    rows.append(hits[0])
same_positions = rows[0][:2] == rows[1][:2]
same_values = rows[0] == rows[1]
verdict = "STABLE_POSITIONS_AND_VALUES" if same_values else (
    "STABLE_POSITIONS_VALUE_DRIFT" if same_positions else "POSITION_DRIFT")
print(f"[107b][split fingerprint] run1={rows[0]} run2={rows[1]} verdict={verdict}")
PY

python3 - "$NP_LOG" "$P_LOG" "$SK_LOG" <<'PY' || fail 'A0 median extraction failed'
import pathlib, re, sys

vals = {}
for label, path in zip(("non-persistent", "persistent", "streamk"), sys.argv[1:]):
    text = pathlib.Path(path).read_text()
    m = re.search(r"\[dense kernel-span-upper\] n=20 median=([0-9.]+) us", text)
    if not m:
        raise SystemExit(f"no median for {label}")
    vals[label] = float(m.group(1))
if min(vals.values()) <= 0:
    raise SystemExit(f"non-positive median: {vals}")
print("[107b][A0] kernel-span medians: " +
      " ".join(f"{k}={v:.3f}us" for k, v in vals.items()))
print(f"[107b][A0] persistent/non-persistent={vals['persistent']/vals['non-persistent']:.6f} "
      f"streamk/non-persistent={vals['streamk']/vals['non-persistent']:.6f}")
PY

printf '\n[107b] PASS: dense absolute-K/fixup/worker seam and repeated launches proved on ppu001\n'
printf '[107b] COLLECTED: fixed-A0 performance plus runtime-adaptive split-path correctness at N=%s\n' "$SPLIT_N"
printf '[107b] SPLIT-PATH: SK-split>0 and peer_excess>0 were prerequisites; correctness is the printed disposition, never an empty PASS\n'
printf '[107b] NOTE: A0 ratios are dense diagnostics; no grouped/MoE result is changed or claimed\n'
printf '[107b] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
