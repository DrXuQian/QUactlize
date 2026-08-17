#!/usr/bin/env bash
# Reproduce the historical Q4_K scale-first dense prefill anchor first, then
# compare normal and forced hybrid Stream-K with one exact fixture and one
# compiled collective/tactic.  This script changes no production selector.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_streamk_q4k65_ab
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_ROOT="${OUT:-/workspace/quactlize-dense-streamk-q4k65-${SHA:0:8}-${STAMP}}"
BUILD_ROOT="$ARTIFACT_ROOT/build"
BUILD_LOG="$ARTIFACT_ROOT/build.log"
LIST_LOG="$ARTIFACT_ROOT/config.log"
BASELINE_LOG="$ARTIFACT_ROOT/normal-historical-fixture.log"
NORMAL_LOG="$ARTIFACT_ROOT/normal-exact-fixture.log"
STREAMK_LOG="$ARTIFACT_ROOT/streamk-exact-fixture.log"
# The public report tag contains spaces, but cutlass::CommandLine tokenizes the
# value again.  Use the retained compact input alias; the output is still
# required below to print the canonical `64x64:64 w64x32 s3 bc0->0` identity.
CONFIG='64x64x64:64x32:s3:bc0->0'
CONFIG_LABEL='64x64:64 w64x32 s3 bc0->0'
CONTINUE_ON_BASELINE_DRIFT="${CONTINUE_ON_BASELINE_DRIFT:-0}"
STREAMK_POLICY="${STREAMK_POLICY:-two-wave}"
STREAMK_POLICY_FLAG=()

fail() {
  printf '[q4k65-streamk] FAIL: %s\n' "$*" >&2
  printf '[q4k65-streamk] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

case "$ARTIFACT_ROOT" in
  /workspace/*) ;;
  *) fail "OUT must name a directory below /workspace (got $ARTIFACT_ROOT)" ;;
esac
case "$CONTINUE_ON_BASELINE_DRIFT" in
  0|1) ;;
  *) fail "CONTINUE_ON_BASELINE_DRIFT must be 0 or 1" ;;
esac
case "$STREAMK_POLICY" in
  two-wave) ;;
  tail-only) STREAMK_POLICY_FLAG=(--streamk-tail-only) ;;
  tail-min-peers) STREAMK_POLICY_FLAG=(--streamk-tail-min-peers) ;;
  *) fail "STREAMK_POLICY must be two-wave, tail-only, or tail-min-peers" ;;
esac

mkdir -p "$ARTIFACT_ROOT" "$BUILD_ROOT"

printf '[q4k65-streamk] sha=%s artifacts=%s\n' "$SHA" "$ARTIFACT_ROOT"
printf '[q4k65-streamk] policy=%s\n' "$STREAMK_POLICY"
printf '[q4k65-streamk] anchor=scale_first M=2048 N=4096 K=4096 gs=32 input_cfg=%s report_cfg=%s historical=209.27us/65.7%%MFU\n' \
  "$CONFIG" "$CONFIG_LABEL"

if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    QUANT=int4 BENCH_GS=32 "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'isolated Q4_K65 normal/Stream-K target did not build'
fi

mapfile -t BINS < <(find "$BUILD_ROOT" -type f -name "$TARGET" -perm -u+x -print)
if [ "${#BINS[@]}" -ne 1 ]; then
  printf '[q4k65-streamk] executable candidates (%d):\n' "${#BINS[@]}" >&2
  printf '  %s\n' "${BINS[@]:-<none>}" >&2
  fail 'expected exactly one executable'
fi
BIN="${BINS[0]}"
printf '[q4k65-streamk] binary=%s sha256=%s\n' "$BIN" "$(sha256sum "$BIN" | awk '{print $1}')"
if command -v hgcc >/dev/null 2>&1; then
  HGCC_VERSION="$(hgcc --version 2>&1 || true)"
  printf '[q4k65-streamk] hgcc=%s\n' "$(printf '%s\n' "$HGCC_VERSION" | sed -n '1p')"
else
  printf '[q4k65-streamk] hgcc=UNAVAILABLE_AFTER_BUILD\n'
fi

if ! "$BIN" --list_configs 2>&1 | tee "$LIST_LOG"; then
  fail '--list_configs failed'
fi
if [ "$(grep -Ec '^  .*tile 64x64x64  warp 64x32  stages 3  instruction=m16$' "$LIST_LOG")" -ne 1 ] ||
   [ "$(grep -Ec '^  .*tile [0-9]+x[0-9]+x[0-9]+  warp [0-9]+x[0-9]+  stages [0-9]+  instruction=m(8|16)$' "$LIST_LOG")" -ne 1 ]; then
  fail 'isolated binary is not exactly the historical 64x64x64/w64x32/s3 row'
fi

COMMON=(--m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1 \
        --alpha=1 --beta=0 --config="$CONFIG")

# Admission precedes the subject: if this exact normal kernel no longer
# reproduces the historical environment, a Stream-K delta cannot be attached
# to the 65.7% anchor and this run stops before launching the subject.
printf '\n== historical normal-scheduler admission ==\n'
if ! "$BIN" "${COMMON[@]}" --iterations=100 2>&1 | tee "$BASELINE_LOG"; then
  fail 'normal historical-fixture arm failed correctness or timing'
fi
python3 - "$BASELINE_LOG" "$CONTINUE_ON_BASELINE_DRIFT" <<'PY' || \
  fail 'normal arm did not reproduce the registered 65% anchor'
import pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
continue_on_drift = sys.argv[2] == "1"
aggregates = re.findall(
    r"\[dense historical aggregate\] n=100 average=([0-9.]+) us "
    r"protocol=PpuTimer-aggregate reference=209\.27us", text)
spans = re.findall(r"\[dense kernel-span-upper\]", text)
rows = re.findall(
    r"\[CUTLASS w4 gs=32 cfg=64x64:64 w64x32 s3 bc0->0 "
    r"scheduler=non-persistent\] M=2048\s+([0-9.]+) us", text)
occupancy = re.findall(
    r"\[dense scheduler=non-persistent\] logical_cta=2048 cu=(\d+) "
    r"occupancy_api=(\d+)", text)
if (len(aggregates) != 1 or len(spans) != 0 or len(rows) != 1 or
        len(occupancy) != 1 or
        "  Disposition: Passed" not in text):
    raise SystemExit(
        "historical arm must have one aggregate timing, zero per-launch medians, "
        "and unique config/occupancy/PASS evidence")
cu, ctas_per_cu = map(int, occupancy[0])
if cu != 72:
    raise SystemExit(f"expected historical 72-CU box, measured cu={cu}; refusing cross-device admission")
us = float(aggregates[0])
if abs(us - float(rows[0])) > 0.01:
    raise SystemExit(f"normal timing lines disagree: aggregate={us} report={rows[0]}")
reference = 209.27
lo, hi = reference * 0.97, reference * 1.03
mfu = (2.0 * 2048 * 4096 * 4096) / (us * 1e-6) / 500.0e12 * 100.0
verdict = "ADMITTED" if lo <= us <= hi else "DRIFTED"
print(f"Q4K65_BASELINE measured_cu={cu} occupancy_api={ctas_per_cu} "
      f"runtime={us:.3f}_us MFU={mfu:.3f}% "
      f"registered=209.27_us/65.7% range=[{lo:.3f},{hi:.3f}] verdict={verdict}")
if verdict != "ADMITTED":
    if not continue_on_drift:
        raise SystemExit("normal baseline drifted beyond the preregistered +/-3% window")
    print("Q4K65_BASELINE_OVERRIDE historical_anchor=DRIFTED "
          "subject_result_scope=CURRENT-SHA-UNANCHORED")
PY

printf '\n== exact normal control ==\n'
if ! "$BIN" "${COMMON[@]}" --iterations=20 --streamk_exact_fixture \
    2>&1 | tee "$NORMAL_LOG"; then
  fail 'exact normal control failed'
fi

printf '\n== exact forced hybrid Stream-K subject ==\n'
if ! "$BIN" "${COMMON[@]}" --iterations=20 --streamk_exact_fixture --streamk \
    "${STREAMK_POLICY_FLAG[@]}" \
    2>&1 | tee "$STREAMK_LOG"; then
  fail 'forced Stream-K subject failed, was not exercised, or was not classifiable'
fi

python3 - "$NORMAL_LOG" "$STREAMK_LOG" "$BASELINE_LOG" "$STREAMK_POLICY" <<'PY' || \
  fail 'normal/Stream-K adjudication failed'
import math, pathlib, re, sys

normal = pathlib.Path(sys.argv[1]).read_text()
streamk = pathlib.Path(sys.argv[2]).read_text()
baseline = pathlib.Path(sys.argv[3]).read_text()
requested_policy = sys.argv[4]
exact = (r"\[streamk fixture exactness\] fixture=q4k65-exact "
         r"shape=2048x4096x4096 .* max\|D\|=2048 .* -> "
         r"ORDER-INDEPENDENT\+FP16-EXACT")
for label, text in (("normal", normal), ("streamk", streamk)):
    if len(re.findall(exact, text)) != 1:
        raise SystemExit(f"{label} lacks one invocation-bound exactness proof")
    if "Failed" in "\n".join(x for x in text.splitlines() if "Disposition:" in x):
        raise SystemExit(f"{label} printed a failed numerical disposition")

def timing(text, scheduler):
    spans = re.findall(r"\[dense kernel-span-upper\] n=20 median=([0-9.]+) us", text)
    rows = re.findall(
        rf"\[CUTLASS w4 gs=32 cfg=64x64:64 w64x32 s3 bc0->0 "
        rf"scheduler={scheduler}\] M=2048\s+([0-9.]+) us", text)
    if len(spans) != 1 or len(rows) != 1:
        raise SystemExit(f"{scheduler}: timing/config identity is not unique")
    if abs(float(spans[0]) - float(rows[0])) > 0.01:
        raise SystemExit(f"{scheduler}: timing lines disagree")
    return float(spans[0])

normal_us = timing(normal, "non-persistent")
streamk_us = timing(streamk, "streamk")
decomp = re.findall(
    r"\[dense streamk decomposition\] actual=(\w+) policy=(two-wave|tail-only|tail-min-peers) "
    r"real_cu=(\d+) occupancy_api=(\d+) blocks_per_cu=(\d+) "
    r"workers=(\d+) scheduler_workers=(\d+) "
    r"sk_tiles=(\d+) sk_units=(\d+) dp_units=(\d+) units=(\d+) "
    r"splits=(\d+) separate=(\d+) workspace=(\d+)", streamk)
partition = re.findall(
    r"\[dense verify partition\] DP=(\d+) SK-whole=(\d+) SK-split=(\d+) "
    r"peer_excess=(\d+) valid_fixup_elements=(\d+) qk_cells=(\d+) "
    r"coverage=exact-once", streamk)
normal_occ = re.findall(
    r"\[dense scheduler=non-persistent\] logical_cta=(\d+) cu=(\d+) "
    r"occupancy_api=(\d+) grid=\((\d+),(\d+),(\d+)\) physical_cta=(\d+)",
    normal)
if len(decomp) != 1 or len(partition) != 1 or len(normal_occ) != 1:
    raise SystemExit("normal occupancy or Stream-K decomposition/partition evidence is missing or duplicated")
actual, policy, cu, occupancy, cpcu, workers, sched_workers, sk_tiles, sk_units, dp_units, units, splits, separate, workspace = decomp[0]
cu, occupancy, cpcu, workers, sched_workers, sk_tiles, sk_units, dp_units, units, splits, separate, workspace = map(
    int, (cu, occupancy, cpcu, workers, sched_workers, sk_tiles, sk_units, dp_units, units, splits, separate, workspace))
dp, whole, split, peers, valid, qk = map(int, partition[0])
q = 2048
normal_q, normal_cu, normal_cpcu, gx, gy, gz, normal_physical = map(
    int, normal_occ[0])
normal_workers = normal_cu * normal_cpcu
normal_tail = q % normal_workers
normal_waves = math.ceil(q / normal_workers)
normal_tail_fill = 100.0 * normal_tail / normal_workers
normal_padding = 100.0 * (normal_waves * normal_workers - q) / q
tail = q % workers
waves = math.ceil(q / workers)
tail_fill = 100.0 * tail / workers
dp_padding = 100.0 * (waves * workers - q) / q
if not (normal_q == q == normal_physical and gx * gy * gz == q and
        actual == "StreamK" and policy == requested_policy and
        workers == cu * cpcu == sched_workers and cpcu == occupancy and
        sk_tiles > 0 and sk_units > 0 and split > 0 and peers > 0 and
        valid > 0 and splits == 1 and separate == 0):
    raise SystemExit(f"forced Stream-K was not genuinely exercised: decomp={decomp[0]} partition={partition[0]}")
if requested_policy in ("tail-only", "tail-min-peers"):
    common = {
        "sk_tiles": q % workers,
        "sk_units": workers,
        "dp_units": q - (q % workers),
        "units": (q - (q % workers)) + workers,
        "dp": q - (q % workers),
        "qk": (q % workers) * 64,
        "workspace": 5_244_160,
    }
    topology = ({
        "whole": 0,
        "split": q % workers,
        "peers": 456,
        "valid": 456 * 64 * 64,
    } if requested_policy == "tail-only" else {
        "whole": 64,
        "split": 256,
        "peers": 256,
        "valid": 256 * 64 * 64,
    })
    expected = {**common, **topology}
    got = {"sk_tiles": sk_tiles, "sk_units": sk_units,
           "dp_units": dp_units, "units": units, "dp": dp,
           "whole": whole, "split": split, "peers": peers,
           "valid": valid, "qk": qk, "workspace": workspace}
    if got != expected:
        raise SystemExit(
            f"{requested_policy} lowering is not the preregistered DP-major partition: "
            f"got={got} expected={expected}")
if "Disposition: Passed (whole-K reference bit-exact; fixup replay closed)" not in streamk:
    raise SystemExit("Stream-K lacks its raw-bit whole-K/fixup PASS")
flops = 2.0 * 2048 * 4096 * 4096
normal_mfu = flops / (normal_us * 1e-6) / 500.0e12 * 100.0
streamk_mfu = flops / (streamk_us * 1e-6) / 500.0e12 * 100.0
speedup = normal_us / streamk_us
baseline_match = re.findall(
    r"\[dense historical aggregate\] n=100 average=([0-9.]+) us ", baseline)
if len(baseline_match) != 1:
    raise SystemExit("historical aggregate identity is missing from final adjudication")
baseline_us = float(baseline_match[0])
anchor = "ADMITTED" if 209.27 * 0.97 <= baseline_us <= 209.27 * 1.03 else "DRIFTED"
print(f"Q4K65_GEOMETRY Q={q} "
      f"policy={policy} "
      f"normal_workers={normal_workers} normal_waves={normal_waves} "
      f"normal_tail={normal_tail}/{normal_workers} normal_tail_fill={normal_tail_fill:.3f}% "
      f"normal_dp_padding={normal_padding:.3f}% "
      f"streamk_workers={workers} streamk_waves={waves} "
      f"streamk_tail={tail}/{workers} streamk_tail_fill={tail_fill:.3f}% "
      f"streamk_dp_padding_counterfactual={dp_padding:.3f}% "
      f"dp_units={dp_units} sk_tiles={sk_tiles} sk_units={sk_units} "
      f"split_tiles={split} peer_excess={peers} workspace={workspace}")
print(f"Q4K65_AB normal={normal_us:.3f}_us/{normal_mfu:.3f}%MFU "
      f"streamk_{policy}={streamk_us:.3f}_us/{streamk_mfu:.3f}%MFU "
      f"speedup={speedup:.5f}x correctness=RAW-BIT/PASS "
      f"historical_anchor={anchor}")
winner = "STREAMK-WINS" if streamk_us < normal_us else "NORMAL-WINS"
scope = "ANCHORED" if anchor == "ADMITTED" else "CURRENT-SHA-UNANCHORED"
print(f"Q4K65_VERDICT {winner} scope={scope}")
PY

printf '[q4k65-streamk] PASS; artifacts: %s\n' "$ARTIFACT_ROOT"
