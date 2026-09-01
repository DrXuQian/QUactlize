#!/usr/bin/env bash
# Execute the five prebuilt FQ K-quant measurement pairs; never compile them.
set -euo pipefail
umask 022

fail() {
  printf '[prebuilt-fq-kquant] FAIL: %s\n' "$*" >&2
  exit 2
}

atomic_text() {
  local path="$1" value="$2" temporary="${1}.current.$$"
  printf '%s\n' "$value" > "$temporary"
  mv -f -- "$temporary" "$path"
}

run_committed() {
  local log="$1" digest_file="${1}.sha256" temporary failed digest rc
  shift
  [ ! -L "$log" ] && [ ! -L "$digest_file" ] || fail "phase path is symlinked: $log"
  if [ -s "$log" ] && [ -s "$digest_file" ]; then
    [ "$resume" = 1 ] || fail "phase exists without RESUME=1: $log"
    digest="$(sha256sum "$log" | awk '{print $1}')"
    [ "$(cat "$digest_file")" = "$digest" ] || fail "committed phase changed: $log"
    printf '[prebuilt-fq-kquant] reuse phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$digest_file" ]; then
    [ "$resume" = 1 ] || fail "phase residue exists without RESUME=1: $log"
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    [ ! -e "$log" ] || mv -- "$log" "$failed"
    [ ! -e "$digest_file" ] || mv -- "$digest_file" "${failed}.sha256"
  fi
  temporary="${log}.current.$$"
  set +e
  "$@" > "$temporary" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    failed="${log}.failed.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv -- "$temporary" "$failed"
    tail -n 160 "$failed" >&2
    fail "phase rc=$rc preserved=$failed"
  fi
  mv -- "$temporary" "$log"
  digest="$(sha256sum "$log" | awk '{print $1}')"
  atomic_text "$digest_file" "$digest"
}

case "${1:-}" in
  "") run_mode=execute ;;
  --verify-only) [ "$#" -eq 1 ] || fail 'accepts at most one mode'; run_mode=verify ;;
  --preflight-only) [ "$#" -eq 1 ] || fail 'accepts at most one mode'; run_mode=preflight ;;
  *) fail 'accepted modes are --verify-only and --preflight-only' ;;
esac

bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(realpath -e -- "$bundle/../../../..")"
manifest="$bundle/manifest.json"
python3 -B "$bundle/verify-bundle.py" "$manifest" --repo-root "$repo"
[ "$run_mode" != verify ] || exit 0

for variable in PPU_DEFS PPU_EXTRA_DEFS PPU_BUILD_DIR TARGET FQ_KQUANT_PERF_QTYPE; do
  [ -z "${!variable-}" ] || fail "ambient $variable is forbidden for a prebuilt execution"
done

resume="${RESUME:-0}"
profile="${SWEEP_PROFILE:-heuristic}"
all_configs="${SWEEP_CONFIGS:-1}"
iterations="${PERF_ITERATIONS:-11}"
warmups="${PERF_WARMUPS:-3}"
rounds="${PERF_ROUNDS:-3}"
threshold="${REGRESSION_THRESHOLD_PCT:-3.0}"
max_leaves="${HEURISTIC_MAX_LEAVES:-8}"
min_leaf_rows="${HEURISTIC_MIN_LEAF_ROWS:-2}"
min_leaf_families="${HEURISTIC_MIN_LEAF_FAMILIES:-1}"

case "$resume:$all_configs" in 0:1|1:1) ;; *) fail 'RESUME must be 0/1 and SWEEP_CONFIGS must be 1';; esac
case "$profile" in heuristic|layout-ab) ;; *) fail 'SWEEP_PROFILE must be heuristic or layout-ab';; esac
case "$iterations:$warmups:$rounds:$max_leaves:$min_leaf_rows:$min_leaf_families" in
  *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0:*|*:*:*:*:0:*|*:*:*:*:*:0)
    fail 'measurement and heuristic integer controls must be positive';;
esac
[ "$rounds" -ge 2 ] || fail 'PERF_ROUNDS must be at least 2 for alternating order'
python3 -B - "$threshold" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 < value < 100.0:
    raise SystemExit("threshold must be finite and between 0 and 100")
PY

[ -n "${PPU_SDK:-}" ] || fail 'PPU_SDK is required for runtime identity and libraries'
[ ! -L "$PPU_SDK" ] || fail 'PPU_SDK must not be a symbolic link'
sdk_root="$(realpath -e -- "$PPU_SDK")"
[ -d "$sdk_root" ] || fail 'PPU_SDK must resolve to a directory'
runtime_dir="$(realpath -e -- "$sdk_root/lib")"
[ -d "$runtime_dir" ] && [ ! -L "$runtime_dir" ] || fail 'SDK runtime lib must be a real directory'
if [ -n "${PPU_RUNTIME_LIB_DIR:-}" ]; then
  [ "$(realpath -e -- "$PPU_RUNTIME_LIB_DIR")" = "$runtime_dir" ] || \
    fail 'PPU_RUNTIME_LIB_DIR must equal PPU_SDK/lib'
fi

runtime_preflight_current=""
sdk_identity_current="$(mktemp "${TMPDIR:-/tmp}/fq-kquant-sdk-identity.XXXXXX.json")" || \
  fail 'cannot allocate SDK identity staging file'
[ -n "$sdk_identity_current" ] && [ -f "$sdk_identity_current" ] && \
  [ ! -L "$sdk_identity_current" ] || fail 'SDK identity staging file is invalid'
cleanup_sdk_identity() {
  [ ! -e "$sdk_identity_current" ] || rm -f -- "$sdk_identity_current"
  [ -z "$runtime_preflight_current" ] || [ ! -e "$runtime_preflight_current" ] || \
    rm -f -- "$runtime_preflight_current"
}
trap cleanup_sdk_identity EXIT

python3 -B - <<'PY'
import os
import pathlib
import re

values = {}
for line in pathlib.Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
if values.get("ID") != "ubuntu" or values.get("VERSION_ID") != "24.04":
    raise SystemExit(
        f"execution requires Ubuntu 24.04, got {values.get('ID')} {values.get('VERSION_ID')}")
libc = os.confstr("CS_GNU_LIBC_VERSION") or ""
match = re.fullmatch(r"glibc ([0-9]+)\.([0-9]+)", libc)
if match is None or tuple(map(int, match.groups())) < (2, 38):
    raise SystemExit(f"execution requires glibc >=2.38, got {libc!r}")
print(f"[prebuilt-fq-kquant] host-floor PASS ubuntu=24.04 {libc}")
PY

python3 -B "$bundle/fq-kquant-sdk-identity.py" \
  --manifest "$manifest" --sdk-root "$sdk_root" --output "$sdk_identity_current"

runtime_tail="$runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for qtype in 10 11 12 13 14; do
  binary="$bundle/q$qtype/test_fq_kquant_layout_perf"
  library="$bundle/q$qtype/libquactlize_ppu.so"
  bin_dynamic="$(readelf -d "$binary")"
  lib_dynamic="$(readelf -d "$library")"
  readelf -h "$binary" | grep -Fq 'Class:                             ELF64' || \
    fail "q$qtype executable is not ELF64"
  readelf -h "$binary" | grep -Fq 'Machine:                           Advanced Micro Devices X86-64' || \
    fail "q$qtype executable host machine differs"
  for needed in libquactlize_ppu.so libhggc_wrapper.so libstdc++.so.6 libgcc_s.so.1 libc.so.6; do
    grep -Fq "Shared library: [$needed]" <<<"$bin_dynamic" || \
      fail "q$qtype executable misses NEEDED $needed"
  done
  grep -Fq 'Library rpath:' <<<"$bin_dynamic" && grep -Fq '$ORIGIN' <<<"$bin_dynamic" || \
    fail "q$qtype executable RPATH identity differs"
  grep -Fq 'Library soname: [libquactlize_ppu.so]' <<<"$lib_dynamic" && \
    grep -Fq '$ORIGIN' <<<"$lib_dynamic" || fail "q$qtype library dynamic identity differs"
  q_library_path="$(dirname "$library"):$runtime_tail"
  set +e
  loader="$(env LD_LIBRARY_PATH="$q_library_path" ldd "$binary" 2>&1)"
  loader_rc=$?
  set -e
  [ "$loader_rc" -eq 0 ] || fail "q$qtype loader closure rc=$loader_rc"
  ! grep -Fq 'not found' <<<"$loader" || \
    fail "q$qtype loader closure misses the runtime floor"
  grep -Fq "libquactlize_ppu.so => $(dirname "$library")/libquactlize_ppu.so" <<<"$loader" || \
    fail "q$qtype loader selected a mismatched quactlize library"
  grep -Fq "libhggc_wrapper.so => $runtime_dir/libhggc_wrapper.so" <<<"$loader" || \
    fail "q$qtype loader selected a mismatched SDK wrapper"
done
printf '[prebuilt-fq-kquant] elf-loader-preflight PASS pairs=5 floor=GLIBC_2.38/GLIBCXX_3.4.32\n'
[ "$run_mode" != preflight ] || exit 0

workspace="$(realpath -e -- "${WORKSPACE:-/workspace}")"
[ -d "$workspace" ] && [ ! -L "$workspace" ] || fail 'WORKSPACE must be a real directory'
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$(realpath -m -- "${OUT:-$workspace/quactlize-fq-kquant-prebuilt-2b513637-${stamp}-$$}")"
case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict WORKSPACE child';; esac
if [ -e "$out" ]; then
  [ "$resume" = 1 ] || fail "refusing existing OUT without RESUME=1: $out"
  [ -d "$out" ] && [ ! -L "$out" ] || fail 'existing OUT must be a real directory'
else
  [ "$resume" = 0 ] || fail 'RESUME=1 requires an existing OUT'
fi
mkdir -p "$out/inputs" "$out/results" "$out/runs"
[ -z "$(find "$out" -type l -print -quit)" ] || fail 'OUT contains a symbolic link'

planner="$repo/tools/plan_fq_kquant_kpack_perf.py"
analyzer="$repo/tools/analyze_fq_kquant_kpack_perf.py"
fitter="$repo/tools/fit_fq_kquant_config_heuristic.py"
case "$profile" in
  heuristic) plan_source="$bundle/plans/plan-heuristic.json" ;;
  layout-ab) plan_source="$bundle/plans/plan-layout-ab.json" ;;
esac
plan="$out/inputs/plan.json"
manifest_copy="$out/inputs/bundle-manifest.json"
if [ "$resume" = 0 ]; then
  install -m 0644 -- "$plan_source" "$plan"
  install -m 0644 -- "$manifest" "$manifest_copy"
else
  [ -f "$plan" ] && [ ! -L "$plan" ] && cmp -s -- "$plan_source" "$plan" || \
    fail 'resume plan differs from the admitted plan'
  [ -f "$manifest_copy" ] && [ ! -L "$manifest_copy" ] && \
    cmp -s -- "$manifest" "$manifest_copy" || fail 'resume manifest differs'
fi

runtime_preflight="$out/inputs/runtime-preflight.json"
runtime_preflight_digest="$out/inputs/runtime-preflight.sha256"
runtime_preflight_current="$out/inputs/.runtime-preflight.current.$$"
python3 -B - "$manifest" "$sdk_identity_current" "$runtime_preflight_current" <<'PY'
import hashlib
import json
import os
import pathlib
import tempfile
import sys

manifest_path = pathlib.Path(sys.argv[1])
sdk_identity_path = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
sdk_identity = json.loads(sdk_identity_path.read_text(encoding="utf-8"))
os_release = {}
for line in pathlib.Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        os_release[key] = value.strip().strip('"')
document = {
    "schema": "quactlize.fq-kquant-prebuilt-runtime-preflight.v2",
    "evidence_grade": sdk_identity["evidence_grade"],
    "host": {
        "distribution": os_release["ID"],
        "version_id": os_release["VERSION_ID"],
        "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
    },
    "runtime_floor": manifest["runtime"]["execution_floor"],
    "sdk_identity": sdk_identity,
    "loader_closures": {
        str(row["qtype"]): {
            "status": "PASS",
            "binary_sha256": row["binary"]["sha256"],
            "library_sha256": row["library"]["sha256"],
        }
        for row in manifest["pairs"]
    },
}
data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".",
                                 suffix=".current", delete=False) as stream:
    temporary = pathlib.Path(stream.name)
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY
current_preflight_sha="$(sha256sum "$runtime_preflight_current" | awk '{print $1}')"
if [ "$resume" = 0 ]; then
  [ ! -e "$runtime_preflight" ] && [ ! -e "$runtime_preflight_digest" ] || \
    fail 'fresh OUT already contains runtime preflight authority'
  mv -- "$runtime_preflight_current" "$runtime_preflight"
  atomic_text "$runtime_preflight_digest" \
    "$current_preflight_sha  runtime-preflight.json"
else
  [ -f "$runtime_preflight" ] && [ ! -L "$runtime_preflight" ] && \
    [ -f "$runtime_preflight_digest" ] && [ ! -L "$runtime_preflight_digest" ] || \
    fail 'resume runtime preflight authority is missing'
  expected_preflight_sha="$(sed -n 's/^\([0-9a-f]\{64\}\)  runtime-preflight\.json$/\1/p' \
    "$runtime_preflight_digest")"
  [ -n "$expected_preflight_sha" ] && \
    [ "$(wc -l < "$runtime_preflight_digest")" -eq 1 ] || \
    fail 'resume runtime preflight sidecar is malformed'
  [ "$(sha256sum "$runtime_preflight" | awk '{print $1}')" = "$expected_preflight_sha" ] || \
    fail 'resume runtime preflight digest differs'
  if ! cmp -s -- "$runtime_preflight_current" "$runtime_preflight"; then
    fail 'resume runtime preflight differs (host, SDK root, policy, or actual identity)'
  fi
  rm -f -- "$runtime_preflight_current"
fi

python3 -B "$planner" self-test
python3 -B "$analyzer" self-test >/dev/null
python3 -B "$fitter" self-test >/dev/null
python3 -B "$planner" validate --plan "$plan"

identity="$out/inputs/box-identity.json"
if [ "$resume" = 0 ]; then
  python3 -B - "$identity" "$repo" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import tempfile
import sys

output = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
external = os.environ.get("BOX_IDENTITY_JSON", "").strip()
if external:
    source = pathlib.Path(os.path.abspath(external))
    if source.is_symlink() or not source.is_file():
        raise SystemExit("BOX_IDENTITY_JSON must name a real regular JSON file")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"BOX_IDENTITY_JSON is invalid: {error}")
    sys.path.insert(0, str(repo / "tools"))
    import box_identity_schema
    try:
        box_identity_schema.validate(payload)
    except box_identity_schema.IdentityProbeError as error:
        raise SystemExit(f"BOX_IDENTITY_JSON fails the repository identity schema: {error}")
    document = {
        "schema": "quactlize.prebuilt-box-execution-identity.v1",
        "source": "external-json",
        "evidence_grade": "schema-validated",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "identity": payload,
    }
else:
    names = {
        "device_model": "QUACTLIZE_BOX_DEVICE_MODEL",
        "pci_identity": "QUACTLIZE_BOX_PCI_IDENTITY",
        "driver_version": "QUACTLIZE_BOX_DRIVER_VERSION",
        "sdk_compiler_identity": "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY",
    }
    identity = {}
    for field, variable in names.items():
        value = os.environ.get(variable, "").strip()
        if not value or any(mark in value for mark in ("\0", "\n", "\r")):
            raise SystemExit(
                f"set BOX_IDENTITY_JSON or all four concrete QUACTLIZE_BOX_* fields; {variable} is missing")
        identity[field] = {"value": value, "source": "operator"}
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
                        identity["pci_identity"]["value"]):
        raise SystemExit("QUACTLIZE_BOX_PCI_IDENTITY must be a full PCI BDF")
    document = {
        "schema": "quactlize.prebuilt-box-execution-identity.v1",
        "source": "operator-environment-weaker",
        "evidence_grade": "operator",
        "identity": identity,
    }
data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".",
                                 suffix=".current", delete=False) as stream:
    temporary = pathlib.Path(stream.name)
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY
  atomic_text "$identity.sha256" "$(sha256sum "$identity" | awk '{print $1}')"
else
  [ -f "$identity" ] && [ ! -L "$identity" ] && [ -s "$identity.sha256" ] || \
    fail 'resume box identity is missing'
  [ "$(sha256sum "$identity" | awk '{print $1}')" = "$(cat "$identity.sha256")" ] || \
    fail 'resume box identity differs'
fi

mapfile -t dense_args < <(python3 -B - "$plan" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1]))
for row in plan["dense"]:
    print(f"--dense={row['m']},{row['n']},{row['k']}")
PY
)
mapfile -t grouped_args < <(python3 -B - "$plan" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1]))
for row in plan["grouped"]:
    print(f"--grouped={row['tokens']},{row['n']},{row['k']},{row['experts']},{row['topk']}")
PY
)
read -r dense_count grouped_count < <(python3 -B - "$plan" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1]))
print(len(plan["dense"]), len(plan["grouped"]))
PY
)
[ "${#dense_args[@]}" -eq "$dense_count" ] && \
  [ "${#grouped_args[@]}" -eq "$grouped_count" ] || fail 'plan-to-argument denominator differs'

for qtype in 10 11 12 13 14; do
  binary="$bundle/q$qtype/test_fq_kquant_layout_perf"
  library="$bundle/q$qtype/libquactlize_ppu.so"
  q_library_path="$(dirname "$library")${runtime_tail:+:$runtime_tail}"
  for round in $(seq 1 "$rounds"); do
    if ((round % 2)); then order=xplane-first; else order=kpack-first; fi
    log="$out/runs/q$qtype-round$round.log"
    run_args=("--iterations=$iterations" "--warmups=$warmups" "--round=$round"
              "--order=$order" "--all-configs=1")
    if [ "$qtype" = 12 ]; then
      run_args+=("${grouped_args[@]}")
    else
      run_args+=("${dense_args[@]}" "${grouped_args[@]}")
    fi
    printf '[prebuilt-fq-kquant] run q=%s round=%s order=%s\n' "$qtype" "$round" "$order"
    run_committed "$log" env LD_LIBRARY_PATH="$q_library_path" \
      "$binary" "${run_args[@]}"
    [ "$(grep -c '^FQ_KQUANT_LAYOUT_RUN ' "$log")" -eq 1 ] || \
      fail "q$qtype round$round completion marker denominator differs"
  done
done

python3 -B "$analyzer" analyze --plan "$plan" --runs "$out/runs" \
  --output "$out/results" --rounds "$rounds" --iterations "$iterations" \
  --threshold-pct "$threshold" --all-configs 1 | tee "$out/results/analyze.log"
python3 -B "$fitter" fit --summary "$out/results/summary.json" \
  --output "$out/results/config-heuristic.json" \
  --regret-threshold-pct "$threshold" --max-leaves "$max_leaves" \
  --min-leaf-rows "$min_leaf_rows" --min-leaf-families "$min_leaf_families" \
  | tee "$out/results/config-heuristic.log"

python3 -B - "$out" "$manifest" "$profile" "$iterations" "$warmups" "$rounds" \
  "$threshold" "$max_leaves" "$min_leaf_rows" "$min_leaf_families" <<'PY'
import hashlib
import json
import os
import pathlib
import tempfile
import sys

out = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
authority = out / "results" / "result-authority.json"
preflight_path = out / "inputs" / "runtime-preflight.json"

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

files = []
for directory in (out / "inputs", out / "runs", out / "results"):
    for path in sorted(directory.rglob("*")):
        if path == authority or path.name == "result-authority.sha256":
            continue
        if path.is_symlink():
            raise SystemExit(f"result path is symlinked: {path}")
        if path.is_file():
            files.append({
                "path": path.relative_to(out).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest(path),
            })
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
if preflight.get("schema") != "quactlize.fq-kquant-prebuilt-runtime-preflight.v2":
    raise SystemExit("runtime preflight schema differs")
sdk_identity = preflight.get("sdk_identity")
if not isinstance(sdk_identity, dict):
    raise SystemExit("runtime preflight SDK identity is missing")
grade = preflight.get("evidence_grade")
if grade not in {"verified-sdk", "unverified-sdk"} or \
        sdk_identity.get("evidence_grade") != grade:
    raise SystemExit("runtime preflight evidence grade differs")
status = sdk_identity.get("identity_status")
mismatch_count = len(sdk_identity.get("mismatches", []))
if ((grade, status, mismatch_count == 0) not in {
        ("verified-sdk", "VERIFIED", True),
        ("unverified-sdk", "MISMATCH_ALLOWED", False)}):
    raise SystemExit("runtime preflight SDK grade/status/mismatch contract differs")
document = {
    "schema": "quactlize.fq-kquant-prebuilt-result-authority.v2",
    "source_commit": "2b513637fc3d315077b14ab81784ff1fb21e1bb7",
    "bundle_manifest_sha256": digest(manifest),
    "evidence_grade": grade,
    "runtime_preflight": {
        "path": "inputs/runtime-preflight.json",
        "sha256": digest(preflight_path),
        "sdk_identity_status": status,
        "sdk_mismatch_count": mismatch_count,
    },
    "controls": {
        "profile": sys.argv[3],
        "all_configs": 1,
        "iterations": int(sys.argv[4]),
        "warmups": int(sys.argv[5]),
        "rounds": int(sys.argv[6]),
        "threshold_pct": float(sys.argv[7]),
        "heuristic_max_leaves": int(sys.argv[8]),
        "heuristic_min_leaf_rows": int(sys.argv[9]),
        "heuristic_min_leaf_families": int(sys.argv[10]),
    },
    "files": files,
}
data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
with tempfile.NamedTemporaryFile(dir=authority.parent, prefix=authority.name + ".",
                                 suffix=".current", delete=False) as stream:
    temporary = pathlib.Path(stream.name)
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, authority)
PY
atomic_text "$out/results/result-authority.sha256" \
  "$(sha256sum "$out/results/result-authority.json" | awk '{print $1}')  result-authority.json"
[ -z "$(find "$out" -type l -print -quit)" ] || fail 'result contains a symbolic link'

printf '[prebuilt-fq-kquant] COMPLETE source=2b513637 profile=%s output=%s\n' "$profile" "$out"
printf '[prebuilt-fq-kquant] summary=%s heuristic=%s\n' \
  "$out/results/summary.tsv" "$out/results/config-heuristic.json"
