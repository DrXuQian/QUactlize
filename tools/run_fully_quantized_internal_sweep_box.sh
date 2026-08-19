#!/usr/bin/env bash
# Exhaustive FullyQuantized component runner for the internal full sweep.
#
# One build shard owns exactly (qtype, ArtifactTileK, BChunk).  It measures
# placed BC GEMV and tensor-core S1 as full-output algorithms; tensor-core
# S2/S4/S8 are producer-only and run the real reducer outside timing to close
# raw-fp16 correctness.  This script never reports those producer numbers as
# Split-K end-to-end results.
set -uo pipefail

atomic_text() {
  local destination="$1" value="$2" current
  current="${destination}.current.$$"
  printf '%s\n' "$value" > "$current" || return 2
  mv -f -- "$current" "$destination" || return 2
}

main() {
  local root workspace_root sha short stamp out spec gguf_set plan
  local jobs iterations repeats per_unit identity identity_current
  local source_hashes binary_hashes run_contract run_contract_current
  local spec_sha plan_source_sha peak_tflops hbm_gbs
  local resume_evidence
  local attempt_id
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-internal-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-internal] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  spec="${INTERNAL_SWEEP_SPEC:-}"
  gguf_set="${GGUF_SET:-}"
  if [ -z "$spec" ] || [ ! -f "$spec" ]; then
    printf '[fq-internal] FAIL: INTERNAL_SWEEP_SPEC must name a %s file\n' \
      'quactlize.gguf_internal_shape_inventory.v2' >&2
    return 2
  fi
  if [ -z "$gguf_set" ] || [ ! -f "$gguf_set" ]; then
    printf '[fq-internal] FAIL: GGUF_SET must name a resolved-models v1 file\n' >&2
    return 2
  fi
  spec="$(realpath -e -- "$spec")" || return 2
  gguf_set="$(realpath -e -- "$gguf_set")" || return 2
  spec_sha="$(sha256sum "$spec" | awk '{print $1}')" || return 2
  jobs="${JOBS:-16}"
  iterations="${ITERATIONS:-${BENCH_REPS:-7}}"
  repeats="${CORRECTNESS_REPEATS:-2}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-4}"
  peak_tflops="${PPU_PEAK_TFLOPS:-500}"
  hbm_gbs="${PPU_HBM_GBS:-2766}"
  attempt_id="${INTERNAL_SWEEP_ATTEMPT_ID:-fq-${stamp}-$$}"
  case "$attempt_id" in
    ""|*[!A-Za-z0-9._:-]*)
      printf '[fq-internal] FAIL: INTERNAL_SWEEP_ATTEMPT_ID must be a nonempty stable token\n' >&2
      return 2 ;;
  esac
  case "$jobs:$iterations:$repeats:$per_unit" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0)
      printf '[fq-internal] FAIL: JOBS/BENCH_REPS/CORRECTNESS_REPEATS/FQ_CONFIGS_PER_UNIT must be positive integers\n' >&2
      return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[fq-internal] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the denominator\n' >&2
    return 2
  fi
  if [ -e "$out" ]; then
    if [ "${RESUME:-0}" != 1 ]; then
      printf '[fq-internal] FAIL: refusing to overwrite %s; set RESUME=1 to continue\n' "$out" >&2
      return 2
    fi
  else
    mkdir "$out" || return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/raw" "$out/results" || return 2
  resume_evidence="$(python3 -B - "$out/raw" "$out/build" <<'PY'
import pathlib, sys
raw, build = map(pathlib.Path, sys.argv[1:])
has_log = any(path.is_file() and path.stat().st_size > 0
              for path in raw.glob("*/run.log"))
has_run_sidecar = any(path.is_file()
                      for pattern in ("*/run.rc", "*/run.commit.json")
                      for path in raw.glob(pattern))
has_binary = any(path.is_file() and path.name == "test_fully_quantized_internal_sweep"
                 for path in build.glob("*/ppu_targets/test_fully_quantized_internal_sweep"))
print(int(has_log or has_run_sidecar or has_binary))
PY
)" || return 2

  python3 -B "$root/tools/analyze_fully_quantized_internal_sweep.py" --self-test || return 2
  python3 -B "$root/tools/fully_quantized_internal_matrix.py" self-test || return 2
  python3 -B "$root/tools/merge_internal_full_sweep.py" self-test || return 2
  python3 -B "$root/ci/check_fq_internal_runner_contract.py" || return 2

  plan="$out/plan.json"
  if [ "$resume_evidence" = 1 ] && [ ! -s "$plan" ]; then
    printf '[fq-internal] FAIL: binary/run.log exists but plan.json is missing\n' >&2
    return 2
  fi
  if [ ! -s "$plan" ]; then
    local plan_current="${plan}.current.$$"
    python3 -B "$root/tools/analyze_fully_quantized_internal_sweep.py" \
      --materialize-plan "$spec" --materialized-output "$plan_current" || return 2
    mv -f -- "$plan_current" "$plan" || return 2
  fi
  plan_source_sha="$(python3 -B - "$plan" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["provenance"]["shape_manifest_sha256"])
PY
)" || return 2
  if [ "$plan_source_sha" != "$spec_sha" ]; then
    printf '[fq-internal] FAIL: current inventory sha=%s differs from materialized plan=%s\n' \
      "$spec_sha" "$plan_source_sha" >&2
    return 2
  fi
  python3 -B "$root/tools/analyze_fully_quantized_internal_sweep.py" \
    --validate-plan "$plan" --gguf-set "$gguf_set" || return 2
  local plan_sha source_state
  plan_sha="$(sha256sum "$plan" | awk '{print $1}')" || return 2
  if [ -s "$out/plan.sha256" ] && [ "$(cat "$out/plan.sha256")" != "$plan_sha" ]; then
    printf '[fq-internal] FAIL: materialized plan changed inside resumed bundle\n' >&2
    return 2
  fi
  if [ "$resume_evidence" = 1 ] && [ ! -s "$out/plan.sha256" ]; then
    printf '[fq-internal] FAIL: binary/run.log exists but plan.sha256 is missing\n' >&2
    return 2
  fi
  atomic_text "$out/plan.sha256" "$plan_sha" || return 2

  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum \
      "$root/benchmarks/fully_quantized_splitk_producer_bench.hpp" \
      "$root/benchmarks/fully_quantized_splitk_producer_unit.inc" \
      "$root/benchmarks/test_fully_quantized_internal_sweep.cu" \
      "$root/quactlize/csrc/fq_internal_sweep.cmake.in" \
      "$root/quactlize/csrc/device/ppu_dense_layout.cu" \
      "$root/quactlize/include/dense_splitk_multiformat_ppu.cuh" \
      "$root/quactlize/include/dense_splitk_parallel_ppu.cuh" \
      "$root/quactlize/include/gguf_bc_vecdot.hpp" \
      "$root/quactlize/include/gguf_packed_unit.hpp" \
      "$root/quactlize/include/ppu_dense_shipping_policy.hpp" \
      "$root/quactlize/include/ppu_format_config.inc" \
      "$root/quactlize/include/ppu_group_schedule.hpp" \
      "$root/quactlize/include/ppu_tactic_space.hpp" \
      "$root/tests/helper.h" \
      "$root/tools/analyze_fully_quantized_internal_sweep.py" \
      "$root/tools/emit_fully_quantized_splitk_superset.cpp" \
      "$root/tools/fully_quantized_internal_matrix.py" \
      "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      "$root/tools/probe_box_identity.py" \
      "$root/tools/box_identity_schema.py" \
      "$root/tools/box_identity_probe.cpp" \
      "$root/tools/run_fully_quantized_internal_sweep_box.sh" \
      "$root/ci/check_fq_internal_runner_contract.py" \
      "$root/quactlize/csrc/CMakeLists.txt.in" "$root/build.sh"
  } | sha256sum | awk '{print $1}')" || return 2
  if [ -s "$out/source-state.sha256" ] && \
     [ "$(cat "$out/source-state.sha256")" != "$source_state" ]; then
    printf '[fq-internal] FAIL: source authority changed inside resumed bundle\n' >&2
    return 2
  fi
  if [ "$resume_evidence" = 1 ] && [ ! -s "$out/source-state.sha256" ]; then
    printf '[fq-internal] FAIL: binary/run.log exists but source-state authority is missing\n' >&2
    return 2
  fi
  if [ ! -s "$out/source-state.sha256" ]; then
    atomic_text "$out/source-state.sha256" "$source_state" || return 2
    local source_patch_current="$out/source.patch.current.$$"
    git -C "$root" diff --binary --no-ext-diff HEAD > "$source_patch_current" || return 2
    mv -f -- "$source_patch_current" "$out/source.patch" || return 2
  fi

  identity="$out/identity.json"
  identity_current="$out/identity.current.json"
  # Re-measure identity on every invocation.  A resumed bundle belongs to one
  # physical/software device identity; a stale identity.json is not evidence
  # that the current process still sees that device.
  mkdir -p "$out/identity-probe" || return 2
  TMPDIR="$out/identity-probe" python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$identity_current" || return 2
  python3 -B - "$identity" "$identity_current" "$resume_evidence" <<'PY' || return 2
import json, os, pathlib, sys
saved, current = map(pathlib.Path, sys.argv[1:3])
completed = sys.argv[3] == "1"
now = json.loads(current.read_text())
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def atomic_write(path, text):
    temporary = path.with_name(path.name + f".current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
if saved.exists():
    before = json.loads(saved.read_text())
    if canonical(before) != canonical(now):
        raise SystemExit("device identity changed inside resumed bundle")
else:
    if completed:
        raise SystemExit("binary/run.log exists but saved device identity is missing")
    atomic_write(saved, current.read_text())
PY
  source_hashes="$out/source-hashes.json"
  python3 - "$root" "$source_hashes" "$resume_evidence" <<'PY' || return 2
import hashlib, json, os, pathlib, subprocess, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
completed = sys.argv[3] == "1"
paths = [
 "benchmarks/fully_quantized_splitk_producer_bench.hpp",
 "benchmarks/fully_quantized_splitk_producer_unit.inc",
 "benchmarks/test_fully_quantized_internal_sweep.cu",
 "quactlize/csrc/fq_internal_sweep.cmake.in",
 "quactlize/csrc/CMakeLists.txt.in",
 "quactlize/csrc/device/ppu_dense_layout.cu",
 "quactlize/include/dense_splitk_multiformat_ppu.cuh",
 "quactlize/include/dense_splitk_parallel_ppu.cuh",
 "quactlize/include/gguf_bc_vecdot.hpp",
 "quactlize/include/gguf_packed_unit.hpp",
 "quactlize/include/ppu_format_config.inc",
 "quactlize/include/ppu_dense_shipping_policy.hpp",
 "quactlize/include/ppu_group_schedule.hpp",
 "quactlize/include/ppu_tactic_space.hpp",
 "tests/helper.h",
 "tools/analyze_fully_quantized_internal_sweep.py",
 "tools/emit_fully_quantized_splitk_superset.cpp",
 "tools/fully_quantized_internal_matrix.py",
 "tools/gen_fully_quantized_splitk_producer_units.py",
 "tools/probe_box_identity.py", "tools/box_identity_schema.py",
 "tools/box_identity_probe.cpp",
 "tools/run_fully_quantized_internal_sweep_box.sh", "build.sh",
 "ci/check_fq_internal_runner_contract.py",
]
def git_sha(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
def tree_sha(relative):
    directory = root / relative
    members = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*")) if path.is_file()
    }
    payload = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
def atomic_write(path, text):
    temporary = path.with_name(path.name + f".current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
fixed_hashes = {
    name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths
}
# Conservative dependency supersets close the transitive-header seam without
# pretending a hand-maintained include list is a compiler depfile.
fixed_hashes.update({
    "tree/quactlize/include": tree_sha("quactlize/include"),
    "tree/third_party/actlize/include": tree_sha("third_party/actlize/include"),
    "tree/third_party/actlize/tools/util/include": tree_sha(
        "third_party/actlize/tools/util/include"),
})
current = {
 "root_sha": git_sha(root), "actlize_sha": git_sha(root / "third_party/actlize"),
 "source_hashes": fixed_hashes,
}
if output.exists():
    previous = json.loads(output.read_text())
    for key in ("root_sha", "actlize_sha", "source_hashes"):
        if previous.get(key) != current[key]:
            raise SystemExit(f"source-hashes authority changed on resume: {key}")
    generated = previous.get("generated_shards")
    if not isinstance(generated, dict):
        raise SystemExit("source-hashes resume lacks generated_shards authority")
else:
    if completed:
        raise SystemExit("binary/run.log exists but source-hashes authority is missing")
    generated = {}
current["generated_shards"] = generated
atomic_write(output, json.dumps(current, indent=2, sort_keys=True) + "\n")
PY
  binary_hashes="$out/binary-hashes.json"
  if [ "$resume_evidence" = 1 ] && [ ! -s "$binary_hashes" ]; then
    printf '[fq-internal] FAIL: binary/run.log exists but binary-hashes authority is missing\n' >&2
    return 2
  fi
  if [ ! -s "$binary_hashes" ]; then atomic_text "$binary_hashes" '{}' || return 2; fi

  # Bind every semantic timing knob before any completed run.log may be
  # reused.  Build parallelism is intentionally absent: it cannot change a
  # measured cell, while the four fields below can.
  run_contract="$out/run-contract.json"
  run_contract_current="$out/run-contract.current.json"
  python3 -B - "$run_contract" "$run_contract_current" \
    "$iterations" "$repeats" "$per_unit" "$peak_tflops" "$hbm_gbs" \
    "$plan_sha" "$source_state" "$identity" "$resume_evidence" <<'PY' || return 2
import hashlib, json, math, os, pathlib, sys
saved, current = map(pathlib.Path, sys.argv[1:3])
iterations, repeats, per_unit = map(int, sys.argv[3:6])
peak_tflops, hbm_gbs = map(float, sys.argv[6:8])
if not all(math.isfinite(value) and value > 0 for value in (peak_tflops, hbm_gbs)):
    raise SystemExit("PPU_PEAK_TFLOPS/PPU_HBM_GBS must be finite and positive")
identity_path = pathlib.Path(sys.argv[10])
resume_evidence = sys.argv[11] == "1"
doc = {
    "schema": "quactlize.fully_quantized_internal_sweep.run_contract.v1",
    "iterations": iterations,
    "correctness_repeats": repeats,
    "configs_per_unit": per_unit,
    "peak_tflops": peak_tflops,
    "hbm_gbs": hbm_gbs,
    "plan_sha256": sys.argv[8],
    "source_state_sha256": sys.argv[9],
    "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
    "identity_probe_tmpdir": "identity-probe",
}
encoded = json.dumps(doc, indent=2, sort_keys=True) + "\n"
def atomic_write(path, text):
    temporary = path.with_name(path.name + f".current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
atomic_write(current, encoded)
if saved.exists() and json.loads(saved.read_text()) != doc:
    raise SystemExit("run contract changed inside resumed bundle")
if not saved.exists():
    if resume_evidence:
        raise SystemExit("binary/run.log exists but run contract is missing")
    atomic_write(saved, encoded)
PY

  local q artifact bc packed shard generated manifest typed bc_supported binary build_log
  local run_log run_rc run_commit existing_rc rc
  local shard_evidence
  local -a qtypes=(10 11 12 13 14) artifacts=(32 64 128 256) bchunks=(0 1)
  for q in "${qtypes[@]}"; do
    case "$q" in 10) packed=2;; 11) packed=3;; 12) packed=0;; 13) packed=1;; 14) packed=4;; esac
    local -a shape_args=()
    while IFS=$'\t' read -r _ shape_id plan_q m n k model route support; do
      if [ "$plan_q" = "$q" ] && [ "$route" = dense ] && [ "$support" = SUPPORTED ]; then
        shape_args+=("--shape=${m}x${n}x${k}")
      fi
    done < <(python3 -B "$root/tools/analyze_fully_quantized_internal_sweep.py" --list-plan "$plan")
    if [ "${#shape_args[@]}" -eq 0 ]; then continue; fi
    # Runtime work is deduplicated by numeric shape inside one qtype.  Model,
    # TP and tensor identities remain in the analyzer's output cells.
    mapfile -t shape_args < <(printf '%s\n' "${shape_args[@]}" | sort -u)
    for artifact in "${artifacts[@]}"; do
      for bc in "${bchunks[@]}"; do
        shard="q${q}-a${artifact}-bc${bc}"
        generated="$out/generated/$shard"
        manifest="$generated/manifest.json"
        shard_evidence=0
        if [ -s "$out/raw/$shard/run.log" ] || \
           [ -e "$out/raw/$shard/run.rc" ] || \
           [ -e "$out/raw/$shard/run.commit.json" ] || \
           [ -f "$out/build/$shard/ppu_targets/test_fully_quantized_internal_sweep" ]; then
          shard_evidence=1
        fi
        if [ "$shard_evidence" = 1 ] && [ ! -s "$manifest" ]; then
          printf '[fq-internal] FAIL: binary/run.log shard=%s lost generated manifest\n' "$shard" >&2
          return 2
        fi
        if [ ! -s "$manifest" ]; then
          mkdir -p "$generated" || return 2
          python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
            --qtype "$q" --artifact-tk "$artifact" --bchunk "$bc" \
            --per-unit "$per_unit" --out-dir "$generated" || return 2
        fi
        python3 -B - "$source_hashes" "$shard" "$generated" \
          "$shard_evidence" <<'PY' || return 2
import hashlib, json, os, pathlib, sys
authority, shard, generated = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
has_evidence = sys.argv[4] == "1"
doc = json.loads(authority.read_text())
files = [path for path in generated.rglob("*") if path.is_file()]
members = {
    str(path.relative_to(generated)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(files)
}
payload = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
digest = hashlib.sha256(payload).hexdigest()
def atomic_write(path, text):
    temporary = path.with_name(path.name + f".current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
old = doc.setdefault("generated_shards", {}).get(shard)
if old is None and has_evidence:
    raise SystemExit(f"binary/run.log shard lost generated authority for {shard}")
if old is not None and old != digest:
    raise SystemExit(f"generated shard authority changed for {shard}")
doc["generated_shards"][shard] = digest
atomic_write(authority, json.dumps(doc, indent=2, sort_keys=True) + "\n")
PY
        typed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["denominator"]["typed_rows"])' "$manifest")" || return 2
        # A generated shard is still source/denominator authority when every
        # row is a named static rejection, but it has no device type to
        # instantiate and therefore no runtime graph.  Building such a shard
        # creates an empty generated target and, worse, lets unrelated build
        # failures masquerade as a rejected tactic.  The manifest's typed
        # denominator is the sole capability boundary for every format/bchunk
        # combination; do not special-case the reason it reached zero.
        if [ "$typed" -eq 0 ]; then
          printf '[fq-internal] static-only shard=%s; no binary required typed=0\n' \
            "$shard"
          continue
        fi
        binary="$out/build/$shard/ppu_targets/test_fully_quantized_internal_sweep"
        build_log="$out/build/$shard.log"
        if [ "$shard_evidence" = 1 ] && \
           { [ ! -f "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; }; then
          printf '[fq-internal] FAIL: committed shard=%s lost its exact binary\n' \
            "$shard" >&2
          return 2
        fi
        if [ "$shard_evidence" = 0 ] && { [ -e "$binary" ] || [ -L "$binary" ]; }; then
          printf '[fq-internal] FAIL: fresh shard=%s has an unexpected binary path\n' \
            "$shard" >&2
          return 2
        fi
        if [ ! -f "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
          printf '[fq-internal] build shard=%s typed=%s\n' "$shard" "$typed"
          (cd "$root" && PPU_BUILD_DIR="$out/build/$shard" PPU_ARCHS=ppu0010 \
            JOBS="$jobs" TARGET=test_fully_quantized_internal_sweep \
            FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE="$q" \
            FQ_SWEEP_ARTIFACT_TK="$artifact" FQ_SWEEP_BCHUNK="$bc" \
            FQ_SWEEP_PACKED_FORMAT="$packed" ./build.sh) > "$build_log" 2>&1
          rc=$?
          if [ "$rc" -ne 0 ]; then
            printf '[fq-internal] FAIL: build shard=%s rc=%d\n' "$shard" "$rc" >&2
            tail -100 "$build_log" >&2
            return "$rc"
          fi
          binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
        fi
        if [ -z "$binary" ] || [ ! -f "$binary" ] || [ ! -x "$binary" ] || \
           [ -L "$binary" ]; then
          printf '[fq-internal] FAIL: binary missing for %s\n' "$shard" >&2
          return 2
        fi
        python3 - "$binary_hashes" "$shard" "$binary" \
          "$shard_evidence" <<'PY' || return 2
import hashlib, json, os, pathlib, sys
path, name, binary = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
has_evidence = sys.argv[4] == "1"
doc = json.loads(path.read_text())
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
def atomic_write(path, text):
    temporary = path.with_name(path.name + f".current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
if name not in doc and has_evidence:
    raise SystemExit(f"binary/run.log shard lost binary authority for {name}")
if name in doc and doc[name] != digest: raise SystemExit(f"binary hash changed for {name}")
doc[name] = digest
atomic_write(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")
PY
        run_log="$out/raw/$shard/run.log"
        run_rc="$out/raw/$shard/run.rc"
        run_commit="$out/raw/$shard/run.commit.json"
        mkdir -p "$(dirname "$run_log")" || return 2
        existing_rc="$(python3 -B - "$run_log" "$run_rc" "$run_commit" \
          "$run_contract" "$source_hashes" "$binary_hashes" "$shard" <<'PY'
import hashlib, json, pathlib, sys
log, rc_path, commit_path, contract_path, source_path, binary_path = map(
    pathlib.Path, sys.argv[1:7])
shard = sys.argv[7]
present = [path.exists() for path in (log, rc_path, commit_path)]
if not any(present):
    print("NONE")
    raise SystemExit(0)
if not all(present) or not log.is_file() or log.stat().st_size == 0:
    raise SystemExit(
        f"binary/run.log shard={shard} has an incomplete run evidence triplet")
try:
    rc_text = rc_path.read_text().strip()
    if not rc_text.isdigit() or not 0 <= int(rc_text) <= 255:
        raise ValueError
    doc = json.loads(commit_path.read_text())
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(f"binary/run.log shard={shard} has malformed run evidence")
sources = json.loads(source_path.read_text())
binaries = json.loads(binary_path.read_text())
expected = {
    "schema": "quactlize.fully_quantized_internal_sweep.run_commit.v1",
    "rc": int(rc_text),
    "run_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "run_rc_sha256": hashlib.sha256(rc_path.read_bytes()).hexdigest(),
    "run_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "generated_source_sha256": sources.get("generated_shards", {}).get(shard),
    "binary_sha256": binaries.get(shard),
}
if doc != expected or any(value is None for value in expected.values()):
    raise SystemExit(f"binary/run.log shard={shard} run evidence authority changed")
print(rc_text)
PY
)" || return 2
        if [ "$existing_rc" = 0 ]; then
          printf '[fq-internal] resume shard=%s\n' "$shard"
          continue
        fi
        printf '[fq-internal] run shard=%s shapes=%d\n' "$shard" "${#shape_args[@]}"
        local run_log_current="${run_log}.current.$$"
        local run_rc_current="${run_rc}.current.$$"
        local run_commit_current="${run_commit}.current.$$"
        "$binary" "${shape_args[@]}" --iterations="$iterations" \
          --correctness-repeats="$repeats" > "$run_log_current" 2>&1
        rc=$?
        printf '%d\n' "$rc" > "$run_rc_current" || return 2
        python3 -B - "$run_log_current" "$run_rc_current" \
          "$run_commit_current" "$run_contract" "$source_hashes" \
          "$binary_hashes" "$shard" <<'PY' || return 2
import hashlib, json, os, pathlib, sys
log, rc_path, output, contract_path, source_path, binary_path = map(
    pathlib.Path, sys.argv[1:7])
shard = sys.argv[7]
sources = json.loads(source_path.read_text())
binaries = json.loads(binary_path.read_text())
doc = {
    "schema": "quactlize.fully_quantized_internal_sweep.run_commit.v1",
    "rc": int(rc_path.read_text().strip()),
    "run_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "run_rc_sha256": hashlib.sha256(rc_path.read_bytes()).hexdigest(),
    "run_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "generated_source_sha256": sources["generated_shards"][shard],
    "binary_sha256": binaries[shard],
}
with output.open("w", encoding="utf-8") as stream:
    json.dump(doc, stream, indent=2, sort_keys=True)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
        # The commit moves last.  An interruption can leave a mismatched
        # triplet, but the next invocation rejects it instead of reusing it.
        mv -f -- "$run_log_current" "$run_log" || return 2
        mv -f -- "$run_rc_current" "$run_rc" || return 2
        mv -f -- "$run_commit_current" "$run_commit" || return 2
        if [ "$rc" -ne 0 ]; then
          printf '[fq-internal] shard=%s runtime rc=%d; continuing for complete failure census\n' "$shard" "$rc" >&2
          tail -40 "$run_log" >&2
        fi
      done
    done
  done

  if [ "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$binary_hashes")" -eq 0 ]; then
    printf '[fq-internal] FAIL: no device binary was built; plan-only cannot be COMPLETE\n' >&2
    return 3
  fi
  python3 -B "$root/tools/analyze_fully_quantized_internal_sweep.py" \
    --plan "$plan" --raw-root "$out/raw" --output "$out/results/summary.json" \
    --identity "$identity" --source-hashes "$source_hashes" \
    --binary-hashes "$binary_hashes" --run-contract "$run_contract" \
    --attempt-id "$attempt_id" \
    --peak-tflops "$peak_tflops" --hbm-gbs "$hbm_gbs"
  rc=$?
  local provenance_current="$out/provenance.txt.current.$$"
  {
    printf 'schema=quactlize.fully_quantized_internal_sweep.run.v2\n'
    printf 'root_sha=%s\nactlize_sha=%s\n' "$sha" "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'source_state_sha256=%s\nplan_sha256=%s\n' "$source_state" "$plan_sha"
    printf 'run_contract_sha256=%s\n' "$(sha256sum "$run_contract" | awk '{print $1}')"
    printf 'orchestration_attempt_id=%s\n' "$attempt_id"
    printf 'inventory_v2=%s\ngguf_set=%s\n' "$spec" "$gguf_set"
    printf 'iterations=%s correctness_repeats=%s configs_per_unit=%s\n' "$iterations" "$repeats" "$per_unit"
    printf 'splitk_scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS\n'
    if [ -s "$out/results/summary.json" ]; then
      printf 'summary_sha256=%s\n' "$(sha256sum "$out/results/summary.json" | awk '{print $1}')"
    fi
  } > "$provenance_current" || return 2
  mv -f -- "$provenance_current" "$out/provenance.txt" || return 2
  printf '[fq-internal] artifacts=%s\n' "$out"
  return "$rc"
}

main "$@"
