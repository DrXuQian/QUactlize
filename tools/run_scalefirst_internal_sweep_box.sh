#!/usr/bin/env bash
# Exhaustive all-format ScaleFirst component runner.
#
# The generated graph owns q8/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, every supported
# ArtifactTileK, and the unpruned tactic axes.  Runtime boards are ordinary
# full output, every exact persistent capacity/balanced grid, and fixed
# Split-K S2/S4/S8 producer-only with an untimed real reducer correctness gate.
set -uo pipefail

main() {
  if [ "$#" -ne 0 ]; then
    printf '[scalefirst-internal] FAIL: no positional arguments are accepted\n' >&2
    return 2
  fi
  local root workspace_root sha short stamp out resume attempt_id
  local requested_spec requested_gguf frozen_spec frozen_gguf plan plan_sha
  local jobs iterations repeats per_unit peak_tflops hbm_gbs schedule_seed
  local source_hashes identity identity_current run_contract binary_hashes raw_hashes
  local source_state resume_evidence rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || {
    printf '[scalefirst-internal] FAIL: /workspace is unavailable\n' >&2
    return 2
  }
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-scalefirst-internal-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[scalefirst-internal] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  resume="${RESUME:-0}"
  case "$resume" in 0|1) ;; *) printf '[scalefirst-internal] FAIL: RESUME must be 0 or 1\n' >&2; return 2;; esac
  if [ -e "$out" ]; then
    if [ "$resume" != 1 ] || [ ! -d "$out" ]; then
      printf '[scalefirst-internal] FAIL: refusing existing OUT=%s; set RESUME=1\n' "$out" >&2
      return 2
    fi
  else
    mkdir -p "$out" || return 2
  fi
  mkdir -p "$out/inputs" "$out/generated" "$out/build" "$out/raw" \
    "$out/results" "$out/identity-probe" || return 2

  # Establish resume state before materializing or refreshing any authority.
  # Otherwise deleting plan.json (or a frozen input) from a measured bundle
  # could silently recreate a plausible-looking but unbound replacement.
  resume_evidence="$(python3 -B - "$out" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
run_sidecar = any(path.is_file()
                  for pattern in ("*/run.log", "*/run.rc", "*/run.commit.json")
                  for path in (root / "raw").glob(pattern))
binary = any(path.name == "test_scalefirst_internal_sweep"
             for path in (root / "build").glob(
                 "*/ppu_targets/test_scalefirst_internal_sweep"))
print(int(run_sidecar or binary))
PY
)" || return 2

  # A top-level orchestration must provide the attempt token.  A standalone
  # component run creates one, and publishes it with exactly the same schema.
  attempt_id="${INTERNAL_SWEEP_ATTEMPT_ID:-}"
  if [ -z "$attempt_id" ] && [ -n "${INTERNAL_SWEEP_COMPONENT:-}" ]; then
    printf '[scalefirst-internal] FAIL: orchestrated run lacks INTERNAL_SWEEP_ATTEMPT_ID\n' >&2
    return 2
  fi
  if [ -z "$attempt_id" ]; then attempt_id="sf-${stamp}-$$"; fi
  case "$attempt_id" in
    *[!A-Za-z0-9._:-]*|'')
      printf '[scalefirst-internal] FAIL: malformed INTERNAL_SWEEP_ATTEMPT_ID\n' >&2
      return 2 ;;
  esac

  jobs="${JOBS:-16}"
  iterations="${ITERATIONS:-${BENCH_REPS:-7}}"
  repeats="${CORRECTNESS_REPEATS:-2}"
  per_unit="${SCALEFIRST_CONFIGS_PER_UNIT:-32}"
  peak_tflops="${PPU_PEAK_TFLOPS:-500}"
  hbm_gbs="${PPU_HBM_GBS:-2766}"
  schedule_seed="${SCALEFIRST_SCHEDULE_SEED:-0x6a09e667f3bcc909}"
  case "$jobs:$iterations:$repeats:$per_unit" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0)
      printf '[scalefirst-internal] FAIL: JOBS/ITERATIONS/CORRECTNESS_REPEATS/SCALEFIRST_CONFIGS_PER_UNIT must be positive integers\n' >&2
      return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[scalefirst-internal] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the denominator\n' >&2
    return 2
  fi

  frozen_spec="$out/inputs/inventory.json"
  frozen_gguf="$out/inputs/resolved-models.json"
  requested_spec="${INTERNAL_SWEEP_SPEC:-}"
  requested_gguf="${GGUF_SET:-}"
  python3 -B - "$requested_spec" "$frozen_spec" "$requested_gguf" \
    "$frozen_gguf" "$resume_evidence" <<'PY' || return 2
import os, pathlib, sys
evidence = sys.argv[5] == "1"
def freeze(requested, frozen, label):
    dst = pathlib.Path(frozen)
    src = pathlib.Path(requested).resolve(strict=True) if requested else None
    if dst.exists():
        if not dst.is_file() or dst.stat().st_size == 0:
            raise SystemExit(f"{label}: frozen authority is empty/non-file")
        if src is not None and src.read_bytes() != dst.read_bytes():
            raise SystemExit(f"{label}: requested authority differs from frozen bundle")
        return
    if evidence:
        raise SystemExit(f"binary/run evidence exists without frozen {label} authority")
    if src is None:
        raise SystemExit(f"{label}: required on a fresh component run")
    data = src.read_bytes()
    if not data: raise SystemExit(f"{label}: requested authority is empty")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.current.{os.getpid()}")
    with tmp.open("wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, dst)
freeze(sys.argv[1], sys.argv[2], "INTERNAL_SWEEP_SPEC")
freeze(sys.argv[3], sys.argv[4], "GGUF_SET")
PY

  python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" --self-test || return 2
  python3 -B "$root/tools/scalefirst_internal_matrix.py" self-test || return 2
  python3 -B "$root/tools/merge_internal_full_sweep.py" self-test || return 2
  python3 -B "$root/ci/check_scalefirst_internal_runner_contract.py" || return 2

  plan="$out/plan.json"
  if [ "$resume_evidence" = 1 ] && [ ! -s "$plan" ]; then
    printf '[scalefirst-internal] FAIL: binary/run evidence exists but plan.json is missing\n' >&2
    return 2
  fi
  if [ ! -s "$plan" ]; then
    python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" \
      --materialize-plan "$frozen_spec" --materialized-output "$plan" || return 2
  fi
  python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" \
    --validate-plan "$plan" --gguf-set "$frozen_gguf" || return 2
  plan_sha="$(sha256sum "$plan" | awk '{print $1}')" || return 2
  python3 -B - "$out/plan.sha256" "$plan_sha" "$resume_evidence" <<'PY' || return 2
import os, pathlib, sys
p, value = pathlib.Path(sys.argv[1]), sys.argv[2]
evidence = sys.argv[3] == "1"
if p.exists() and p.read_text().strip() != value: raise SystemExit("plan changed inside bundle")
if not p.exists() and evidence: raise SystemExit("binary/run evidence exists but plan.sha256 is missing")
t = p.with_name(f".{p.name}.current.{os.getpid()}")
with t.open("w") as f: f.write(value+"\n"); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
  source_hashes="$out/source-hashes.json"
  python3 -B - "$root" "$source_hashes" "$resume_evidence" <<'PY' || return 2
import hashlib,json,os,pathlib,subprocess,sys
root,out=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]); evidence=sys.argv[3]=="1"
paths=[
"benchmarks/scalefirst_internal_sweep_bench.hpp",
"benchmarks/scalefirst_internal_sweep_unit.inc",
"benchmarks/test_scalefirst_internal_sweep.cu",
"quactlize/csrc/scalefirst_internal_sweep.cmake.in",
"quactlize/csrc/CMakeLists.txt.in","quactlize/csrc/device/ppu_dense_layout.cu",
"quactlize/include/dense_splitk_multiformat_ppu.cuh",
"quactlize/include/dense_splitk_parallel_ppu.cuh",
"quactlize/include/ppu_format_config.inc","quactlize/include/ppu_group_schedule.hpp",
"quactlize/include/ppu_tactic_space.hpp","quactlize/include/scalefirst_persistent_policy.hpp",
"tests/helper.h","tools/analyze_fully_quantized_internal_sweep.py",
"tools/analyze_scalefirst_internal_sweep.py","tools/emit_scalefirst_internal_superset.cpp",
"tools/gen_scalefirst_internal_units.py","tools/probe_box_identity.py",
"tools/box_identity_schema.py","tools/box_identity_probe.cpp",
"tools/run_scalefirst_internal_sweep_box.sh","tools/scalefirst_internal_matrix.py",
"ci/check_scalefirst_internal_runner_contract.py","build.sh"]
def gitsha(p): return subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
def tree(rel):
 d=root/rel
 members={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(d.rglob("*")) if p.is_file()}
 return hashlib.sha256(json.dumps(members,sort_keys=True,separators=(",",":")).encode()).hexdigest()
fixed={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
fixed.update({"tree/quactlize/include":tree("quactlize/include"),
 "tree/third_party/actlize/include":tree("third_party/actlize/include"),
 "tree/third_party/actlize/tools/util/include":tree("third_party/actlize/tools/util/include")})
current={"root_sha":gitsha(root),"actlize_sha":gitsha(root/"third_party/actlize"),
         "source_hashes":fixed,"generated_shards":{}}
if out.exists():
 old=json.loads(out.read_text())
 for key in ("root_sha","actlize_sha","source_hashes"):
  if old.get(key)!=current[key]: raise SystemExit(f"source authority changed: {key}")
 current["generated_shards"]=old.get("generated_shards",{})
 if not isinstance(current["generated_shards"],dict): raise SystemExit("generated authority malformed")
elif evidence: raise SystemExit("binary/run evidence exists without source authority")
tmp=out.with_name(f".{out.name}.current.{os.getpid()}")
with tmp.open("w") as f: json.dump(current,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY

  identity="$out/identity.json"
  identity_current="$out/identity.current.json"
  TMPDIR="$out/identity-probe" python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$identity_current" || return 2
  python3 -B - "$identity" "$identity_current" "$resume_evidence" <<'PY' || return 2
import json,os,pathlib,sys
saved,current=map(pathlib.Path,sys.argv[1:3]); evidence=sys.argv[3]=="1"
now=json.loads(current.read_text()); canon=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)
if saved.exists():
 if canon(json.loads(saved.read_text()))!=canon(now): raise SystemExit("device identity changed on resume")
elif evidence: raise SystemExit("binary/run evidence exists without saved device identity")
else:
 tmp=saved.with_name(f".{saved.name}.current.{os.getpid()}")
 with tmp.open("w") as f: json.dump(now,f,indent=2,sort_keys=True,ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,saved)
PY

  # Generate the complete static authority before freezing the run contract.
  # Otherwise the contract would bind an empty generated_shards map and the
  # source authority would change underneath it during the first build.
  local -a qtypes=() artifacts=(32 64 128 256) bchunks=(0 1)
  local q artifact bc shard generated manifest shard_evidence
  mapfile -t qtypes < <(python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" \
    --list-plan "$plan" | awk -F '\t' '$7=="dense" && $8=="SUPPORTED" && ($2==8 || ($2>=10 && $2<=14)) {print $2}' | sort -nu)
  for q in "${qtypes[@]}"; do
    for artifact in "${artifacts[@]}"; do
      for bc in "${bchunks[@]}"; do
        shard="q${q}-a${artifact}-bc${bc}"
        generated="$out/generated/$shard"
        manifest="$generated/manifest.json"
        shard_evidence=0
        if [ -s "$out/raw/$shard/run.log" ] || \
           [ -e "$out/raw/$shard/run.rc" ] || \
           [ -e "$out/raw/$shard/run.commit.json" ] || \
           [ -e "$out/build/$shard/ppu_targets/test_scalefirst_internal_sweep" ] || \
           [ -L "$out/build/$shard/ppu_targets/test_scalefirst_internal_sweep" ]; then
          shard_evidence=1
        fi
        if [ "$shard_evidence" = 1 ] && [ ! -s "$manifest" ]; then
          printf '[scalefirst-internal] FAIL: %s lost generated manifest\n' "$shard" >&2
          return 2
        fi
        if [ ! -s "$manifest" ]; then
          mkdir -p "$generated" || return 2
          python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
            --qtype "$q" --artifact-tk "$artifact" --bchunk "$bc" \
            --per-unit "$per_unit" --out-dir "$generated" || return 2
        fi
        python3 -B - "$source_hashes" "$shard" "$generated" "$shard_evidence" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
path,shard,generated=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]); evidence=sys.argv[4]=="1"
doc=json.loads(path.read_text()); members={str(p.relative_to(generated)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generated.rglob("*")) if p.is_file()}
digest=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(",",":")).encode()).hexdigest()
old=doc.setdefault("generated_shards",{}).get(shard)
if old is None and evidence: raise SystemExit(f"{shard}: evidence lost generated authority")
if old is not None and old!=digest: raise SystemExit(f"{shard}: generated authority changed")
doc["generated_shards"][shard]=digest
tmp=path.with_name(f".{path.name}.current.{os.getpid()}")
with tmp.open("w") as f: json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
      done
    done
  done
  python3 -B - "$root" "$plan" "$source_hashes" "$out/generated" <<'PY' || return 2
import json,pathlib,sys
root,plan_path,hash_path,generated=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(root/"tools"))
import analyze_scalefirst_internal_sweep as analyzer
plan=analyzer.load_plan(plan_path); expected=analyzer.generated_shards(plan["cells"])
doc=json.loads(hash_path.read_text()); observed=set(doc.get("generated_shards",{}))
dirs={p.parent.name for p in generated.glob("*/manifest.json")}
if observed!=expected or dirs!=expected:
 raise SystemExit(f"generated graph mismatch hashes={sorted(observed^expected)} dirs={sorted(dirs^expected)}")
PY

  source_state="$(sha256sum "$source_hashes" | awk '{print $1}')" || return 2
  run_contract="$out/run-contract.json"
  python3 -B - "$run_contract" "$iterations" "$repeats" "$per_unit" \
    "$peak_tflops" "$hbm_gbs" "$schedule_seed" "$plan_sha" "$source_state" "$identity" \
    "$resume_evidence" <<'PY' || return 2
import hashlib,json,math,os,pathlib,sys
out=pathlib.Path(sys.argv[1]); values=list(map(int,sys.argv[2:5])); peak,hbm=map(float,sys.argv[5:7])
if not all(math.isfinite(x) and x>0 for x in (peak,hbm)): raise SystemExit("invalid metric denominator")
try: seed=int(sys.argv[7],0)
except ValueError: raise SystemExit("SCALEFIRST_SCHEDULE_SEED must be an integer")
if seed < 0 or seed >= 1<<64: raise SystemExit("SCALEFIRST_SCHEDULE_SEED is outside uint64")
doc={"schema":"quactlize.scalefirst_internal_sweep.run_contract.v1",
 "iterations":values[0],"correctness_repeats":values[1],"configs_per_unit":values[2],
 "peak_tflops":peak,"hbm_gbs":hbm,"schedule_seed":seed,
 "plan_sha256":sys.argv[8],"source_state_sha256":sys.argv[9],
 "identity_sha256":hashlib.sha256(pathlib.Path(sys.argv[10]).read_bytes()).hexdigest(),
 "identity_probe_tmpdir":"identity-probe"}
evidence=sys.argv[11]=="1"
if out.exists() and json.loads(out.read_text())!=doc: raise SystemExit("run contract changed on resume")
if not out.exists() and evidence: raise SystemExit("binary/run evidence exists without run contract")
tmp=out.with_name(f".{out.name}.current.{os.getpid()}")
with tmp.open("w") as f: json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
  binary_hashes="$out/binary-hashes.json"
  raw_hashes="$out/raw-log-hashes.json"
  for map in "$binary_hashes" "$raw_hashes"; do
    if [ ! -s "$map" ]; then
      if [ "$resume_evidence" = 1 ]; then
        printf '[scalefirst-internal] FAIL: binary/run evidence exists but %s authority is missing\n' \
          "$(basename "$map")" >&2
        return 2
      fi
      python3 -B - "$map" <<'PY' || return 2
import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); t=p.with_name(f".{p.name}.current.{os.getpid()}")
with t.open("w") as f: json.dump({},f); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
    fi
  done

  local typed
  local binary build_log run_log run_rc_file run_commit current_log existing_rc
  local -a shape_args=()
  for q in "${qtypes[@]}"; do
    shape_args=()
    while IFS=$'\t' read -r _ plan_q m n k _ route support; do
      if [ "$plan_q" = "$q" ] && [ "$route" = dense ] && [ "$support" = SUPPORTED ]; then
        shape_args+=("--shape=${m}x${n}x${k}")
      fi
    done < <(python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" --list-plan "$plan")
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
           [ -e "$out/build/$shard/ppu_targets/test_scalefirst_internal_sweep" ] || \
           [ -L "$out/build/$shard/ppu_targets/test_scalefirst_internal_sweep" ]; then
          shard_evidence=1
        fi
        if [ "$shard_evidence" = 1 ] && [ ! -s "$manifest" ]; then
          printf '[scalefirst-internal] FAIL: %s lost generated manifest\n' "$shard" >&2
          return 2
        fi
        if [ ! -s "$manifest" ]; then
          mkdir -p "$generated" || return 2
          python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
            --qtype "$q" --artifact-tk "$artifact" --bchunk "$bc" \
            --per-unit "$per_unit" --out-dir "$generated" || return 2
        fi
        python3 -B - "$source_hashes" "$shard" "$generated" "$shard_evidence" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
path,shard,generated=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]); evidence=sys.argv[4]=="1"
doc=json.loads(path.read_text()); members={str(p.relative_to(generated)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generated.rglob("*")) if p.is_file()}
digest=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(",",":")).encode()).hexdigest()
old=doc.setdefault("generated_shards",{}).get(shard)
if old is None and evidence: raise SystemExit(f"{shard}: evidence lost generated authority")
if old is not None and old!=digest: raise SystemExit(f"{shard}: generated authority changed")
doc["generated_shards"][shard]=digest
tmp=path.with_name(f".{path.name}.current.{os.getpid()}")
with tmp.open("w") as f: json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
        typed="$(python3 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["denominator"]["typed_rows"])' "$manifest")" || return 2
        if [ "$typed" -eq 0 ]; then
          printf '[scalefirst-internal] static-only shard=%s\n' "$shard"
          continue
        fi
        binary="$out/build/$shard/ppu_targets/test_scalefirst_internal_sweep"
        build_log="$out/build/$shard.log"
        if [ "$shard_evidence" = 1 ]; then
          # Resume may validate an already-bound binary, never rebuild it.
          # This check precedes the build branch deliberately: run evidence
          # plus a missing/directory/symlink/substituted binary is terminal.
          python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" \
            --validate-bind-binary "$binary" --binary-hashes "$binary_hashes" \
            --binary-shard "$shard" --binary-evidence 1 || return 2
        else
          if [ -e "$binary" ] || [ -L "$binary" ]; then
            printf '[scalefirst-internal] FAIL: fresh shard %s has an unexpected binary path\n' \
              "$shard" >&2
            return 2
          fi
          printf '[scalefirst-internal] build shard=%s typed=%s\n' "$shard" "$typed"
          current_log="$build_log.current.$$"
          (cd "$root" && PPU_BUILD_DIR="$out/build/$shard" PPU_ARCHS=ppu0010 \
            JOBS="$jobs" TARGET=test_scalefirst_internal_sweep \
            SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
            SCALEFIRST_SWEEP_QTYPE="$q" SCALEFIRST_SWEEP_ARTIFACT_TK="$artifact" \
            SCALEFIRST_SWEEP_BCHUNK="$bc" ./build.sh) >"$current_log" 2>&1
          rc=$?
          mv -f -- "$current_log" "$build_log" || return 2
          if [ "$rc" -ne 0 ]; then
            printf '[scalefirst-internal] FAIL: build %s rc=%d\n' "$shard" "$rc" >&2
            tail -100 "$build_log" >&2
            return "$rc"
          fi
          # Fresh build admission uses the exact same regular-file predicate
          # as resume, then atomically publishes its first and only digest.
          python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" \
            --validate-bind-binary "$binary" --binary-hashes "$binary_hashes" \
            --binary-shard "$shard" --binary-evidence 0 || return 2
        fi
        run_log="$out/raw/$shard/run.log"
        run_rc_file="$out/raw/$shard/run.rc"
        run_commit="$out/raw/$shard/run.commit.json"
        mkdir -p "$out/raw/$shard" || return 2
        existing_rc="$(python3 -B - "$run_log" "$run_rc_file" "$run_commit" \
          "$run_contract" "$source_hashes" "$binary_hashes" "$raw_hashes" \
          "$shard" <<'PY'
import hashlib, json, os, pathlib, sys
log, rc_path, commit_path, contract_path, source_path, binary_path, raw_path = map(
    pathlib.Path, sys.argv[1:8])
shard = sys.argv[8]
present = [path.exists() for path in (log, rc_path, commit_path)]
if not any(present):
    print("NONE")
    raise SystemExit(0)
if not all(present) or not log.is_file() or log.stat().st_size == 0:
    raise SystemExit(f"{shard}: incomplete run evidence triplet")
try:
    rc_text = rc_path.read_text().strip()
    if not rc_text.isdigit() or not 0 <= int(rc_text) <= 255:
        raise ValueError
    commit = json.loads(commit_path.read_text())
    sources = json.loads(source_path.read_text())
    binaries = json.loads(binary_path.read_text())
    raw_hashes = json.loads(raw_path.read_text())
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(f"{shard}: malformed run evidence")
expected = {
    "schema": "quactlize.scalefirst_internal_sweep.run_commit.v1",
    "rc": int(rc_text),
    "run_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "run_rc_sha256": hashlib.sha256(rc_path.read_bytes()).hexdigest(),
    "run_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "generated_source_sha256": sources.get("generated_shards", {}).get(shard),
    "binary_sha256": binaries.get(shard),
}
if commit != expected or any(value is None for value in expected.values()):
    raise SystemExit(f"{shard}: run evidence authority changed")
old = raw_hashes.get(shard)
if old is not None and old != expected["run_log_sha256"]:
    raise SystemExit(f"{shard}: raw-log hash authority changed")
if old is None:
    # This is the sole recoverable interruption seam: the immutable commit
    # already binds the exact log before the derived digest index is updated.
    raw_hashes[shard] = expected["run_log_sha256"]
    tmp = raw_path.with_name(f".{raw_path.name}.current.{os.getpid()}")
    with tmp.open("w") as stream:
        json.dump(raw_hashes, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, raw_path)
print(rc_text)
PY
)" || return 2
        if [ "$existing_rc" = 0 ]; then
          printf '[scalefirst-internal] resume shard=%s\n' "$shard"
          continue
        fi
        if [ "$existing_rc" != NONE ]; then
          printf '[scalefirst-internal] FAIL: committed runtime %s rc=%s\n' \
            "$shard" "$existing_rc" >&2
          return "$existing_rc"
        fi
        printf '[scalefirst-internal] run shard=%s shapes=%d\n' "$shard" "${#shape_args[@]}"
        current_log="$out/raw/$shard/.run.log.current.$$"
        local current_rc="$out/raw/$shard/.run.rc.current.$$"
        local current_commit="$out/raw/$shard/.run.commit.current.$$"
        "$binary" "${shape_args[@]}" --iterations="$iterations" \
          --correctness-repeats="$repeats" --schedule-seed="$schedule_seed" \
          >"$current_log" 2>&1
        rc=$?
        printf '%d\n' "$rc" > "$current_rc" || return 2
        python3 -B - "$current_log" "$current_rc" "$current_commit" \
          "$run_contract" "$source_hashes" "$binary_hashes" "$shard" <<'PY' || return 2
import hashlib, json, os, pathlib, sys
log, rc_path, output, contract_path, source_path, binary_path = map(
    pathlib.Path, sys.argv[1:7])
shard = sys.argv[7]
sources = json.loads(source_path.read_text())
binaries = json.loads(binary_path.read_text())
doc = {
    "schema": "quactlize.scalefirst_internal_sweep.run_commit.v1",
    "rc": int(rc_path.read_text().strip()),
    "run_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "run_rc_sha256": hashlib.sha256(rc_path.read_bytes()).hexdigest(),
    "run_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "generated_source_sha256": sources["generated_shards"][shard],
    "binary_sha256": binaries[shard],
}
with output.open("w") as stream:
    json.dump(doc, stream, indent=2, sort_keys=True)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
        # Commit moves last.  A killed process leaves an incomplete triplet,
        # which resume rejects rather than silently rerunning or rebinding.
        mv -f -- "$current_log" "$run_log" || return 2
        mv -f -- "$current_rc" "$run_rc_file" || return 2
        mv -f -- "$current_commit" "$run_commit" || return 2
        python3 -B - "$raw_hashes" "$shard" "$run_log" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
path,name,log=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]); doc=json.loads(path.read_text()); digest=hashlib.sha256(log.read_bytes()).hexdigest()
if name in doc and doc[name]!=digest: raise SystemExit(f"{name}: raw log hash changed")
doc[name]=digest; tmp=path.with_name(f".{path.name}.current.{os.getpid()}")
with tmp.open("w") as f: json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
        if [ "$rc" -ne 0 ]; then
          printf '[scalefirst-internal] FAIL: runtime %s rc=%d\n' "$shard" "$rc" >&2
          tail -80 "$run_log" >&2
          return "$rc"
        fi
      done
    done
  done

  python3 -B "$root/tools/analyze_scalefirst_internal_sweep.py" \
    --plan "$plan" --generated-root "$out/generated" --raw-root "$out/raw" \
    --output "$out/results/summary.json" --identity "$identity" \
    --source-hashes "$source_hashes" --binary-hashes "$binary_hashes" \
    --raw-log-hashes "$raw_hashes" --run-contract "$run_contract" \
    --attempt-id "$attempt_id" --peak-tflops "$peak_tflops" --hbm-gbs "$hbm_gbs"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[scalefirst-internal] INCOMPLETE rc=%d artifacts=%s\n' "$rc" "$out" >&2
    return "$rc"
  fi
  python3 -B - "$out/results/summary.json" "$out/results/summary.sha256" <<'PY' || return 2
import hashlib,os,pathlib,sys
source,out=map(pathlib.Path,sys.argv[1:]); value=hashlib.sha256(source.read_bytes()).hexdigest()+"\n"
tmp=out.with_name(f".{out.name}.current.{os.getpid()}")
with tmp.open("w") as f: f.write(value); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
  printf '[scalefirst-internal] PASS attempt=%s artifacts=%s\n' "$attempt_id" "$out"
  return 0
}

main "$@"
