#!/usr/bin/env bash
# Real-GGUF Q4_K ScaleFirst sweep with shape-specific conservative pruning.
#
# Each ArtifactTileK layout is compiled once.  Every real dense prefill shape
# first screens the complete typed layout graph with ordinary NP, then expands
# scheduler choices only for the audited shortlist, and finally confirms that
# shape's retained rows.  Split-K boards are producer-only by construction and
# never compete with FULL_OUTPUT.
set -uo pipefail

main() {
  if [ "$#" -ne 0 ]; then
    printf '[q4k-real-shapes] FAIL: no positional arguments are accepted\n' >&2
    return 2
  fi
  local root workspace_root sha short stamp out resume inventory master
  local frozen_inventory frozen_master plan jobs per_unit base_seed rc
  local artifact expected generated manifest build_dir binary build_log
  local source_authority plan_rows ordinal artifact_evidence
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || {
    printf '[q4k-real-shapes] FAIL: /workspace is unavailable\n' >&2
    return 2
  }
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-q4k-real-shapes-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[q4k-real-shapes] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  resume="${RESUME:-0}"
  case "$resume" in 0|1) ;; *) printf '[q4k-real-shapes] FAIL: RESUME must be 0 or 1\n' >&2; return 2;; esac
  if [ -e "$out" ]; then
    if [ "$resume" != 1 ] || [ ! -d "$out" ]; then
      printf '[q4k-real-shapes] FAIL: refusing existing OUT=%s; set RESUME=1 for a committed-cell resume\n' "$out" >&2
      return 2
    fi
  else
    mkdir -p "$out" || return 2
  fi
  mkdir -p "$out/inputs" "$out/policies" "$out/generated" "$out/build" \
    "$out/raw" "$out/results" "$out/models" || return 2

  inventory="${INTERNAL_SWEEP_SPEC:-}"
  if [ -z "$inventory" ] || [ ! -s "$inventory" ]; then
    printf '[q4k-real-shapes] FAIL: INTERNAL_SWEEP_SPEC must name the COMPLETE inventory-v2 JSON\n' >&2
    return 2
  fi
  inventory="$(realpath -e -- "$inventory")" || return 2
  master="$root/benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json"
  frozen_inventory="$out/inputs/inventory-v2.json"
  frozen_master="$out/inputs/scalefirst_q4k_real_shapes_pruned_policy.json"
  plan="$out/plan.json"
  jobs="${JOBS:-16}"
  per_unit="${SCALEFIRST_CONFIGS_PER_UNIT:-32}"
  base_seed="${SCALEFIRST_SCHEDULE_SEED:-0x6a09e667f3bcc909}"
  case "$jobs:$per_unit" in
    *[!0-9:]*|0:*|*:0)
      printf '[q4k-real-shapes] FAIL: JOBS/SCALEFIRST_CONFIGS_PER_UNIT must be positive integers\n' >&2
      return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[q4k-real-shapes] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the compiled denominator\n' >&2
    return 2
  fi

  python3 -B - "$inventory" "$frozen_inventory" "$master" "$frozen_master" \
    "$resume" <<'PY' || return 2
import os,pathlib,sys
resume=sys.argv[5]=="1"
for source_name,frozen_name,label in ((sys.argv[1],sys.argv[2],"inventory-v2"),
                                      (sys.argv[3],sys.argv[4],"master policy")):
 source=pathlib.Path(source_name); frozen=pathlib.Path(frozen_name)
 data=source.read_bytes()
 if not data: raise SystemExit(f"{label} is empty")
 if frozen.exists():
  if frozen.read_bytes()!=data: raise SystemExit(f"{label} differs from frozen bundle")
 elif resume:
  raise SystemExit(f"resume bundle lost frozen {label}")
 else:
  temporary=frozen.with_name(f".{frozen.name}.current.{os.getpid()}")
  with temporary.open("wb") as stream:
   stream.write(data); stream.flush(); os.fsync(stream.fileno())
  os.replace(temporary,frozen)
PY

  source_authority="$out/inputs/source-authority.json"
  python3 -B - "$root" "$source_authority" "$sha" "$resume" <<'PY' || return 2
import hashlib,json,os,pathlib,subprocess,sys
root,out,commit=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
resume=sys.argv[4]=="1"
paths=[
 "benchmarks/scalefirst_internal_sweep_bench.hpp",
 "benchmarks/scalefirst_internal_sweep_unit.inc",
 "benchmarks/test_scalefirst_internal_sweep.cu",
 "benchmarks/scalefirst_q4k_pruned_policy.json",
 "benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json",
 "ci/check_scalefirst_q4k_pruned_runner.py",
 "quactlize/csrc/scalefirst_internal_sweep.cmake.in",
 "tools/analyze_fully_quantized_internal_sweep.py",
 "tools/gen_scalefirst_internal_units.py",
 "tools/plan_scalefirst_q4k_real_shapes.py",
 "tools/prune_scalefirst_q4k_pilot.py",
 "tools/run_scalefirst_internal_sweep_box.sh",
 "tools/run_scalefirst_q4k_pruned_box.sh",
 "tools/run_scalefirst_q4k_real_shapes_pruned_box.sh",
 "tools/scalefirst_internal_matrix.py",
 "build.sh"]
for rel in paths:
 if not (root/rel).is_file(): raise SystemExit(f"source authority lacks {rel}")
dirty=subprocess.check_output(
 ["git","-C",str(root),"status","--porcelain","--"]+paths,text=True)
if dirty: raise SystemExit("real-shape source authority is dirty:\n"+dirty)
doc={"schema":"quactlize.scalefirst_q4k_real_shapes_source.v1",
     "git_sha":commit,
     "files":{rel:hashlib.sha256((root/rel).read_bytes()).hexdigest()
              for rel in paths}}
if out.exists():
 if json.loads(out.read_text())!=doc: raise SystemExit("source authority changed on resume")
elif resume: raise SystemExit("resume bundle lost source authority")
else:
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
 with temporary.open("w") as stream:
  json.dump(doc,stream,indent=2,sort_keys=True); stream.write("\n")
  stream.flush(); os.fsync(stream.fileno())
 os.replace(temporary,out)
PY

  python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" self-test || return 2
  python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" self-test || return 2
  python3 -B "$root/ci/check_scalefirst_q4k_pruned_runner.py" || return 2

  if [ ! -s "$plan" ]; then
    if [ "$resume" = 1 ]; then
      printf '[q4k-real-shapes] FAIL: resume bundle lost plan.json\n' >&2
      return 2
    fi
    python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" materialize \
      --inventory "$frozen_inventory" --master-policy "$frozen_master" \
      --output "$plan" --policies-dir "$out/policies" || return 2
  fi
  plan_rows="$out/plan.tsv"
  python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" list \
    --plan "$plan" >"$plan_rows" || return 2
  if [ ! -s "$plan_rows" ]; then
    printf '[q4k-real-shapes] FAIL: materialized plan has no cells\n' >&2
    return 2
  fi
  printf '[q4k-real-shapes] plan shapes=%s layout-cells=%s inventory=%s\n' \
    "$(cut -f2 "$plan_rows" | sort -u | wc -l)" "$(wc -l < "$plan_rows")" \
    "$frozen_inventory"

  for artifact in 32 64 128 256; do
    case "$artifact" in
      32) expected=2340 ;;
      64) expected=1824 ;;
      128) expected=1036 ;;
      256) expected=401 ;;
      *) return 2 ;;
    esac
    generated="$out/generated/q12-a${artifact}-bc0"
    manifest="$generated/manifest.json"
    artifact_evidence=0
    if find "$out/raw/a${artifact}" "$out/results/a${artifact}" \
        -mindepth 1 -type f -print -quit 2>/dev/null | grep -q .; then
      artifact_evidence=1
    fi
    if [ ! -s "$manifest" ]; then
      if [ "$artifact_evidence" = 1 ]; then
        printf '[q4k-real-shapes] FAIL: A%s measured evidence lost its generated manifest\n' "$artifact" >&2
        return 2
      fi
      mkdir -p "$generated" || return 2
      python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
        --qtype 12 --artifact-tk "$artifact" --bchunk 0 \
        --per-unit "$per_unit" --out-dir "$generated" || return 2
    fi
    python3 -B - "$manifest" "$artifact" "$expected" <<'PY' || return 2
import json,sys
manifest=json.load(open(sys.argv[1])); artifact=int(sys.argv[2]); expected=int(sys.argv[3])
identity={"qtype":12,"format":"Q4_K","artifact_tile_k":artifact,"bchunk":0}
if manifest.get("identity")!=identity: raise SystemExit("generated identity differs")
rows=manifest.get("typed_rows",[])
if manifest.get("denominator",{}).get("typed_rows")!=expected or len(rows)!=expected:
 raise SystemExit(f"A{artifact} typed denominator differs {len(rows)}/{expected}")
if len({row.get("symbol") for row in rows})!=expected:
 raise SystemExit(f"A{artifact} symbol denominator is not unique")
PY
    python3 -B - "$out/inputs/generated-hashes.json" "$artifact" \
      "$generated" "$artifact_evidence" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
out,artifact,generated=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3])
evidence=sys.argv[4]=="1"
members={str(path.relative_to(generated)):hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(generated.rglob("*")) if path.is_file()}
digest=hashlib.sha256(json.dumps(
 members,sort_keys=True,separators=(",",":")).encode()).hexdigest()
doc=json.loads(out.read_text()) if out.exists() else {}
key=f"q12-a{artifact}-bc0"
if key in doc and doc[key]!=digest: raise SystemExit(f"{key} generated graph changed")
if key not in doc and evidence: raise SystemExit(f"measured evidence lost {key} generated hash")
doc[key]=digest
temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
with temporary.open("w") as stream:
 json.dump(doc,stream,indent=2,sort_keys=True); stream.write("\n")
 stream.flush(); os.fsync(stream.fileno())
os.replace(temporary,out)
PY
    build_dir="$out/build/q12-a${artifact}-bc0"
    binary="$build_dir/ppu_targets/test_scalefirst_internal_sweep"
    build_log="$out/build/q12-a${artifact}-bc0.log"
    if [ ! -x "$binary" ]; then
      if [ "$artifact_evidence" = 1 ]; then
        printf '[q4k-real-shapes] FAIL: A%s measured evidence lost its executable\n' "$artifact" >&2
        return 2
      fi
      printf '[q4k-real-shapes] build A%s typed=%s\n' "$artifact" "$expected"
      (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 \
        JOBS="$jobs" TARGET=test_scalefirst_internal_sweep \
        SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
        SCALEFIRST_SWEEP_QTYPE=12 SCALEFIRST_SWEEP_ARTIFACT_TK="$artifact" \
        SCALEFIRST_SWEEP_BCHUNK=0 ./build.sh) >"$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        printf '[q4k-real-shapes] FAIL: A%s build rc=%d\n' "$artifact" "$rc" >&2
        tail -100 "$build_log" >&2
        return "$rc"
      fi
    fi
    if [ ! -x "$binary" ]; then
      printf '[q4k-real-shapes] FAIL: A%s exact executable is absent\n' "$artifact" >&2
      return 2
    fi
    python3 -B - "$out/inputs/binary-hashes.json" "$artifact" "$binary" \
      "$artifact_evidence" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
out,artifact,binary=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3])
evidence=sys.argv[4]=="1"; digest=hashlib.sha256(binary.read_bytes()).hexdigest()
doc=json.loads(out.read_text()) if out.exists() else {}
key=f"q12-a{artifact}-bc0"
if key in doc and doc[key]!=digest: raise SystemExit(f"{key} binary hash changed")
if key not in doc and evidence: raise SystemExit(f"measured evidence lost {key} binary hash")
doc[key]=digest
temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
with temporary.open("w") as stream:
 json.dump(doc,stream,indent=2,sort_keys=True); stream.write("\n")
 stream.flush(); os.fsync(stream.fileno())
os.replace(temporary,out)
PY
  done

  ordinal=0
  while IFS=$'\t' read -r artifact shape_key shape policy_rel; do
    ordinal=$((ordinal + 1))
    generated="$out/generated/q12-a${artifact}-bc0"
    manifest="$generated/manifest.json"
    binary="$out/build/q12-a${artifact}-bc0/ppu_targets/test_scalefirst_internal_sweep"
    local policy="$out/$policy_rel"
    local raw_dir="$out/raw/a${artifact}/$shape_key"
    local result_dir="$out/results/a${artifact}/$shape_key"
    local commit="$result_dir/commit.json"
    mkdir -p "$raw_dir" "$result_dir" || return 2
    if [ -s "$commit" ]; then
      python3 -B - "$commit" "$out" <<'PY' || return 2
import hashlib,json,pathlib,sys
commit,root=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
doc=json.loads(commit.read_text())
if doc.get("schema")!="quactlize.scalefirst_q4k_shape_commit.v1":
 raise SystemExit(f"malformed cell commit {commit}")
binary_hashes=json.loads((root/"inputs/binary-hashes.json").read_text())
if binary_hashes.get(doc.get("binary_key"))!=doc.get("binary_sha256"):
 raise SystemExit(f"committed cell binary authority changed: {commit}")
for rel,digest in doc.get("files",{}).items():
 path=root/rel
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
  raise SystemExit(f"committed cell authority changed: {rel}")
PY
      printf '[q4k-real-shapes] resume A%s %s committed=PASS\n' "$artifact" "$shape_key"
      continue
    fi
    if find "$raw_dir" "$result_dir" -mindepth 1 -type f -print -quit | grep -q .; then
      printf '[q4k-real-shapes] FAIL: incomplete uncommitted evidence at A%s/%s; use a fresh OUT\n' \
        "$artifact" "$shape_key" >&2
      return 2
    fi
    read -r screen_iterations screen_repeats scheduler_iterations \
      scheduler_repeats confirm_iterations confirm_repeats < <(
        python3 -B - "$policy" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(
 p["screen"]["iterations"],p["screen"]["correctness_repeats"],
 p["scheduler"]["iterations"],p["scheduler"]["correctness_repeats"],
 p["confirm"]["iterations"],p["confirm"]["correctness_repeats"])
PY
      ) || return 2
    local cell_seed=$((base_seed + ordinal * 4 + artifact))
    printf '[q4k-real-shapes] cell=%d A%s shape=%s phase=screen full-typed-graph\n' \
      "$ordinal" "$artifact" "$shape"
    "$binary" --shape="$shape" --algorithm=nonpersistent \
      --iterations="$screen_iterations" \
      --correctness-repeats="$screen_repeats" \
      --schedule-seed="$cell_seed" >"$raw_dir/screen.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      printf '[q4k-real-shapes] FAIL: A%s/%s screen rc=%d\n' "$artifact" "$shape_key" "$rc" >&2
      tail -100 "$raw_dir/screen.log" >&2
      return "$rc"
    fi
    python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" screen \
      --policy "$policy" --manifest "$manifest" --log "$raw_dir/screen.log" \
      --output "$result_dir/screen.json" \
      --symbols-output "$result_dir/screen-shortlist.txt" || return 2

    printf '[q4k-real-shapes] cell=%d A%s shape=%s phase=scheduler shortlist=%s\n' \
      "$ordinal" "$artifact" "$shape" \
      "$(wc -l < "$result_dir/screen-shortlist.txt")"
    "$binary" --shape="$shape" --algorithm=all \
      --symbol-file="$result_dir/screen-shortlist.txt" \
      --iterations="$scheduler_iterations" \
      --correctness-repeats="$scheduler_repeats" \
      --schedule-seed="$((cell_seed + 1))" >"$raw_dir/scheduler.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      printf '[q4k-real-shapes] FAIL: A%s/%s scheduler rc=%d\n' "$artifact" "$shape_key" "$rc" >&2
      tail -100 "$raw_dir/scheduler.log" >&2
      return "$rc"
    fi
    python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" scheduler \
      --policy "$policy" --manifest "$manifest" \
      --log "$raw_dir/scheduler.log" \
      --expected-symbols "$result_dir/screen-shortlist.txt" \
      --output "$result_dir/scheduler.json" \
      --symbols-output "$result_dir/confirm-shortlist.txt" || return 2

    printf '[q4k-real-shapes] cell=%d A%s shape=%s phase=confirm shortlist=%s\n' \
      "$ordinal" "$artifact" "$shape" \
      "$(wc -l < "$result_dir/confirm-shortlist.txt")"
    "$binary" --shape="$shape" --algorithm=all \
      --symbol-file="$result_dir/confirm-shortlist.txt" \
      --iterations="$confirm_iterations" \
      --correctness-repeats="$confirm_repeats" \
      --schedule-seed="$((cell_seed + 2))" >"$raw_dir/confirm.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      printf '[q4k-real-shapes] FAIL: A%s/%s confirm rc=%d\n' "$artifact" "$shape_key" "$rc" >&2
      tail -100 "$raw_dir/confirm.log" >&2
      return "$rc"
    fi
    python3 -B "$root/tools/prune_scalefirst_q4k_pilot.py" confirm \
      --policy "$policy" --manifest "$manifest" \
      --log "$raw_dir/confirm.log" \
      --expected-symbols "$result_dir/confirm-shortlist.txt" \
      --output "$result_dir/summary.json" | tee "$result_dir/winners.txt" || return 2
    python3 -B - "$out" "$commit" "$policy" "$manifest" \
      "$out/inputs/binary-hashes.json" "q12-a${artifact}-bc0" \
      "$raw_dir" "$result_dir" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
root,commit,policy,manifest,binary_hashes=map(pathlib.Path,sys.argv[1:6])
binary_key=sys.argv[6]; raw,result=map(pathlib.Path,sys.argv[7:9])
members=[policy,manifest]
members += sorted(path for path in raw.iterdir() if path.is_file())
members += sorted(path for path in result.iterdir()
                  if path.is_file() and path!=commit)
doc={"schema":"quactlize.scalefirst_q4k_shape_commit.v1",
     "binary_key":binary_key,
     "binary_sha256":json.loads(binary_hashes.read_text())[binary_key],
     "files":{str(path.relative_to(root)):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in members}}
temporary=commit.with_name(f".{commit.name}.current.{os.getpid()}")
with temporary.open("w") as stream:
 json.dump(doc,stream,indent=2,sort_keys=True); stream.write("\n")
 stream.flush(); os.fsync(stream.fileno())
os.replace(temporary,commit)
PY
  done <"$plan_rows"

  python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" summarize \
    --plan "$plan" --results-root "$out/results" \
    --output "$out/summary.json" --tsv "$out/summary.tsv" \
    --models-root "$out/models" | tee "$out/winners.txt" || return 2
  python3 -B - "$out" "$sha" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
root,commit=pathlib.Path(sys.argv[1]),sys.argv[2]
members=[]
for path in sorted(root.rglob("*")):
 if not path.is_file() or path.name=="bundle.json" or ".current." in path.name:
  continue
 rel=path.relative_to(root)
 # Build intermediates and thousands of generated unit sources are bound by
 # binary-hashes.json and generated-hashes.json.  Hashing them again would
 # turn bundle publication into another multi-minute build-sized pass.
 if rel.parts[0]=="build" and path.name!="test_scalefirst_internal_sweep" and \
    not (len(rel.parts)==2 and rel.suffix==".log"):
  continue
 if rel.parts[0]=="generated" and path.name!="manifest.json":
  continue
 members.append(path)
doc={"schema":"quactlize.scalefirst_q4k_real_shapes_bundle.v1",
     "git_sha":commit,"file_count":len(members),
     "files":{str(path.relative_to(root)):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in members}}
out=root/"bundle.json"; temporary=root/f".bundle.json.current.{os.getpid()}"
with temporary.open("w") as stream:
 json.dump(doc,stream,indent=2,sort_keys=True); stream.write("\n")
 stream.flush(); os.fsync(stream.fileno())
os.replace(temporary,out)
PY
  printf '[q4k-real-shapes] PASS sha=%s artifacts=%s summary=%s\n' \
    "$sha" "$out" "$out/summary.tsv"
  return 0
}

main "$@"
