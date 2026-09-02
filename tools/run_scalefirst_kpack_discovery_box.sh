#!/usr/bin/env bash
# Execute a locally compiled canonical K-pack ScaleFirst bundle on one PPU.
# This runner is prebuilt-only: it verifies and executes indexed payloads and
# never invokes a compiler or build system on the box.

set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1

fail() { printf '[sf-kpack-prebuilt] FAIL: %s\n' "$*" >&2; return 2; }
run_atomic() {
  local output="$1"; shift; local temporary="${output}.current.$$"
  "$@" >"$temporary" 2>&1; local rc=$?
  if [ "$rc" -ne 0 ]; then
    tail -160 "$temporary" >&2
    mv "$temporary" "${output}.failed.$(date -u +%Y%m%dT%H%M%SZ)" || true
    return "$rc"
  fi
  mv "$temporary" "$output"
}

main() {
  if [ "$#" -ne 0 ]; then fail "no positional arguments are accepted"; return $?; fi
  local root workspace_root bundle out resume phase sdk plan workloads pilot_limit
  local screen_iterations confirm_iterations repeats
  local require_full scope sdk_identity_fallback
  local identity_probe
  local shard_id q operator manifest_rel binary_rel binary manifest log symbols sidecar count
  local workload source_class m n k tokens topk experts profile rows_file total_rows max_rows rows_sha256
  local -a grouped_args
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  [ -n "${BUNDLE:-}" ] || { fail "BUNDLE is required"; return $?; }
  bundle="$(realpath -e -- "$BUNDLE")" || return 2
  [ -d "$bundle" ] && [ -s "$bundle/bundle.json" ] || {
    fail "BUNDLE lacks bundle.json"; return $?; }
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-sf-kpack-prebuilt-$(date -u +%Y%m%dT%H%M%SZ)-$$}")" || return 2
  case "$out" in "$workspace_root"/*) ;; *)
    fail "OUT must be a strict /workspace child"; return $?;; esac
  resume="${RESUME:-0}"; phase="${PHASE:-all}"; require_full="${REQUIRE_FULL:-0}"
  case "$resume:$require_full" in 0:0|0:1|1:0|1:1) ;; *)
    fail "RESUME and REQUIRE_FULL must be 0 or 1"; return $?;; esac
  case "$phase" in screen|confirm|all) ;; *)
    fail "PHASE must be screen, confirm or all"; return $?;; esac
  if [ -e "$out" ]; then
    if [ "$resume" != 1 ] || [ ! -d "$out" ]; then
      fail "refusing existing OUT; set RESUME=1"; return $?
    fi
  else
    mkdir -p "$out" || return 2
  fi
  mkdir -p "$out"/{inputs,results} || return 2
  export TMPDIR="$out/inputs" TMP="$out/inputs" TEMP="$out/inputs"
  screen_iterations="${SCREEN_ITERATIONS:-5}"
  confirm_iterations="${CONFIRM_ITERATIONS:-11}"
  repeats="${CORRECTNESS_REPEATS:-256}"
  pilot_limit="${PILOT_WORKLOAD_LIMIT:-0}"
  for value in "$screen_iterations" "$confirm_iterations" "$repeats"; do
    case "$value" in ''|*[!0-9]*|0) fail "numeric controls must be positive integers"; return $?;; esac
  done
  case "$pilot_limit" in 0|1) ;; *) fail "PILOT_WORKLOAD_LIMIT must be 0 or 1"; return $?;; esac
  case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*|*[!0-9]*)
    fail "CUDA_VISIBLE_DEVICES must name exactly one device ordinal"; return $?;; esac
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" || {
    fail "set PPU_SDK to the compatible runtime SDK"; return $?; }
  [ -x "$sdk/bin/hgobjdump" ] || {
    fail "box runtime SDK lacks hgobjdump"; return $?; }

  python3 -B "$root/tools/analyze_scalefirst_kpack_discovery.py" self-test || return 2
  python3 -B "$root/tools/scalefirst_kpack_binary_shards.py" self-test || return 2
  python3 -B - "$root" "$bundle" "$out/inputs/bundle.json" \
    "$out/inputs/shard-index.tsv" "$sdk" "$require_full" <<'PY' || return 2
import hashlib,json,os,pathlib,re,subprocess,sys
root,bundle,frozen,index_tsv,sdk=map(pathlib.Path,sys.argv[1:6])
require_full=int(sys.argv[6])
sys.path.insert(0,str(root/"tools"))
import analyze_scalefirst_kpack_discovery as a
import scalefirst_kpack_binary_shards as planner

doc=json.loads((bundle/"bundle.json").read_text())
if doc.get("schema")!="quactlize.scalefirst_kpack_prebuilt_bundle.v2" or \
        doc.get("route")!="scalefirst" or doc.get("scope") not in {"full","pilot"}:
 raise SystemExit("prebuilt bundle schema/route/scope differs")
if require_full and doc["scope"]!="full":
 raise SystemExit("REQUIRE_FULL=1 rejects a pilot bundle")
head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
if head!=doc.get("source_sha"): raise SystemExit("checkout HEAD differs from prebuilt source")
repo=doc.get("repository") or {}
tree=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD^{tree}"],text=True).strip()
if tree!=repo.get("tree"): raise SystemExit("checkout tree differs from prebuilt source")
dirty=[line for line in subprocess.check_output(
 ["git","-C",str(root),"status","--porcelain","--untracked-files=no"],text=True).splitlines()
 if line[3:] not in set(repo.get("tracked_dirty_ignored",[]))]
if dirty: raise SystemExit("box tracked source is dirty")
build_roots=("benchmarks/","ci/","cmake/","dev/","quactlize/",
             "tools/","third_party/")
untracked=[path for path in subprocess.check_output(
 ["git","-C",str(root),"ls-files","--others","--exclude-standard"],
 text=True).splitlines() if path.startswith(build_roots) or path in {
     "CMakeLists.txt","build.sh"}]
if untracked: raise SystemExit("box has untracked build/validation source")
actual_sub=[]
for line in subprocess.check_output(
 ["git","-C",str(root),"submodule","status","--recursive"],text=True).splitlines():
 if not line or line[0]!=" ": raise SystemExit(f"submodule absent/dirty/conflicted: {line}")
 fields=line[1:].split(); path=fields[1]
 current=subprocess.check_output(["git","-C",str(root/path),"rev-parse","HEAD"],text=True).strip()
 if current!=fields[0] or subprocess.check_output(
   ["git","-C",str(root/path),"status","--porcelain"],text=True).strip():
  raise SystemExit(f"submodule current/dirty identity differs: {path}")
 actual_sub.append({"path":path,"gitlink":fields[0],"current":current})
if actual_sub!=repo.get("submodules"): raise SystemExit("recursive submodule identity differs")

def inside(relative):
 if not isinstance(relative,str) or not relative or pathlib.PurePosixPath(relative).is_absolute():
  raise SystemExit("bundle contains a non-relative payload path")
 path=(bundle/relative).resolve(strict=True)
 try: path.relative_to(bundle)
 except ValueError: raise SystemExit("bundle payload path escaped bundle")
 return path
expected_sdk=doc.get("sdk") or {}

authority_row=doc.get("build_input_authority") or {}
authority=inside(authority_row.get("path",""))
if a.sha256(authority)!=authority_row.get("sha256"):
 raise SystemExit("build input authority hash differs")
authority_doc=json.loads(authority.read_text())
plan_row=doc.get("shard_plan") or {}; plan_path=inside(plan_row.get("path",""))
if a.sha256(plan_path)!=plan_row.get("sha256"):
 raise SystemExit("binary shard plan hash differs")
plan=json.loads(plan_path.read_text()); planner.validate_plan(plan)
if (doc.get("scope")!=plan.get("scope") or
    doc.get("parents_per_binary")!=plan.get("parents_per_binary") or
    plan_row.get("pairs")!=plan.get("pairs")):
 raise SystemExit("bundle and binary shard plan differ")
if (authority_doc.get("schema")!="quactlize.scalefirst_kpack_build_input.v1" or
    authority_doc.get("source_sha")!=doc.get("source_sha") or
    authority_doc.get("source_tree")!=repo.get("tree") or
    authority_doc.get("submodules")!=repo.get("submodules") or
    authority_doc.get("sdk")!=expected_sdk or
    authority_doc.get("configuration",{}).get("scope")!=plan["scope"] or
    authority_doc.get("configuration",{}).get("parents_per_binary")!=plan["parents_per_binary"] or
    authority_doc.get("configuration",{}).get("shard_plan_sha256")!=a.sha256(plan_path) or
    authority_doc.get("configuration",{}).get("scratch_policy")!=
       "ONE_SHARD_THEN_COMPACT_PAYLOAD"):
 raise SystemExit("build input authority chain differs")

probe_row=doc.get("runtime_probe") or {}; probe=inside(probe_row.get("binary",""))
probe_receipt=inside(probe_row.get("receipt",""))
expected_probe={
 "schema":"quactlize.scalefirst_kpack_identity_probe_receipt.v1",
 "build_input_authority_sha256":a.sha256(authority),
 "source_sha256":a.sha256(root/"tools/box_identity_probe.cpp"),
 "binary_sha256":a.sha256(probe)}
host=subprocess.check_output(["readelf","-h",str(probe)],text=True)
host_machine=next((line.split(":",1)[1].strip() for line in host.splitlines()
                   if "Machine:" in line),"")
if (not os.access(probe,os.X_OK) or a.sha256(probe)!=probe_row.get("binary_sha256") or
    a.sha256(probe_receipt)!=probe_row.get("receipt_sha256") or
    json.loads(probe_receipt.read_text())!=expected_probe or
    host_machine!=probe_row.get("host_machine")):
 raise SystemExit("prebuilt runtime probe authority differs")

planned=plan["shards"]; rows=doc.get("shards")
if not isinstance(rows,list) or [row.get("shard_id") for row in rows]!=[
        row["shard_id"] for row in planned]:
 raise SystemExit("bundle shard index differs from exact plan order")
coverage={}; index=[]
for expected,row in zip(planned,rows):
 q=int(expected["qtype"]); op=str(expected["operator"]); shard_id=expected["shard_id"]
 if (row.get("shard_id")!=shard_id or row.get("route")!="scalefirst" or
     row.get("qtype")!=q or row.get("operator")!=op or
     row.get("layout")!=a.LAYOUT[q] or
     row.get("mapping_id")!=a.MAPPING[a.LAYOUT[q]] or
     row.get("parent_begin")!=expected["parent_begin"] or
     row.get("parent_end")!=expected["parent_end"] or
     row.get("authority_parents")!=expected["authority_parents"]):
  raise SystemExit(f"{shard_id}: bundle identity differs from plan")
 parent_ids=row.get("parent_ids"); parent_symbols=row.get("parent_symbols")
 if (parent_ids!=list(range(expected["parent_begin"],expected["parent_end"])) or
     parent_symbols!=planner.authority_symbols(op,q)[
         expected["parent_begin"]:expected["parent_end"]]):
  raise SystemExit(f"{shard_id}: stable parent identity differs")
 manifest=inside(row.get("manifest","")); binary=inside(row.get("binary",""))
 binary_receipt=inside(row.get("binary_receipt",""))
 if (a.sha256(manifest)!=row.get("manifest_sha256") or
     a.sha256(binary)!=row.get("binary_sha256") or
     a.sha256(binary_receipt)!=row.get("binary_receipt_sha256")):
  raise SystemExit(f"{shard_id}: manifest/binary hash differs")
 expected_receipt={"schema":"quactlize.scalefirst_kpack_binary_receipt.v1",
  "build_input_authority_sha256":a.sha256(authority),
  "manifest_sha256":a.sha256(manifest),"binary_sha256":a.sha256(binary)}
 if json.loads(binary_receipt.read_text())!=expected_receipt:
  raise SystemExit(f"{shard_id}: binary receipt chain differs")
 parsed=a.validate_manifest(op,q,manifest)
 if ([item["parent_id"] for item in parsed["compiled_parents"]]!=parent_ids or
     [item["symbol"] for item in parsed["compiled_parents"]]!=parent_symbols):
  raise SystemExit(f"{shard_id}: manifest parent IDs/symbols differ")
 elf=subprocess.check_output([str(sdk/"bin/hgobjdump"),"-lelf",str(binary)],
                             text=True,stderr=subprocess.STDOUT)
 match=re.search(r"ELF FILE \d+ \((PPU [^)]+)\)",elf)
 host=subprocess.check_output(["readelf","-h",str(binary)],text=True)
 host_machine=next((line.split(":",1)[1].strip() for line in host.splitlines()
                    if "Machine:" in line),"")
 expected_elf=row.get("elf") or {}
 if (not os.access(binary,os.X_OK) or not match or
     match.group(1)!=expected_elf.get("device_arch") or
     host_machine!=expected_elf.get("host_machine")):
  raise SystemExit(f"{shard_id}: binary ELF architecture differs")
 coverage.setdefault((q,op),[]).extend(parent_ids)
 index.append("\t".join((shard_id,str(q),op,row["manifest"],row["binary"],
                          str(len(parent_ids)))))
for pair in plan["pairs"]:
 key=(int(pair["qtype"]),str(pair["operator"])); ids=coverage.get(key,[])
 limit=(pair["authority_parents"] if plan["scope"]=="full" else
        min(pair["authority_parents"],plan["parents_per_binary"]))
 if ids!=list(range(limit)) or len(ids)!=len(set(ids)):
  raise SystemExit(f"q{key[0]}/{key[1]}: bundle parent union has gap/overlap")
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if frozen.exists() and frozen.read_text()!=encoded:
 raise SystemExit("resumed bundle authority changed")
if not frozen.exists():
 temporary=frozen.with_name(f".{frozen.name}.current.{os.getpid()}")
 temporary.write_text(encoded); os.replace(temporary,frozen)
index_encoded="\n".join(index)+"\n"
if index_tsv.exists() and index_tsv.read_text()!=index_encoded:
 raise SystemExit("resumed shard execution index changed")
if not index_tsv.exists():
 temporary=index_tsv.with_name(f".{index_tsv.name}.current.{os.getpid()}")
 temporary.write_text(index_encoded); os.replace(temporary,index_tsv)
print(f"[sf-kpack-prebuilt] VERIFIED scope={doc['scope']} shards={len(rows)} "
      "hashes=source+plan+manifest+binary+probe")
PY
  scope="$(python3 -B - "$out/inputs/bundle.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["scope"])
PY
)" || return 2
  identity_probe="$(python3 -B - "$out/inputs/bundle.json" "$bundle" <<'PY'
import json,pathlib,sys
doc=json.load(open(sys.argv[1]))
print((pathlib.Path(sys.argv[2])/doc["runtime_probe"]["binary"]).resolve(strict=True))
PY
)" || return 2
  sdk_identity_fallback="${QUACTLIZE_BOX_SDK_COMPILER_IDENTITY:-}"
  if [ -z "$sdk_identity_fallback" ]; then
    sdk_identity_fallback="runtime-inspector-sha256:$(sha256sum "$sdk/bin/hgobjdump" | awk '{print $1}')" || return 2
  fi
  QUACTLIZE_BOX_SDK_COMPILER_IDENTITY="$sdk_identity_fallback" \
  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$out/inputs/device-identity.json" \
    --runtime-probe-binary "$identity_probe" || {
      fail "prebuilt device identity probe failed"; return $?; }
  python3 -B - "$out/inputs/device-identity.json" <<'PY' || return 2
import json,sys
probe=json.load(open(sys.argv[1]))["device_probe"]
if probe.get("device_count")!=1 or probe.get("status") not in {
        "measured","properties-unavailable"}:
 raise SystemExit("one visible measured PPU device is required")
PY
  if [ "$scope" = full ] && [ "$pilot_limit" != 0 ]; then
    fail "full bundle forbids PILOT_WORKLOAD_LIMIT"; return $?
  fi

  # Freeze the canonical cross-route workload authority.  ScaleFirst executes
  # its complete dense/grouped projection; real inventory, router controls and
  # Q4 historical anchors are all measured work.
  plan="$out/inputs/route-plan.json"
  local plan_candidate="$out/inputs/.route-plan.current.$$"
  if [ -n "${PLAN:-}" ]; then
    local supplied
    supplied="$(realpath -e -- "$PLAN")" || return 2
    install -m 0644 "$supplied" "$plan_candidate" || return 2
  else
    python3 -B "$root/tools/plan_fq_kpack_route_optimal.py" materialize \
      --output "$plan_candidate" || return 2
  fi
  python3 -B "$root/tools/plan_fq_kpack_route_optimal.py" validate-plan \
    --plan "$plan_candidate" || return 2
  if [ -e "$plan" ]; then
    cmp -s "$plan_candidate" "$plan" || {
      fail "resumed canonical route plan differs"; return $?; }
    rm -f "$plan_candidate" || return 2
  else
    mv "$plan_candidate" "$plan" || return 2
  fi
  workloads="$out/inputs/workloads"
  python3 -B "$root/tools/materialize_kpack_discovery_workloads.py" materialize \
    --plan "$plan" --output "$workloads" || return 2

  python3 -B - "$out/inputs/bundle.json" "$out/inputs/device-identity.json" \
    "$out/inputs/result-authority.json" "$sdk" "$root" "$plan" \
    "$workloads/index.json" "$out/inputs" "$pilot_limit" \
    "$screen_iterations" "$confirm_iterations" "$repeats" "$phase" \
    "$identity_probe" "$bundle" <<'PY' || return 2
import csv,hashlib,json,os,pathlib,re,subprocess,sys
bundle,device,out,sdk,root,plan_path,workload_index_path,inputs=map(pathlib.Path,sys.argv[1:9])
limit,screen,confirm,repeats=map(int,sys.argv[9:13]); phase=sys.argv[13]
identity_probe=pathlib.Path(sys.argv[14]); bundle_root=pathlib.Path(sys.argv[15])
sys.path.insert(0,str(root/"tools"))
import materialize_kpack_discovery_workloads as workload_authority
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
index=json.loads(bundle.read_text()); plan=json.loads(plan_path.read_text())
workload_index=workload_authority.validate(plan_path,workload_index_path.parent)
if workload_index.get("format_cells")!=1381:
 raise SystemExit("complete workload denominator differs")

def write_frozen(path,text):
 if path.exists() and path.read_text()!=text:
  raise SystemExit(f"resumed workload projection differs: {path.name}")
 if not path.exists():
  temporary=path.with_name(f".{path.name}.current.{os.getpid()}")
  temporary.write_text(text); os.replace(temporary,path)

qtypes=sorted({int(row["qtype"]) for row in index["shards"]})
executed={}
for qtype in qtypes:
 per_q={}
 for operator,expected in (("dense",429 if qtype==12 else 143),("grouped",76)):
  source=workload_index_path.parent/f"q{qtype}.{operator}.tsv"
  with source.open(newline="") as stream:
   rows=list(csv.DictReader(stream,delimiter="\t"))
  if len(rows)!=expected:
   raise SystemExit(f"q{qtype}/{operator}: complete workload projection differs")
  if limit: rows=rows[:limit]
  for row in rows:
   if not re.fullmatch(r"[A-Za-z0-9_.-]+",row["workload_key"]):
    raise SystemExit("workload key is not path-safe")
  columns=list(rows[0]) if rows else []
  text="\t".join(columns)+"\n"+"".join(
      "\t".join(row[column] for column in columns)+"\n" for row in rows)
  write_frozen(inputs/f"q{qtype}.{operator}.tsv",text)
  per_q[operator]=len(rows)
 executed[str(qtype)]=per_q

def runtime_file(path,relative=False):
 return {"path":str(path.relative_to(sdk) if relative else path),
         "size":path.stat().st_size,"sha256":sha(path),
         "symlink_target":os.readlink(path) if path.is_symlink() else None}
runtime_files=[]
for path in [sdk/"release.yaml",sdk/"VERSION.txt",sdk/"bin/hgobjdump"]:
 if path.exists() and (path.is_file() or path.is_symlink()):
  runtime_files.append(runtime_file(path,True))
runtime_candidates=[runtime_file(path,True) for path in sorted(
    sdk.rglob("libhggc*.so*")) if path.is_file() or path.is_symlink()]
if not runtime_candidates: raise SystemExit("runtime SDK exposes no libhggc candidates")
def loaded_libhggc(binary):
 output=subprocess.check_output(["ldd",str(binary)],text=True,stderr=subprocess.STDOUT)
 loaded=[]
 for line in output.splitlines():
  match=re.match(r"\s*(libhggc\S*)\s+=>\s+(\S+)",line)
  if not match: continue
  path=pathlib.Path(match.group(2))
  if not path.is_file(): raise SystemExit(f"loaded runtime library is unavailable: {line}")
  loaded.append({"soname":match.group(1),**runtime_file(path.resolve())})
 if not loaded: raise SystemExit(f"cannot identify loaded libhggc runtime libraries: {binary}")
 return loaded
loaded=loaded_libhggc(identity_probe)
signature=lambda rows:sorted((row["soname"],row["path"],row["size"],row["sha256"]) for row in rows)
probe_signature=signature(loaded); checked=[]
for row in index["shards"]:
 binary_path=(bundle_root/row["binary"]).resolve(strict=True)
 if signature(loaded_libhggc(binary_path))!=probe_signature:
  raise SystemExit(f"{row['shard_id']}: payload/identity-probe libhggc sets differ")
 checked.append(row["shard_id"])
doc={"schema":"quactlize.scalefirst_kpack_result_authority.v3",
 "scope":index["scope"],"route":"scalefirst","phase":phase,
 "bundle_sha256":sha(bundle),"canonical_route_plan_sha256":sha(plan_path),
 "workload_index_sha256":sha(workload_index_path),
 "device_identity_sha256":sha(device),
 "shard_ids":[row["shard_id"] for row in index["shards"]],
 "heuristic_algorithm_denominator":["NONPERSISTENT","PERSISTENT"],
 "split_k_policy":"EXCLUDED_DIAGNOSTIC_ONLY",
 "full_denominator":{"format_cells":1381,"dense_cells":1001,
                     "grouped_cells":380,"router_controls":120,
                     "q4_historical_anchors":286},
 "pilot_workload_limit":limit,"executed_per_qtype":executed,
 "measurement":{"screen_iterations":screen,"confirm_iterations":confirm,
                "correctness_repeats":repeats,
                "dense_algorithm":"full-output",
                "grouped_algorithms":["GROUPED_NONPERSISTENT","GROUPED_PERSISTENT"]},
 "runtime_sdk":{"root":str(sdk),"files":runtime_files,
                "libhggc_candidates":runtime_candidates,
                "loaded_libhggc":loaded,
                "payload_loaded_set":{"verdict":"ALL_EQUAL_TO_IDENTITY_PROBE",
                                      "checked_shards":checked}},
 "device":json.loads(device.read_text())}
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if out.exists() and out.read_text()!=encoded:
 raise SystemExit("result authority changed on resume")
if not out.exists():
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
 temporary.write_text(encoded); os.replace(temporary,out)
PY

  if [ "$phase" = screen ] || [ "$phase" = all ]; then
    while IFS=$'\t' read -r shard_id q operator manifest_rel binary_rel count; do
      [ -n "$shard_id" ] || { fail "empty row in shard execution index"; return $?; }
      binary="$bundle/$binary_rel"; manifest="$bundle/$manifest_rel"
      if [ "$operator" = dense ]; then
        while IFS=$'\t' read -r workload source_class m n k; do
          [ "$workload" != workload_key ] || continue
          log="$out/results/$shard_id.$workload.screen.log"
          if [ ! -s "$log" ]; then
            run_atomic "$log" "$binary" --shape="${m}x${n}x${k}" \
              --algorithm=full-output --iterations="$screen_iterations" \
              --correctness-repeats="$repeats" || return $?
          fi
          grep -q "^SF_COMPLETE status=COMPLETE shape=${m}x${n}x${k} typed_rows=$count " "$log" || {
            fail "$shard_id dense $workload screen incomplete"; return $?; }
          symbols="$out/results/$shard_id.$workload.symbols"
          sidecar="$out/results/$shard_id.$workload.retention.json"
          python3 -B "$root/tools/analyze_scalefirst_kpack_discovery.py" retain \
            --operator dense --qtype "$q" --screen "$log" --manifest "$manifest" \
            --output "$symbols" --sidecar "$sidecar" || return 2
        done < "$out/inputs/q$q.dense.tsv"
      else
        while IFS=$'\t' read -r workload source_class tokens topk experts n k \
            profile rows_file total_rows max_rows rows_sha256; do
          [ "$workload" != workload_key ] || continue
          log="$out/results/$shard_id.$workload.screen.log"
          if [ ! -s "$log" ]; then
            grouped_args=(--experts="$experts" --n="$n" --k="$k"
              --workload-key="$workload" --router-profile="$profile")
            if [ "$rows_file" = - ]; then
              grouped_args+=(--tokens="$tokens" --topk="$topk")
            else
              grouped_args+=(--rows-file="$workloads/$rows_file")
            fi
            run_atomic "$log" "$binary" "${grouped_args[@]}" \
              --iterations="$screen_iterations" --correctness-repeats="$repeats" || return $?
          fi
          grep -q "^SF_GROUPED_SHARD .*total_rows=$total_rows max_rows=$max_rows .*workload=$workload router_profile=$profile .*" "$log" || {
            fail "$shard_id grouped $workload fixture identity differs"; return $?; }
          grep -q "^SF_GROUPED_COMPLETE .*status=PASS rows=$count " "$log" || {
            fail "$shard_id grouped $workload screen incomplete"; return $?; }
          symbols="$out/results/$shard_id.$workload.symbols"
          sidecar="$out/results/$shard_id.$workload.retention.json"
          python3 -B "$root/tools/analyze_scalefirst_kpack_discovery.py" retain \
            --operator grouped --qtype "$q" --screen "$log" --manifest "$manifest" \
            --output "$symbols" --sidecar "$sidecar" || return 2
        done < "$out/inputs/q$q.grouped.tsv"
      fi
    done < "$out/inputs/shard-index.tsv"
  fi
  if [ "$phase" = screen ]; then
    printf '[sf-kpack-prebuilt] SCREEN_COMPLETE scope=%s elimination=STRUCTURAL-ONLY top_n=NONE artifacts=%s\n' \
      "$scope" "$out"
    return 0
  fi

  while IFS=$'\t' read -r shard_id q operator manifest_rel binary_rel count; do
    [ -n "$shard_id" ] || { fail "empty row in shard execution index"; return $?; }
    binary="$bundle/$binary_rel"
    if [ "$operator" = dense ]; then
      while IFS=$'\t' read -r workload source_class m n k; do
        [ "$workload" != workload_key ] || continue
        symbols="$out/results/$shard_id.$workload.symbols"
        [ -s "$symbols" ] || { fail "$shard_id/$workload retention missing"; return $?; }
        log="$out/results/$shard_id.$workload.confirm.log"
        if [ ! -s "$log" ]; then
          run_atomic "$log" "$binary" --shape="${m}x${n}x${k}" \
            --algorithm=full-output --iterations="$confirm_iterations" \
            --correctness-repeats="$repeats" --symbol-file="$symbols" || return $?
        fi
        grep -q "^SF_COMPLETE status=COMPLETE shape=${m}x${n}x${k} .*" "$log" || {
          fail "$shard_id dense $workload confirmation incomplete"; return $?; }
      done < "$out/inputs/q$q.dense.tsv"
    else
      while IFS=$'\t' read -r workload source_class tokens topk experts n k \
          profile rows_file total_rows max_rows rows_sha256; do
        [ "$workload" != workload_key ] || continue
        symbols="$out/results/$shard_id.$workload.symbols"
        [ -s "$symbols" ] || { fail "$shard_id/$workload retention missing"; return $?; }
        log="$out/results/$shard_id.$workload.confirm.log"
        if [ ! -s "$log" ]; then
          grouped_args=(--experts="$experts" --n="$n" --k="$k"
            --workload-key="$workload" --router-profile="$profile"
            --symbol-file="$symbols")
          if [ "$rows_file" = - ]; then
            grouped_args+=(--tokens="$tokens" --topk="$topk")
          else
            grouped_args+=(--rows-file="$workloads/$rows_file")
          fi
          run_atomic "$log" "$binary" "${grouped_args[@]}" \
            --iterations="$confirm_iterations" --correctness-repeats="$repeats" || return $?
        fi
        grep -q "^SF_GROUPED_SHARD .*total_rows=$total_rows max_rows=$max_rows .*workload=$workload router_profile=$profile .*" "$log" || {
          fail "$shard_id grouped $workload confirmation fixture differs"; return $?; }
        grep -q '^SF_GROUPED_COMPLETE .*status=PASS' "$log" || {
          fail "$shard_id grouped $workload confirmation incomplete"; return $?; }
      done < "$out/inputs/q$q.grouped.tsv"
    fi
  done < "$out/inputs/shard-index.tsv"
  printf '[sf-kpack-prebuilt] DIAGNOSTIC_COMPLETE scope=%s shards=%s workloads=COMPLETE top_n=NONE artifacts=%s\n' \
    "$scope" "$(wc -l < "$out/inputs/shard-index.tsv")" "$out"
}

main "$@"
