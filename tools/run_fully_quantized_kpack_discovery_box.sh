#!/usr/bin/env bash
# Run an immutable parent-range FullyQuantized K-pack discovery bundle on PPU.
# PREBUILT ONLY: no compiler, CMake, build.sh, or generator is invoked here.

set -uo pipefail

fail() { printf '[fq-kpack-prebuilt] FAIL: %s\n' "$*" >&2; return 2; }
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
  [ "$#" -eq 0 ] || { fail "no positional arguments are accepted"; return $?; }
  local root workspace bundle out resume phase sdk runtime_sdk_receipt plan workloads screen_iterations confirm_iterations repeats
  local shard_index mode shard_count key q op binary manifest begin end count authority log
  local workload tokens n k experts topk identity_probe pilot_limit
  local -a dense_shapes dense_args grouped_args
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  [ -n "${BUNDLE:-}" ] || { fail "BUNDLE is required"; return $?; }
  bundle="$(realpath -e -- "$BUNDLE")" || return 2
  [ -d "$bundle" ] && [ -s "$bundle/bundle.json" ] || { fail "BUNDLE lacks bundle.json"; return $?; }
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-kpack-prebuilt-$(date -u +%Y%m%dT%H%M%SZ)-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail "OUT must be a strict /workspace child"; return $?;; esac
  resume="${RESUME:-0}"; phase="${PHASE:-all}"
  case "$resume" in 0|1) ;; *) fail "RESUME must be 0 or 1"; return $?;; esac
  case "$phase" in screen|confirm|all) ;; *) fail "PHASE must be screen, confirm or all"; return $?;; esac
  if [ -e "$out" ]; then
    [ "$resume" = 1 ] && [ -d "$out" ] || { fail "refusing existing OUT; set RESUME=1"; return $?; }
  else mkdir -p "$out" || return 2; fi
  mkdir -p "$out"/{inputs,results} || return 2
  export TMPDIR="$out/inputs" TMP="$out/inputs" TEMP="$out/inputs" PYTHONDONTWRITEBYTECODE=1
  screen_iterations="${SCREEN_ITERATIONS:-5}"
  confirm_iterations="${CONFIRM_ITERATIONS:-11}"
  repeats="${CORRECTNESS_REPEATS:-256}"
  case "$screen_iterations:$confirm_iterations:$repeats" in
    *[!0-9:]*|0:*|*:0:*|*:*:0) fail "iteration controls must be positive"; return $?;; esac
  case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*|*[!0-9]*)
    fail "CUDA_VISIBLE_DEVICES must name exactly one ordinal"; return $?;; esac
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" || {
    fail "set PPU_SDK to the runtime SDK"; return $?; }
  [ -x "$sdk/bin/hgobjdump" ] || { fail "runtime SDK lacks hgobjdump"; return $?; }
  if [ -f "$sdk/release.yaml" ]; then runtime_sdk_receipt="$sdk/release.yaml";
  elif [ -f "$sdk/VERSION.txt" ]; then runtime_sdk_receipt="$sdk/VERSION.txt";
  else fail "runtime SDK lacks release.yaml/VERSION.txt"; return $?; fi

  # Freeze and validate the complete range index, source/submodule identity,
  # manifest/binary/receipt chain, and every PPU image before the first run.
  python3 -B - "$root" "$bundle" "$out/inputs/bundle.json" "$sdk" <<'PY' || return 2
import hashlib,json,os,pathlib,re,subprocess,sys
root,bundle,frozen,sdk=map(pathlib.Path,sys.argv[1:]); doc=json.loads((bundle/"bundle.json").read_text())
sys.path.insert(0,str(root/"tools"))
import fully_quantized_kpack_bundle_index as index
import gen_fully_quantized_grouped_kpack_units as grouped
import gen_fully_quantized_kpack_discovery_units as dense
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bundle=bundle.resolve()
def inside(relative,label):
 if not isinstance(relative,str) or not relative or pathlib.Path(relative).is_absolute():
  raise SystemExit(f"{label}: bundle path is not strict relative")
 try:
  path=(bundle/relative).resolve(strict=True); path.relative_to(bundle)
 except (OSError,ValueError) as exc:
  raise SystemExit(f"{label}: bundle path escapes or is missing: {exc}")
 return path
index.validate_index(doc)
if subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()!=doc.get("source_sha") or \
   subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD^{tree}"],text=True).strip()!=doc.get("source_tree"):
 raise SystemExit("checkout source identity differs from bundle")
dirty=[line for line in subprocess.check_output(
 ["git","-C",str(root),"status","--porcelain","--untracked-files=no"],text=True).splitlines()
 if line[3:] not in {".coord/BOX.md",".coord/INBOX.md"}]
if dirty: raise SystemExit("box checkout has tracked source inputs")
untracked=subprocess.check_output(
 ["git","-C",str(root),"ls-files","--others","--exclude-standard"],text=True).splitlines()
source_roots=("benchmarks/","ci/","cmake/","dev/","quactlize/","tools/","third_party/")
relevant=[path for path in untracked if path in {"CMakeLists.txt","build.sh"} or path.startswith(source_roots)]
if relevant: raise SystemExit("box checkout has untracked build/validation source inputs")
actual=[]
for line in subprocess.check_output(["git","-C",str(root),"submodule","status","--recursive"],text=True).splitlines():
 if not line or line[0]!=" ": raise SystemExit("submodule absent/dirty/conflicted")
 fields=line[1:].split(); checkout=root/fields[1]
 current=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
 if current!=fields[0] or subprocess.check_output(["git","-C",str(checkout),"status","--porcelain"],text=True).strip():
  raise SystemExit(f"submodule identity differs: {fields[1]}")
 actual.append({"path":fields[1],"gitlink":fields[0],"current":current})
if actual!=doc.get("submodules"): raise SystemExit("recursive submodule census differs")
authority=inside(doc.get("build_input_authority"),"build-input-authority")
if not authority.is_file() or sha(authority)!=doc.get("build_input_authority_sha256"):
 raise SystemExit("build input authority differs")
build=json.loads(authority.read_text())
if build.get("source_sha")!=doc.get("source_sha") or build.get("source_tree")!=doc.get("source_tree"):
 raise SystemExit("build input source chain differs")
if build.get("sdk")!=doc.get("sdk"):
 raise SystemExit("top-level bundle SDK differs from build authority")
probe=doc.get("runtime_identity_probe") or {}
probe_binary=inside(probe.get("path"),"runtime-identity-probe")
probe_receipt=inside(probe.get("receipt"),"runtime-identity-probe-receipt")
if (not probe_binary.is_file() or not os.access(probe_binary,os.X_OK) or sha(probe_binary)!=probe.get("sha256") or
    not probe_receipt.is_file() or sha(probe_receipt)!=probe.get("receipt_sha256")):
 raise SystemExit("prebuilt runtime identity probe differs")
def loaded_libhggc(binary):
 output=subprocess.check_output(["ldd",str(binary)],text=True,stderr=subprocess.STDOUT)
 rows=[]
 for line in output.splitlines():
  match=re.match(r"\s*(libhggc\S*)\s+=>\s+(\S+)",line)
  if not match: continue
  path=pathlib.Path(match.group(2))
  if not path.is_file(): raise SystemExit(f"{binary}: loaded runtime library is unavailable: {line}")
  path=path.resolve(); rows.append({"soname":match.group(1),"path":str(path),
    "size":path.stat().st_size,"sha256":sha(path),"symlink_target":None})
 if not rows: raise SystemExit(f"{binary}: no loaded libhggc runtime set")
 return sorted(rows,key=lambda row:(row["soname"],row["path"]))
probe_loaded=loaded_libhggc(probe_binary); payload_count=0
for key,row in sorted(doc["shards"].items()):
 manifest=inside(row.get("manifest"),f"{key}:manifest")
 binary=inside(row.get("binary"),f"{key}:binary")
 receipt=inside(row.get("binary_receipt"),f"{key}:receipt")
 if (not manifest.is_file() or not binary.is_file() or not os.access(binary,os.X_OK) or not receipt.is_file() or
     sha(manifest)!=row["manifest_sha256"] or sha(binary)!=row["binary_sha256"] or
     sha(receipt)!=row["binary_receipt_sha256"]):
  raise SystemExit(f"{key}: payload hash differs")
 parsed=json.loads(manifest.read_text()); (dense.validate_manifest if row["operator"]=="dense" else grouped.validate_manifest)(parsed)
 parents=parsed["dense_tc_parents"] if row["operator"]=="dense" else parsed["grouped_parents"]
 if [x["static_candidate_id"] for x in parents]!=row["parent_ids"]:
  raise SystemExit(f"{key}: manifest parent ids differ")
 receipt_doc=json.loads(receipt.read_text()); index.validate_receipt(receipt_doc,row,sha(manifest),sha(binary))
 expected_chain={
  "build_input_authority_sha256":sha(authority),
  "source_sha":build["source_sha"],"source_tree":build["source_tree"],
  "submodules":build["submodules"],
  "sdk_compiler_sha256":build["sdk"]["compiler"]["sha256"],
  "sdk_inspector_sha256":build["sdk"]["inspector"]["sha256"],
  "manifest":row["manifest"],"binary":row["binary"],
  "device_arch":row["device_arch"],
  "inspector_output_sha256":row["inspector_output_sha256"],
 }
 if any(receipt_doc.get(field)!=value for field,value in expected_chain.items()):
  raise SystemExit(f"{key}: binary receipt authority chain differs")
 elf=subprocess.check_output([str(sdk/"bin/hgobjdump"),"-lelf",str(binary)],text=True,stderr=subprocess.STDOUT)
 match=re.search(r"ELF FILE \d+ \((PPU [^)]+)\)",elf)
 if not match or match.group(1)!=row["device_arch"]:
  raise SystemExit(f"{key}: PPU image architecture differs")
 if loaded_libhggc(binary)!=probe_loaded:
  raise SystemExit(f"{key}: payload/identity-probe loaded libhggc sets differ")
 payload_count+=1
linkage={"schema":"quactlize.fq_kpack_payload_runtime_linkage.v1",
 "identity_probe":str(probe_binary),"loaded_libhggc":probe_loaded,
 "payload_binary_count":payload_count,"all_payloads_same_runtime":True}
linkage_path=frozen.parent/"payload-runtime-linkage.json"
linkage_encoded=json.dumps(linkage,indent=2,sort_keys=True)+"\n"
if linkage_path.exists() and linkage_path.read_text()!=linkage_encoded:
 raise SystemExit("resumed payload runtime linkage differs")
if not linkage_path.exists():
 temporary=linkage_path.with_name(f".{linkage_path.name}.current.{os.getpid()}")
 temporary.write_text(linkage_encoded); os.replace(temporary,linkage_path)
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if frozen.exists() and frozen.read_text()!=encoded: raise SystemExit("resumed bundle authority changed")
if not frozen.exists():
 temporary=frozen.with_name(f".{frozen.name}.current.{os.getpid()}"); temporary.write_text(encoded); os.replace(temporary,frozen)
print(f"[fq-kpack-prebuilt] VERIFIED mode={doc['mode']} shards={len(doc['shards'])} parent-union=EXACT")
PY
  identity_probe="$(python3 -B - "$out/inputs/bundle.json" "$bundle" <<'PY'
import json,pathlib,sys
doc=json.load(open(sys.argv[1])); print(pathlib.Path(sys.argv[2])/doc["runtime_identity_probe"]["path"])
PY
)" || return 2
  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$out/inputs/device-identity.json" \
    --runtime-probe-binary "$identity_probe" || return 2

  mode="$(python3 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["mode"])' "$out/inputs/bundle.json")" || return 2
  pilot_limit="${PILOT_WORKLOAD_LIMIT:-0}"
  case "$pilot_limit" in 0|1) ;; *) fail "PILOT_WORKLOAD_LIMIT must be 0 or 1"; return $?;; esac
  if [ "$mode" = FULL ] && [ "$pilot_limit" != 0 ]; then
    fail "FULL bundle forbids PILOT_WORKLOAD_LIMIT"; return $?
  fi

  # Always materialize/validate the complete 1,381-cell product plan.  Router
  # controls and Q4 historical anchors are measured work, not comments.
  # Resume compares exact bytes so a stale caller plan cannot change the
  # workload denominator.
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
    cmp -s "$plan_candidate" "$plan" || { fail "resumed canonical route plan differs"; return $?; }
    rm -f "$plan_candidate" || return 2
  else
    mv "$plan_candidate" "$plan" || return 2
  fi
  workloads="$out/inputs/workloads"
  python3 -B "$root/tools/materialize_kpack_discovery_workloads.py" materialize \
    --plan "$plan" --output "$workloads" || return 2

  python3 -B - "$plan" "$out/inputs/bundle.json" "$out/inputs" \
    "$out/inputs/result-authority.json" "$pilot_limit" \
    "$screen_iterations" "$confirm_iterations" "$repeats" "$phase" \
    "$out/inputs/device-identity.json" "$sdk" "$runtime_sdk_receipt" \
    "$identity_probe" "$out/inputs/payload-runtime-linkage.json" \
    "$workloads/index.json" <<'PY' || return 2
import hashlib,json,os,pathlib,re,subprocess,sys
plan_path,bundle_path,inputs,authority=map(pathlib.Path,sys.argv[1:5])
limit,screen,confirm,repeats=map(int,sys.argv[5:9]); phase=sys.argv[9]
device=pathlib.Path(sys.argv[10]); sdk=pathlib.Path(sys.argv[11]); runtime_receipt=pathlib.Path(sys.argv[12])
identity_probe=pathlib.Path(sys.argv[13])
linkage_path=pathlib.Path(sys.argv[14]); linkage=json.loads(linkage_path.read_text())
workload_index_path=pathlib.Path(sys.argv[15]); workload_index=json.loads(workload_index_path.read_text())
plan=json.loads(plan_path.read_text()); bundle=json.loads(bundle_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def runtime_file(path):
 return {"path":str(path),"size":path.stat().st_size,"sha256":sha(path),
         "symlink_target":os.readlink(path) if path.is_symlink() else None}
runtime_candidates=[runtime_file(path) for path in sorted(sdk.rglob("libhggc*.so*"))
                    if path.is_file() or path.is_symlink()]
if not runtime_candidates: raise SystemExit("runtime SDK exposes no libhggc candidates")
ldd=subprocess.check_output(["ldd",str(identity_probe)],text=True,stderr=subprocess.STDOUT)
loaded=[]
for line in ldd.splitlines():
 match=re.match(r"\s*(libhggc\S*)\s+=>\s+(\S+)",line)
 if not match: continue
 path=pathlib.Path(match.group(2))
 if not path.is_file(): raise SystemExit(f"loaded runtime library is unavailable: {line}")
 loaded.append({"soname":match.group(1),**runtime_file(path.resolve())})
if not loaded: raise SystemExit("cannot identify loaded libhggc runtime libraries")
loaded=sorted(loaded,key=lambda row:(row["soname"],row["path"]))
if (linkage.get("loaded_libhggc")!=loaded or
        linkage.get("all_payloads_same_runtime") is not True or
        not isinstance(linkage.get("payload_binary_count"),int) or
        linkage["payload_binary_count"]<=0):
 raise SystemExit("payload runtime linkage authority differs")
if (workload_index.get("schema")!="quactlize.kpack-discovery-workloads.v1" or
    workload_index.get("plan_file_sha256")!=sha(plan_path) or
    workload_index.get("format_cells")!=1381):
 raise SystemExit("complete workload projection authority differs")
workload_root=workload_index_path.parent
qtypes=sorted({row["qtype"] for row in bundle["shards"].values()})
executed={}
for qtype in qtypes:
 dense_lines=(workload_root/f"q{qtype}.dense.tsv").read_text().splitlines()
 grouped_lines=(workload_root/f"q{qtype}.grouped.tsv").read_text().splitlines()
 expected_dense=429 if qtype==12 else 143
 if len(dense_lines)!=expected_dense+1 or len(grouped_lines)!=77:
  raise SystemExit(f"q{qtype}: complete workload denominator differs")
 dense_rows=dense_lines[1:]; grouped_rows=grouped_lines[1:]
 if limit:
  dense_rows=dense_rows[:limit]; grouped_rows=grouped_rows[:limit]
 dense_text="".join(row.split("\t")[2]+"x"+row.split("\t")[3]+"x"+
                    row.split("\t")[4]+"\n" for row in dense_rows)
 grouped_text="\n".join(grouped_rows)+"\n"
 for path,text in ((inputs/f"q{qtype}.dense.shapes",dense_text),
                   (inputs/f"q{qtype}.grouped.tsv",grouped_text)):
  if path.exists() and path.read_text()!=text: raise SystemExit(f"resumed workload projection differs: {path.name}")
  if not path.exists():
   temporary=path.with_name(f".{path.name}.current.{os.getpid()}"); temporary.write_text(text); os.replace(temporary,path)
 executed[str(qtype)]={"dense":len(dense_rows),"grouped":len(grouped_rows)}
doc={"schema":"quactlize.fq_kpack_discovery_result_authority.v1",
 "bundle_sha256":sha(bundle_path),"canonical_route_plan_sha256":sha(plan_path),
 "device_identity_sha256":sha(device),
 "runtime_sdk":{"receipt_kind":runtime_receipt.name,
   "receipt_sha256":sha(runtime_receipt),
   "inspector_sha256":sha(sdk/"bin/hgobjdump"),
   "libhggc_candidates":runtime_candidates,
   "loaded_libhggc":loaded,
   "payload_runtime_linkage_sha256":sha(linkage_path),
   "validated_payload_binary_count":linkage["payload_binary_count"]},
 "source_class":"complete-product-denominator","bundle_mode":bundle["mode"],
 "phase":phase,
 "admission":("DIAGNOSTIC_ONLY" if bundle["mode"]=="PILOT" else
              "SCREEN_DIAGNOSTIC" if phase=="screen" else
              "CONFIRM_REQUIRED_FOR_HEURISTIC"),
 "pilot_workload_limit":limit,
 "full_denominator":{"format_cells":1381,"dense_cells":1001,
                     "grouped_cells":380,"router_controls":120,
                     "q4_historical_anchors":286},
 "workload_index_sha256":sha(workload_index_path),
 "executed_qtypes":qtypes,"executed_per_qtype":executed,
 "timing":{"screen_iterations":screen,"confirm_iterations":confirm,
           "correctness_repeats":repeats}}
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if authority.exists() and authority.read_text()!=encoded: raise SystemExit("resumed result authority differs")
if not authority.exists():
 temporary=authority.with_name(f".{authority.name}.current.{os.getpid()}"); temporary.write_text(encoded); os.replace(temporary,authority)
PY
  shard_index="$out/inputs/shards.tsv"
  python3 -B - "$out/inputs/bundle.json" "$bundle" "$shard_index" <<'PY' || return 2
import json,os,pathlib,sys
doc=json.load(open(sys.argv[1])); bundle=pathlib.Path(sys.argv[2]); out=pathlib.Path(sys.argv[3])
text="".join("\t".join(map(str,(key,row["qtype"],row["operator"],bundle/row["binary"],
 bundle/row["manifest"],row["parent_begin"],row["parent_end"],row["parent_count"],row["authority_count"])))+"\n"
 for key,row in sorted(doc["shards"].items(),key=lambda item:(item[1]["qtype"],item[1]["operator"],item[1]["parent_begin"])) )
if out.exists() and out.read_text()!=text: raise SystemExit("resumed shard execution index differs")
if not out.exists():
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}"); temporary.write_text(text); os.replace(temporary,out)
PY
  shard_count="$(wc -l <"$shard_index")"

  if [ "$phase" = screen ] || [ "$phase" = all ]; then
    while IFS=$'\t' read -r key q op binary manifest begin end count authority; do
      if [ "$op" = dense ]; then
        mapfile -t dense_shapes <"$out/inputs/q$q.dense.shapes"
        [ "${#dense_shapes[@]}" -gt 0 ] || { fail "q$q dense workload denominator is empty"; return $?; }
        dense_args=(); for workload in "${dense_shapes[@]}"; do dense_args+=("--shape=$workload"); done
        log="$out/results/$key.screen.log"
        if [ ! -s "$log" ]; then
          run_atomic "$log" "$binary" "${dense_args[@]}" \
            --iterations="$screen_iterations" --correctness-repeats="$repeats" \
            --bc-mode=skip || return $?
        fi
        [ "$(grep -c "^FQ_SHAPE_DONE .*typed_rows=$count selected_rows=$count .*status=PASS" "$log")" -eq "${#dense_shapes[@]}" ] || {
          fail "$key dense screen denominator incomplete"; return $?; }
      else
        while IFS=$'\t' read -r workload source_class tokens topk experts n k \
            profile rows_file total_rows max_rows rows_sha256; do
          log="$out/results/$key.$workload.screen.log"
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
          grep -q "^FQ_GROUPED_KPACK_SHARD .*total_rows=$total_rows max_rows=$max_rows .*workload=$workload router_profile=$profile .*" "$log" || {
            fail "$key grouped $workload fixture identity differs"; return $?; }
          grep -q "^FQ_GROUPED_KPACK_COMPLETE .*status=PASS rows=$count " "$log" || {
            fail "$key grouped $workload screen incomplete"; return $?; }
        done <"$out/inputs/q$q.grouped.tsv"
      fi
    done <"$shard_index"
  fi
  if [ "$phase" = screen ]; then
    printf '[fq-kpack-prebuilt] SCREEN_COMPLETE mode=%s shards=%s retention=ALL_RAW_BIT_CLEAN top_n=NONE artifacts=%s\n' \
      "$mode" "$shard_count" "$out"
    return 0
  fi

  while IFS=$'\t' read -r key q op binary manifest begin end count authority; do
    if [ "$op" = dense ]; then
      mapfile -t dense_shapes <"$out/inputs/q$q.dense.shapes"
      [ "${#dense_shapes[@]}" -gt 0 ] || { fail "q$q dense workload denominator is empty"; return $?; }
      dense_args=(); for workload in "${dense_shapes[@]}"; do dense_args+=("--shape=$workload"); done
      log="$out/results/$key.confirm.log"
      if [ ! -s "$log" ]; then
        run_atomic "$log" "$binary" "${dense_args[@]}" \
          --iterations="$confirm_iterations" --correctness-repeats="$repeats" \
          --bc-mode=skip || return $?
      fi
      [ "$(grep -c "^FQ_SHAPE_DONE .*typed_rows=$count selected_rows=$count .*status=PASS" "$log")" -eq "${#dense_shapes[@]}" ] || {
        fail "$key dense confirmation incomplete"; return $?; }
    else
      while IFS=$'\t' read -r workload source_class tokens topk experts n k \
          profile rows_file total_rows max_rows rows_sha256; do
        log="$out/results/$key.$workload.confirm.log"
        if [ ! -s "$log" ]; then
          grouped_args=(--experts="$experts" --n="$n" --k="$k"
            --workload-key="$workload" --router-profile="$profile")
          if [ "$rows_file" = - ]; then
            grouped_args+=(--tokens="$tokens" --topk="$topk")
          else
            grouped_args+=(--rows-file="$workloads/$rows_file")
          fi
          run_atomic "$log" "$binary" "${grouped_args[@]}" \
            --iterations="$confirm_iterations" --correctness-repeats="$repeats" || return $?
        fi
        grep -q "^FQ_GROUPED_KPACK_SHARD .*total_rows=$total_rows max_rows=$max_rows .*workload=$workload router_profile=$profile .*" "$log" || {
          fail "$key grouped $workload fixture identity differs"; return $?; }
        grep -q "^FQ_GROUPED_KPACK_COMPLETE .*status=PASS rows=$count " "$log" || {
          fail "$key grouped $workload confirmation incomplete"; return $?; }
      done <"$out/inputs/q$q.grouped.tsv"
    fi
  done <"$shard_index"
  printf '[fq-kpack-prebuilt] DIAGNOSTIC_COMPLETE mode=%s shards=%s top_n=NONE artifacts=%s\n' \
    "$mode" "$shard_count" "$out"
}

main "$@"
