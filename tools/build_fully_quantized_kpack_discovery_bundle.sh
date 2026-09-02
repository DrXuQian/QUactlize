#!/usr/bin/env bash
# Build parent-range FullyQuantized canonical K-pack discovery payloads locally.
# Every linked device binary owns at most 32 generated parents.  The PPU box
# runner consumes only these compact payloads and never compiles.

set -uo pipefail

fail() { printf '[fq-kpack-bundle] FAIL: %s\n' "$*" >&2; return 2; }
layout_for() { if [ "$1" = 12 ]; then printf '1\n'; else printf '2\n'; fi; }
format_for() {
  case "$1" in 10) printf '2\n';; 11) printf '3\n';; 12) printf '0\n';;
    13) printf '1\n';; 14) printf '4\n';; *) return 2;; esac
}
check_free_space() {
  local root="$1" required="$2" free
  free="$(df -Pk "$root" | awk 'NR==2 {print $4}')" || return 2
  [ -n "$free" ] && [ "$free" -ge "$required" ] || {
    fail "insufficient $root space before shard: free_kb=${free:-unknown} required_kb=$required"; return $?; }
}
ensure_owned_scratch() {
  local bundle="$1" preexisting="$2" scratch="$bundle/scratch"
  mkdir -p "$scratch" || return 2
  if [ ! -f "$scratch/.fq-kpack-owned-scratch" ]; then
    [ "$preexisting" = 0 ] || {
      fail "existing OUT scratch lacks ownership marker; refusing adoption"; return $?; }
    printf 'quactlize-fq-kpack-owned-v2\n' >"$scratch/.fq-kpack-owned-scratch" || return 2
  fi
  python3 -B - "$bundle" <<'PY'
import pathlib,sys
bundle=pathlib.Path(sys.argv[1]).resolve()
scratch=(bundle/"scratch").resolve()
if scratch.parent!=bundle or scratch.name!="scratch": raise SystemExit("scratch boundary differs")
if (scratch/".fq-kpack-owned-scratch").read_text()!="quactlize-fq-kpack-owned-v2\n":
 raise SystemExit("scratch ownership marker differs")
PY
}
clear_owned_shard_scratch() {
  local bundle="$1" target="$2"
  python3 -B - "$bundle" "$target" <<'PY'
import pathlib,re,shutil,sys
bundle=pathlib.Path(sys.argv[1]).resolve(); target=pathlib.Path(sys.argv[2])
scratch=(bundle/"scratch").resolve()
if (scratch/".fq-kpack-owned-scratch").read_text()!="quactlize-fq-kpack-owned-v2\n":
 raise SystemExit("scratch ownership marker differs")
if target.is_symlink(): raise SystemExit("shard scratch is a symlink")
if target.exists():
 resolved=target.resolve()
 if resolved.parent!=scratch or not re.fullmatch(r"q(?:10|11|12|13|14)-(?:dense|grouped)-p\d{5}-\d{5}",resolved.name):
  raise SystemExit("shard scratch escaped its owned parent")
 shutil.rmtree(resolved)
PY
}
shard_receipt() {
  local mode="$1" root="$2" out="$3" sdk="$4" authority="$5"
  local key="$6" q="$7" op="$8" manifest="$9" binary="${10}" receipt="${11}"
  python3 -B - "$mode" "$root" "$out" "$sdk" "$authority" "$key" \
    "$q" "$op" "$manifest" "$binary" "$receipt" <<'PY'
import hashlib,json,os,pathlib,re,subprocess,sys
mode,root,out,sdk,authority,key,q,op,manifest,binary,receipt=sys.argv[1:]
root,out,sdk,authority,manifest,binary,receipt=map(
 pathlib.Path,(root,out,sdk,authority,manifest,binary,receipt)); q=int(q)
sys.path.insert(0,str(root/"tools"))
import fully_quantized_kpack_bundle_index as index
import gen_fully_quantized_grouped_kpack_units as grouped
import gen_fully_quantized_kpack_discovery_units as dense
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not authority.is_file() or not manifest.is_file() or not binary.is_file():
 raise SystemExit("shard receipt input is incomplete")
parsed=json.loads(manifest.read_text())
(dense.validate_manifest if op=="dense" else grouped.validate_manifest)(parsed)
selection=parsed["parent_range"]
rows=parsed["dense_tc_parents"] if op=="dense" else parsed["grouped_parents"]
shard={"shard_key":key,"qtype":q,"operator":op,"route":"fully-quantized",
 "parent_begin":selection["begin"],"parent_end":selection["end"],
 "parent_count":selection["count"],"authority_count":selection["authority_count"],
 "parent_ids":[row["static_candidate_id"] for row in rows]}
if key!=index.shard_key(q,op,selection["begin"],selection["end"]):
 raise SystemExit("shard key/range differs")
binary_name=("test_fully_quantized_internal_sweep" if op=="dense" else
             "test_fully_quantized_grouped_kpack_discovery")
published_binary=out/"payloads"/key/binary_name
elf=subprocess.check_output([str(sdk/"bin/hgobjdump"),"-lelf",str(binary)],
 text=True,stderr=subprocess.STDOUT)
match=re.search(r"ELF FILE \d+ \((PPU [^)]+)\)",elf)
if not match: raise SystemExit("linked payload has no PPU image")
build=json.loads(authority.read_text())
doc={"schema":index.RECEIPT_SCHEMA,**shard,
 "build_input_authority_sha256":sha(authority),
 "source_sha":build["source_sha"],"source_tree":build["source_tree"],
 "submodules":build["submodules"],
 "sdk_compiler_sha256":build["sdk"]["compiler"]["sha256"],
 "sdk_inspector_sha256":build["sdk"]["inspector"]["sha256"],
 "manifest":str(manifest.relative_to(out)),"manifest_sha256":sha(manifest),
 "binary":str(published_binary.relative_to(out)),"binary_sha256":sha(binary),
 "device_arch":match.group(1),
 "inspector_output_sha256":hashlib.sha256(match.group(1).encode()).hexdigest()}
index.validate_receipt(doc,{**shard,"typed_rows":selection["count"]},
                       sha(manifest),sha(binary))
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if mode=="validate":
 if not receipt.is_file() or receipt.read_text()!=encoded:
  raise SystemExit("shard receipt differs")
elif mode=="write":
 if receipt.exists(): raise SystemExit("refusing to replace shard receipt")
 temporary=receipt.with_name(f".{receipt.name}.current.{os.getpid()}")
 temporary.write_text(encoded); os.replace(temporary,receipt)
else: raise SystemExit("unknown shard receipt mode")
PY
}
identity_probe_receipt() {
  local mode="$1" root="$2" sdk="$3" authority="$4" binary="$5" receipt="$6"
  python3 -B - "$mode" "$root" "$sdk" "$authority" "$binary" "$receipt" <<'PY'
import hashlib,json,os,pathlib,sys
mode,root,sdk,authority,binary,receipt=sys.argv[1:]
root,sdk,authority,binary,receipt=map(pathlib.Path,(root,sdk,authority,binary,receipt))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not authority.is_file() or not binary.is_file(): raise SystemExit("identity probe input missing")
doc={"schema":"quactlize.fq_kpack_identity_probe_receipt.v1",
 "build_input_authority_sha256":sha(authority),
 "source_sha256":sha(root/"tools/box_identity_probe.cpp"),
 "binary_sha256":sha(binary)}
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if mode=="validate":
 if not receipt.is_file() or receipt.read_text()!=encoded:
  raise SystemExit("identity probe receipt differs")
elif mode=="write":
 if receipt.exists(): raise SystemExit("refusing to replace identity probe receipt")
 temporary=receipt.with_name(f".{receipt.name}.current.{os.getpid()}")
 temporary.write_text(encoded); os.replace(temporary,receipt)
else: raise SystemExit("unknown identity probe receipt mode")
PY
}

main() {
  [ "$#" -eq 0 ] || { fail "no positional arguments are accepted"; return $?; }
  local root autodl_root out resume pilot jobs per_unit max_parents sdk sdk_receipt preexisting
  local min_free_kb dirty untracked build_authority global_preflight range_plan identity_bin identity_receipt
  local key q op begin end count total layout format gen payload binary receipt log rc
  local scratch_build built_binary build_resume stage staged_binary staged_receipt
  local partitioned partition_plan partition_id frozen_partition_plan
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  autodl_root="$(realpath -e /root/autodl-tmp)" || {
    fail "/root/autodl-tmp is required for large build outputs"; return $?; }
  out="$(realpath -m -- "${OUT:-$autodl_root/quactlize-fq-kpack-discovery-$(git -C "$root" rev-parse --short=8 HEAD)}")" || return 2
  case "$out" in "$autodl_root"/*) ;; *) fail "OUT must be a strict /root/autodl-tmp child"; return $?;; esac
  resume="${RESUME:-0}"; pilot="${PILOT:-0}"; jobs="${JOBS:-16}"
  per_unit="${FQ_KPACK_CONFIGS_PER_UNIT:-4}"
  max_parents="${FQ_KPACK_MAX_PARENTS_PER_BINARY:-32}"
  partition_plan="${KPACK_BUILD_PARTITION_PLAN:-}"
  partition_id="${KPACK_BUILD_PARTITION_ID:-}"
  partitioned=0
  if [ -n "$partition_plan" ] || [ -n "$partition_id" ]; then
    [ -n "$partition_plan" ] && [ -n "$partition_id" ] || {
      fail "KPACK_BUILD_PARTITION_PLAN and KPACK_BUILD_PARTITION_ID must be set together"; return $?; }
    [ -n "${OUT:-}" ] || {
      fail "partition builds require an explicit unique OUT"; return $?; }
    partition_plan="$(realpath -e -- "$partition_plan")" || return 2
    case "$partition_id" in ''|*[!0-9]*)
      fail "KPACK_BUILD_PARTITION_ID must be a nonnegative integer"; return $?;; esac
    partitioned=1
  fi
  case "$resume:$pilot" in 0:0|0:1|1:0|1:1) ;; *) fail "RESUME/PILOT must be 0 or 1"; return $?;; esac
  case "$jobs:$per_unit:$max_parents" in *[!0-9:]*|0:*|*:0:*|*:*:0)
    fail "JOBS/per-unit/max-parents must be positive integers"; return $?;; esac
  [ "$per_unit" -le 32 ] && [ "$max_parents" -le 32 ] || {
    fail "per-unit and max parents per binary must be <=32"; return $?; }
  if [ "$partitioned" = 1 ] && {
       [ "$pilot" != 0 ] || [ "$max_parents" != 32 ]; }; then
    fail "distributed partition builds require PILOT=0 and max-parents=32"; return $?
  fi
  preexisting=0
  if [ -e "$out" ]; then
    preexisting=1
    [ "$resume" = 1 ] && [ -d "$out" ] || { fail "refusing existing OUT; set RESUME=1"; return $?; }
  else
    [ "$resume" = 0 ] || { fail "RESUME=1 requires existing OUT"; return $?; }
    mkdir -p "$out" || return 2
  fi
  if [ "$preexisting" = 1 ] && [ ! -f "$out/scratch/.fq-kpack-owned-scratch" ]; then
    fail "existing OUT scratch lacks ownership marker; refusing adoption"
    return $?
  fi
  mkdir -p "$out"/{generated,payloads,results,tmp,scratch,inputs} || return 2
  ensure_owned_scratch "$out" "$preexisting" || return 2
  export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp" PYTHONDONTWRITEBYTECODE=1
  min_free_kb="${FQ_KPACK_MIN_FREE_KB:-8388608}"
  case "$min_free_kb" in ''|*[!0-9]*) fail "FQ_KPACK_MIN_FREE_KB must be an integer"; return $?;; esac
  check_free_space "$autodl_root" "$min_free_kb" || return 2
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail "ambient PPU_DEFS/PPU_EXTRA_DEFS changes the denominator"; return $?
  fi
  dirty="$(git -C "$root" status --porcelain --untracked-files=no | \
    awk 'substr($0,4)!=".coord/BOX.md" && substr($0,4)!=".coord/INBOX.md"')" || return 2
  [ -z "$dirty" ] || { printf '%s\n' "$dirty" >&2; fail "tracked build source is dirty"; return $?; }
  untracked="$(git -C "$root" ls-files --others --exclude-standard | awk '
    /^(benchmarks|ci|cmake|dev|quactlize|tools|third_party)\// ||
    $0 == "CMakeLists.txt" || $0 == "build.sh" {print}')" || return 2
  [ -z "$untracked" ] || {
    printf '%s\n' "$untracked" >&2; fail "untracked build source is not in the clean commit"; return $?; }
  git -C "$root" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain)"' || { fail "recursive submodule is dirty"; return $?; }
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" || {
    fail "set PPU_SDK to the exact build SDK"; return $?; }
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || {
    fail "SDK lacks hgcc/hgobjdump"; return $?; }
  if [ -f "$sdk/release.yaml" ]; then sdk_receipt="$sdk/release.yaml";
  elif [ -f "$sdk/VERSION.txt" ]; then sdk_receipt="$sdk/VERSION.txt";
  else fail "SDK lacks release.yaml/VERSION.txt"; return $?; fi

  global_preflight="$out/inputs/kpack-global-preflight.json"
  if [ "$preexisting" = 1 ]; then
    python3 -B "$root/tools/kpack_global_build_preflight.py" verify \
      --root "$root" --receipt "$global_preflight" || return 2
  elif [ -n "${KPACK_GLOBAL_PREFLIGHT_RECEIPT:-}" ]; then
    local shared_global_preflight
    shared_global_preflight="$(realpath -e -- "$KPACK_GLOBAL_PREFLIGHT_RECEIPT")" || return 2
    [ -f "$shared_global_preflight" ] && [ ! -L "$shared_global_preflight" ] || {
      fail "KPACK_GLOBAL_PREFLIGHT_RECEIPT must name one regular non-symlink file"; return $?; }
    python3 -B "$root/tools/kpack_global_build_preflight.py" verify \
      --root "$root" --receipt "$shared_global_preflight" || return 2
    install -m 0444 "$shared_global_preflight" "$global_preflight" || return 2
    cmp -s "$shared_global_preflight" "$global_preflight" || {
      fail "copied global preflight receipt differs"; return $?; }
  else
    python3 -B "$root/tools/kpack_global_build_preflight.py" create \
      --root "$root" --output "$global_preflight" || return 2
  fi

  if [ "$partitioned" = 1 ]; then
    frozen_partition_plan="$out/inputs/build-partition-plan.json"
    range_plan="$out/inputs/selected-shards.tsv"
    python3 -B "$root/tools/kpack_discovery_build_partitions.py" select \
      --plan "$partition_plan" --partition "$partition_id" \
      --route fully-quantized --freeze-plan "$frozen_partition_plan" \
      --output "$range_plan" || return 2
  else
    frozen_partition_plan="-"
  fi

  build_authority="$out/inputs/build-input-authority.json"
  python3 -B - "$root" "$sdk" "$sdk_receipt" "$build_authority" \
    "$pilot" "$per_unit" "$max_parents" "$min_free_kb" "$preexisting" \
    "$frozen_partition_plan" "$partition_id" "$global_preflight" <<'PY' || return 2
import hashlib,json,os,pathlib,shlex,shutil,subprocess,sys
root,sdk,receipt,out=map(pathlib.Path,sys.argv[1:5])
pilot,per_unit,max_parents,min_free,preexisting=map(int,sys.argv[5:10])
partition_plan_arg=sys.argv[10]; partition_id_arg=sys.argv[11]
global_preflight=pathlib.Path(sys.argv[12])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sdk_file(path):
 return {"path":str(path.relative_to(sdk)),"size":path.stat().st_size,
         "sha256":sha(path),"symlink_target":os.readlink(path) if path.is_symlink() else None}
subs=[]
for line in subprocess.check_output(["git","-C",str(root),"submodule","status","--recursive"],text=True).splitlines():
 if not line or line[0]!=" ": raise SystemExit("submodule absent/dirty/conflicted")
 fields=line[1:].split(); checkout=root/fields[1]
 current=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
 if current!=fields[0] or subprocess.check_output(["git","-C",str(checkout),"status","--porcelain"],text=True).strip():
  raise SystemExit(f"submodule identity differs: {fields[1]}")
 subs.append({"path":fields[1],"gitlink":fields[0],"current":current})
runtime=[sdk_file(p) for p in sorted(sdk.rglob("libhggc*.so*")) if p.is_file() or p.is_symlink()]
if not runtime: raise SystemExit("SDK exposes no libhggc runtime")
cxx_words=shlex.split(os.environ.get("CXX","c++"))
if not cxx_words: raise SystemExit("CXX expands to an empty command")
cxx_path=shutil.which(cxx_words[0])
if cxx_path is None: raise SystemExit("host C++ compiler is unavailable")
cxx_path=pathlib.Path(cxx_path).resolve()
doc={"schema":"quactlize.fully_quantized_kpack_build_input.v2",
 "source_sha":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
 "source_tree":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD^{tree}"],text=True).strip(),
 "submodules":subs,
 "sdk":{"receipt":sdk_file(receipt),"compiler":sdk_file(sdk/"bin/hgcc"),
        "inspector":sdk_file(sdk/"bin/hgobjdump"),"runtime_libraries":runtime},
 "host_cxx":{"command":cxx_words,"resolved_path":str(cxx_path),
             "sha256":sha(cxx_path)},
 "global_preflight":{"path":"inputs/kpack-global-preflight.json",
                     "sha256":sha(global_preflight),
                     "schema":json.loads(global_preflight.read_text()).get("schema")},
 "configuration":{"mode":"PILOT" if pilot else "FULL","ppu_arch":"ppu0010",
   "configs_per_unit":per_unit,"max_parents_per_binary":max_parents,
   "preserve_stale_build_trees":True,
   "scratch_policy":"ONE_PARENT_RANGE_THEN_COMPACT_PAYLOAD"},
 "minimum_free_kib":min_free}
if partition_plan_arg != "-":
 sys.path.insert(0,str(root/"tools"))
 import kpack_discovery_build_partitions as partitions
 partition_path=pathlib.Path(partition_plan_arg)
 partition_doc=partitions.read_plan(partition_path)
 doc["configuration"]["distributed_partition"] = partitions.authority_partition_record(
     partition_path,partition_doc,int(partition_id_arg),"fully-quantized")
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if preexisting:
 if not out.is_file(): raise SystemExit("existing OUT lacks build input authority")
 if out.read_text()!=encoded: raise SystemExit("resumable build input authority differs")
else:
 if out.exists(): raise SystemExit("fresh OUT unexpectedly contains build input authority")
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}"); temporary.write_text(encoded); os.replace(temporary,out)
PY

  if [ "$partitioned" = 0 ]; then
    range_plan="$out/inputs/shard-plan.tsv"
    python3 -B - "$root" "$pilot" "$max_parents" "$range_plan" <<'PY' || return 2
import os,pathlib,sys
root=pathlib.Path(sys.argv[1]); pilot=bool(int(sys.argv[2])); maximum=int(sys.argv[3]); out=pathlib.Path(sys.argv[4])
sys.path.insert(0,str(root/"tools")); import fully_quantized_kpack_bundle_index as index
text="".join("\t".join(map(str,(r["shard_key"],r["qtype"],r["operator"],r["parent_begin"],r["parent_end"],r["parent_count"],r["authority_count"])))+"\n" for r in index.plan(pilot,maximum))
if out.exists() and out.read_text()!=text: raise SystemExit("resumable shard plan differs")
if not out.exists():
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}"); temporary.write_text(text); os.replace(temporary,out)
PY
  fi
  python3 -B "$root/tools/fully_quantized_kpack_bundle_index.py" || return 2
  python3 -B "$root/ci/check_fully_quantized_kpack_discovery.py" || return 2

  identity_bin="$out/payloads/box_identity_probe"
  identity_receipt="$out/payloads/box_identity_probe.receipt.json"
  if [ -e "$identity_bin" ] || [ -e "$identity_receipt" ]; then
    identity_probe_receipt validate "$root" "$sdk" "$build_authority" \
      "$identity_bin" "$identity_receipt" || { fail "stale identity probe payload"; return $?; }
  else
    python3 -B - "$root" "$sdk" "$identity_bin" <<'PY' || return 2
import os,pathlib,subprocess,sys
root,sdk,out=map(pathlib.Path,sys.argv[1:]); sys.path.insert(0,str(root/"tools"))
import probe_box_identity as probe
temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
command=probe._probe_compile_command(sdk,temporary,dict(os.environ))
run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
if run.returncode or not temporary.is_file(): raise SystemExit("identity probe build failed:\n"+run.stdout[-4000:])
temporary.chmod(0o755); os.replace(temporary,out)
PY
    identity_probe_receipt write "$root" "$sdk" "$build_authority" \
      "$identity_bin" "$identity_receipt" || return 2
  fi

  while IFS=$'\t' read -r key q op begin end count total; do
    layout="$(layout_for "$q")" || return 2; format="$(format_for "$q")" || return 2
    gen="$out/generated/$key"; payload="$out/payloads/$key"
    if [ "$op" = dense ]; then binary="$payload/test_fully_quantized_internal_sweep";
    else binary="$payload/test_fully_quantized_grouped_kpack_discovery"; fi
    receipt="$payload/binary-receipt.json"
    if [ -e "$payload" ]; then
      shard_receipt validate "$root" "$out" "$sdk" "$build_authority" \
        "$key" "$q" "$op" "$gen/manifest.json" "$binary" "$receipt" || {
          fail "$key partial/stale resume state"; return $?; }
      clear_owned_shard_scratch "$out" "$out/scratch/$key" || return 2
      continue
    fi
    if [ -e "$gen" ]; then
      python3 -B - "$root" "$gen/manifest.json" "$key" "$q" "$op" \
        "$begin" "$end" "$count" "$total" <<'PY' || return 2
import json,pathlib,sys
root,manifest,key,q,op,begin,end,count,total=sys.argv[1:]
root=pathlib.Path(root); manifest=pathlib.Path(manifest); q=int(q)
begin,end,count,total=map(int,(begin,end,count,total)); sys.path.insert(0,str(root/"tools"))
import fully_quantized_kpack_bundle_index as index
import gen_fully_quantized_grouped_kpack_units as grouped
import gen_fully_quantized_kpack_discovery_units as dense
if not manifest.is_file(): raise SystemExit("partial generated shard lacks manifest")
doc=json.loads(manifest.read_text()); (dense.validate_manifest if op=="dense" else grouped.validate_manifest)(doc)
rows=doc["dense_tc_parents"] if op=="dense" else doc["grouped_parents"]
wanted={"begin":begin,"end":end,"count":count,"authority_count":total}
if doc["parent_range"]!=wanted or key!=index.shard_key(q,op,begin,end) or \
   [row["static_candidate_id"] for row in rows]!=index.authority_parent_ids(q,op)[begin:end]:
 raise SystemExit("generated shard range authority differs")
PY
    elif [ "$op" = dense ]; then
      python3 -B "$root/tools/gen_fully_quantized_kpack_discovery_units.py" \
        --qtype "$q" --per-unit "$per_unit" --parent-begin "$begin" \
        --parent-count "$count" --out-dir "$gen" || return 2
    else
      python3 -B "$root/tools/gen_fully_quantized_grouped_kpack_units.py" \
        --qtype "$q" --per-unit "$per_unit" --parent-begin "$begin" \
        --parent-count "$count" --out-dir "$gen" || return 2
    fi
    check_free_space "$autodl_root" "$min_free_kb" || return 2
    scratch_build="$out/scratch/$key"; build_resume=0
    if [ -e "$scratch_build" ]; then
      if [ -d "$scratch_build" ] && [ ! -L "$scratch_build" ] && \
         [ -f "$scratch_build/CMakeCache.txt" ] && \
         [ -f "$scratch_build/.quactlize-source-head" ]; then
        build_resume=1
      else
        fail "$key scratch is partial/unversioned; refusing unsafe resume"
        return $?
      fi
    else
      mkdir -p "$scratch_build" || return 2
    fi
    log="$out/results/$key.build.log"
    printf '[fq-kpack-bundle] build shard=%s parents=[%s,%s)\n' "$key" "$begin" "$end"
    if [ "$op" = dense ]; then
      (cd "$root" && PPU_BUILD_DIR="$scratch_build" PPU_BUILD_RESUME="$build_resume" PPU_ARCHS=ppu0010 \
        PPU_SDK="$sdk" PPU_PRESERVE_STALE_BUILD_TREES=1 JOBS="$jobs" \
        QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$global_preflight" \
        TARGET=test_fully_quantized_internal_sweep \
        FQ_SWEEP_GENERATED_DIR="$gen" FQ_SWEEP_QTYPE="$q" \
        FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
        FQ_SWEEP_PACKED_FORMAT="$format" FQ_SWEEP_WEIGHT_LAYOUT="$layout" \
        ./build.sh) >"$log" 2>&1
      rc=$?; built_binary="$scratch_build/ppu_targets/test_fully_quantized_internal_sweep"
    else
      (cd "$root" && PPU_BUILD_DIR="$scratch_build" PPU_BUILD_RESUME="$build_resume" PPU_ARCHS=ppu0010 \
        PPU_SDK="$sdk" PPU_PRESERVE_STALE_BUILD_TREES=1 JOBS="$jobs" \
        QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$global_preflight" \
        TARGET=test_fully_quantized_grouped_kpack_discovery \
        FQ_GROUPED_KPACK_GENERATED_DIR="$gen" FQ_GROUPED_KPACK_QTYPE="$q" \
        FQ_GROUPED_KPACK_WEIGHT_LAYOUT="$layout" \
        FQ_GROUPED_KPACK_PACKED_FORMAT="$format" ./build.sh) >"$log" 2>&1
      rc=$?; built_binary="$scratch_build/ppu_targets/test_fully_quantized_grouped_kpack_discovery"
    fi
    if [ "$rc" -ne 0 ]; then tail -160 "$log" >&2; return "$rc"; fi
    [ -x "$built_binary" ] || { fail "$key linked binary is missing"; return $?; }
    stage="$out/payloads/.$key.current.$$"
    [ ! -e "$stage" ] || { fail "$key payload staging path already exists"; return $?; }
    mkdir -p "$stage" || return 2
    staged_binary="$stage/$(basename "$binary")"
    staged_receipt="$stage/binary-receipt.json"
    install -m 0755 "$built_binary" "$staged_binary" || return 2
    shard_receipt write "$root" "$out" "$sdk" "$build_authority" \
      "$key" "$q" "$op" "$gen/manifest.json" \
      "$staged_binary" "$staged_receipt" || return 2
    shard_receipt validate "$root" "$out" "$sdk" "$build_authority" \
      "$key" "$q" "$op" "$gen/manifest.json" \
      "$staged_binary" "$staged_receipt" || return 2
    [ ! -e "$payload" ] || { fail "$key payload appeared during atomic publication"; return $?; }
    mv "$stage" "$payload" || return 2
    shard_receipt validate "$root" "$out" "$sdk" "$build_authority" \
      "$key" "$q" "$op" "$gen/manifest.json" "$binary" "$receipt" || return 2
    clear_owned_shard_scratch "$out" "$scratch_build" || return 2
  done <"$range_plan"

  if [ "$partitioned" = 1 ]; then
    PPU_SDK="$sdk" python3 -B \
      "$root/tools/kpack_discovery_build_partitions.py" record \
      --root "$out" --route fully-quantized || return 2
    printf '[fq-kpack-bundle] PARTITION_COMPLETE id=%s shards=%s bundle=%s\n' \
      "$partition_id" "$(wc -l < "$range_plan")" "$out"
    return 0
  fi

  python3 -B - "$root" "$out" "$sdk" "$sdk_receipt" "$build_authority" \
    "$pilot" "$max_parents" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
root,out,sdk,sdk_receipt,authority=map(pathlib.Path,sys.argv[1:6]); pilot=bool(int(sys.argv[6])); maximum=int(sys.argv[7])
sys.path.insert(0,str(root/"tools"))
import fully_quantized_kpack_bundle_index as index
import gen_fully_quantized_grouped_kpack_units as grouped
import gen_fully_quantized_kpack_discovery_units as dense
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads(authority.read_text()); shards={}
for planned in index.plan(pilot,maximum):
 key=planned["shard_key"]; op=planned["operator"]
 manifest=out/f"generated/{key}/manifest.json"
 parsed=json.loads(manifest.read_text()); (dense.validate_manifest if op=="dense" else grouped.validate_manifest)(parsed)
 rows=parsed["dense_tc_parents"] if op=="dense" else parsed["grouped_parents"]
 binary_name="test_fully_quantized_internal_sweep" if op=="dense" else "test_fully_quantized_grouped_kpack_discovery"
 binary=out/f"payloads/{key}/{binary_name}"; receipt=out/f"payloads/{key}/binary-receipt.json"
 receipt_doc=json.loads(receipt.read_text())
 index.validate_receipt(receipt_doc,{**planned,"typed_rows":len(rows)},sha(manifest),sha(binary))
 if [row["static_candidate_id"] for row in rows]!=planned["parent_ids"]:
  raise SystemExit(f"{key}: manifest parent ids differ")
 shards[key]={**planned,"typed_rows":len(rows),
  "manifest":str(manifest.relative_to(out)),"manifest_sha256":sha(manifest),
  "binary":str(binary.relative_to(out)),"binary_sha256":sha(binary),
  "binary_receipt":str(receipt.relative_to(out)),"binary_receipt_sha256":sha(receipt),
  "device_arch":receipt_doc["device_arch"],
  "inspector_output_sha256":receipt_doc["inspector_output_sha256"]}
probe=out/"payloads/box_identity_probe"; probe_receipt=out/"payloads/box_identity_probe.receipt.json"
doc={"schema":index.BUNDLE_SCHEMA,"mode":"PILOT" if pilot else "FULL",
 "max_parents_per_binary":maximum,"source_sha":build["source_sha"],
 "source_tree":build["source_tree"],"submodules":build["submodules"],
 "sdk":build["sdk"],"build_input_authority":str(authority.relative_to(out)),
 "build_input_authority_sha256":sha(authority),
 "runtime_identity_probe":{"path":str(probe.relative_to(out)),"sha256":sha(probe),
  "receipt":str(probe_receipt.relative_to(out)),"receipt_sha256":sha(probe_receipt)},
 "shards":shards}
index.validate_index(doc)
path=out/"bundle.json"; encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if path.exists() and path.read_text()!=encoded: raise SystemExit("resumed bundle authority changed")
if not path.exists():
 temporary=path.with_name(f".{path.name}.current.{os.getpid()}"); temporary.write_text(encoded); os.replace(temporary,path)
PY
  printf '[fq-kpack-bundle] COMPLETE mode=%s shards=%s bundle=%s\n' \
    "$([ "$pilot" = 1 ] && printf PILOT || printf FULL)" \
    "$(wc -l <"$range_plan")" "$out"
}

main "$@"
