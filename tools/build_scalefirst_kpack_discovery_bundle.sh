#!/usr/bin/env bash
# Compile the complete canonical K-pack ScaleFirst discovery bundle locally.
# The resulting directory is immutable input to the prebuilt-only box runner.

set -uo pipefail

fail() { printf '[sf-kpack-bundle] FAIL: %s\n' "$*" >&2; return 2; }
layout_for() { if [ "$1" = 12 ]; then printf '1\n'; else printf '2\n'; fi; }
binary_receipt() {
  python3 -B - "$@" <<'PY'
import hashlib,json,os,pathlib,sys
mode=sys.argv[1]
authority,manifest,binary,receipt=map(pathlib.Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not authority.is_file() or not manifest.is_file() or not binary.is_file():
 raise SystemExit("binary receipt input is missing")
doc={"schema":"quactlize.scalefirst_kpack_binary_receipt.v1",
     "build_input_authority_sha256":sha(authority),
     "manifest_sha256":sha(manifest),"binary_sha256":sha(binary)}
if mode=="verify":
 if not receipt.is_file(): raise SystemExit("resumable binary receipt is missing")
 if json.loads(receipt.read_text())!=doc: raise SystemExit("resumable binary receipt differs")
elif mode=="record":
 encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
 if receipt.exists() and receipt.read_text()!=encoded:
  raise SystemExit("binary receipt changed")
 if not receipt.exists():
  temporary=receipt.with_name(f".{receipt.name}.current.{os.getpid()}")
  temporary.write_text(encoded); os.replace(temporary,receipt)
else:
 raise SystemExit("binary receipt mode must be record or verify")
PY
}
clear_owned_scratch() {
  python3 -B - "$1" "$2" <<'PY'
import pathlib,shutil,sys
parent=pathlib.Path(sys.argv[1]).resolve(strict=True)
target=pathlib.Path(sys.argv[2])
if target.is_symlink(): raise SystemExit("scratch target is a symlink")
resolved=target.resolve(strict=True)
if resolved.parent!=parent or resolved==parent or not resolved.is_dir():
 raise SystemExit("scratch target escaped its owned parent")
shutil.rmtree(resolved)
PY
}
require_shard_space() {
  local out="$1" minimum="$2" label="$3" available
  available="$(df -Pk "$out" | awk 'NR==2 {print $4}')" || return 2
  case "$available" in ''|*[!0-9]*) fail "cannot measure $label free space"; return $?;; esac
  [ "$available" -ge "$minimum" ] || {
    fail "$label free space ${available}KiB fell below one-shard preflight ${minimum}KiB";
    return $?;
  }
}
validate_owned_path() {
  python3 -B - "$1" "$2" "$3" <<'PY'
import pathlib,sys
parent=pathlib.Path(sys.argv[1]).resolve(strict=True)
target=pathlib.Path(sys.argv[2]); kind=sys.argv[3]
if target.is_symlink(): raise SystemExit(f"{kind} target is a symlink")
if target.parent.resolve(strict=True)!=parent or target.parent==target:
 raise SystemExit(f"{kind} target escaped its owned parent")
if target.exists():
 resolved=target.resolve(strict=True)
 if resolved.parent!=parent or not resolved.is_dir():
  raise SystemExit(f"{kind} target is not one owned directory")
PY
}
build_shard() {
  local root="$1" out="$2" sdk="$3" input_authority="$4"
  local global_preflight="$5" jobs="$6" per_unit="$7" min_free_kib="$8"
  local shard_id="$9" q="${10}" operator="${11}" layout="${12}"
  local begin="${13}" end="${14}"
  local generated manifest payload scratch binary_name target binary receipt
  local built stage log build_resume rc count available
  count=$((end - begin))
  [ "$count" -gt 0 ] && [ "$count" -le 32 ] || {
    fail "$shard_id parent count is outside [1,32]"; return $?; }
  generated="$out/generated/$shard_id"
  manifest="$generated/manifest.json"
  if [ "$operator" = dense ]; then
    python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
      --qtype "$q" --artifact-tk 0 --bchunk 0 --weight-layout "$layout" \
      --parent-begin "$begin" --parent-count "$count" \
      --per-unit "$per_unit" --out-dir "$generated" || return 2
    binary_name=test_scalefirst_internal_sweep
    target=$binary_name
  elif [ "$operator" = grouped ]; then
    python3 -B "$root/tools/gen_scalefirst_grouped_kpack_units.py" \
      --qtype "$q" --parent-begin "$begin" --parent-count "$count" \
      --per-unit "$per_unit" --out-dir "$generated" || return 2
    binary_name=test_scalefirst_grouped_kpack_discovery
    target=$binary_name
  else
    fail "$shard_id has unsupported operator $operator"; return $?
  fi
  payload="$out/payloads/$shard_id"
  validate_owned_path "$out/payloads" "$payload" "payload" || return 2
  binary="$payload/$binary_name"
  receipt="$payload/binary-receipt.json"
  if [ -x "$binary" ]; then
    binary_receipt verify "$input_authority" "$manifest" \
      "$binary" "$receipt" || return 2
    return 0
  fi
  [ ! -e "$payload" ] || {
    fail "$shard_id payload exists without one complete executable"; return $?; }
  require_shard_space "$out" "$min_free_kib" "$shard_id" || return $?
  scratch="$out/scratch/$shard_id"
  validate_owned_path "$out/scratch" "$scratch" "scratch" || return 2
  log="$out/results/$shard_id.build.log"
  build_resume=0
  if [ -e "$scratch" ]; then
    if [ -f "$scratch/CMakeCache.txt" ] &&
       [ -f "$scratch/.quactlize-source-head" ]; then
      build_resume=1
    else
      fail "$shard_id partial scratch lacks exact resume markers"; return $?
    fi
  fi
  printf '[sf-kpack-bundle] build shard=%s q=%s operator=%s range=[%s,%s)\n' \
    "$shard_id" "$q" "$operator" "$begin" "$end"
  if [ "$operator" = dense ]; then
    (cd "$root" && PPU_SDK="$sdk" PPU_BUILD_DIR="$scratch" \
      PPU_BUILD_RESUME="$build_resume" PPU_PRESERVE_STALE_BUILD_TREES=1 \
      QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$global_preflight" \
      PPU_ARCHS=ppu0010 JOBS="$jobs" \
      TARGET="$target" SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
      SCALEFIRST_SWEEP_QTYPE="$q" SCALEFIRST_SWEEP_ARTIFACT_TK=0 \
      SCALEFIRST_SWEEP_BCHUNK=0 SCALEFIRST_SWEEP_WEIGHT_LAYOUT="$layout" \
      ./build.sh) >"$log" 2>&1
  else
    (cd "$root" && PPU_SDK="$sdk" PPU_BUILD_DIR="$scratch" \
      PPU_BUILD_RESUME="$build_resume" PPU_PRESERVE_STALE_BUILD_TREES=1 \
      QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$global_preflight" \
      PPU_ARCHS=ppu0010 JOBS="$jobs" \
      TARGET="$target" SCALEFIRST_GROUPED_KPACK_GENERATED_DIR="$generated" \
      SCALEFIRST_GROUPED_KPACK_QTYPE="$q" \
      SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT="$layout" ./build.sh) \
      >"$log" 2>&1
  fi
  rc=$?; if [ "$rc" -ne 0 ]; then tail -160 "$log" >&2; return "$rc"; fi
  built="$scratch/ppu_targets/$binary_name"
  [ -x "$built" ] || { fail "$shard_id linked binary is missing"; return $?; }
  stage="$out/payloads/.$shard_id.current.$$"
  validate_owned_path "$out/payloads" "$stage" "payload staging" || return 2
  [ ! -e "$stage" ] || { fail "$shard_id payload staging path exists"; return $?; }
  mkdir -p "$stage" || return 2
  cp --preserve=mode,timestamps -- "$built" "$stage/$binary_name" || return 2
  cmp -s -- "$built" "$stage/$binary_name" || {
    fail "$shard_id payload copy differs"; return $?; }
  binary_receipt record "$input_authority" "$manifest" \
    "$stage/$binary_name" "$stage/binary-receipt.json" || return 2
  mv -- "$stage" "$payload" || return 2
  binary_receipt verify "$input_authority" "$manifest" \
    "$binary" "$receipt" || return 2
  clear_owned_scratch "$out/scratch" "$scratch" || return 2
}
build_identity_probe() {
  local root="$1" out="$2" sdk="$3" host_cxx="$4" input_authority="$5"
  local payload="$out/payloads/support" binary receipt stage
  validate_owned_path "$out/payloads" "$payload" \
    "identity probe payload" || return 2
  binary="$payload/box_identity_probe"
  receipt="$payload/identity-probe-receipt.json"
  if [ -x "$binary" ]; then
    python3 -B - verify "$input_authority" \
      "$root/tools/box_identity_probe.cpp" "$binary" "$receipt" <<'PY' || return 2
import hashlib,json,pathlib,sys
mode=sys.argv[1]; authority,source,binary,receipt=map(pathlib.Path,sys.argv[2:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
expected={"schema":"quactlize.scalefirst_kpack_identity_probe_receipt.v1",
 "build_input_authority_sha256":sha(authority),"source_sha256":sha(source),
 "binary_sha256":sha(binary)}
if mode!="verify" or not receipt.is_file() or json.loads(receipt.read_text())!=expected:
 raise SystemExit("resumable identity probe receipt differs")
PY
    return 0
  fi
  [ ! -e "$payload" ] || {
    fail "identity probe payload exists without a complete executable"; return $?; }
  stage="$out/payloads/.support.current.$$"
  validate_owned_path "$out/payloads" "$stage" \
    "identity probe staging" || return 2
  [ ! -e "$stage" ] || { fail "identity probe staging path exists"; return $?; }
  mkdir -p "$stage" || return 2
  CXX="$host_cxx" python3 -B - "$root" "$sdk" \
    "$stage/box_identity_probe" <<'PY' || return 2
import os,pathlib,subprocess,sys
root,sdk,output=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(root/"tools"))
import probe_box_identity
command=probe_box_identity._probe_compile_command(sdk,output,dict(os.environ))
completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,check=False)
if completed.returncode:
 raise SystemExit("identity probe compile failed:\n"+completed.stdout[-4000:])
output.chmod(output.stat().st_mode | 0o111)
PY
  python3 -B - record "$input_authority" "$root/tools/box_identity_probe.cpp" \
    "$stage/box_identity_probe" "$stage/identity-probe-receipt.json" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
mode=sys.argv[1]; authority,source,binary,receipt=map(pathlib.Path,sys.argv[2:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if mode!="record" or not binary.is_file(): raise SystemExit("identity probe was not produced")
doc={"schema":"quactlize.scalefirst_kpack_identity_probe_receipt.v1",
 "build_input_authority_sha256":sha(authority),"source_sha256":sha(source),
 "binary_sha256":sha(binary)}
temporary=receipt.with_name(f".{receipt.name}.current.{os.getpid()}")
temporary.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
os.replace(temporary,receipt)
PY
  mv -- "$stage" "$payload" || return 2
  build_identity_probe "$root" "$out" "$sdk" "$host_cxx" "$input_authority"
}

main() {
  if [ "$#" -ne 0 ]; then fail "no positional arguments are accepted"; return $?; fi
  local root out out_root requested_out_root resume jobs per_unit sdk receipt host_cxx dirty untracked
  local preexisting input_authority global_preflight pilot scope parents_per_binary plan plan_current
  local shard_rows shard_id q operator layout begin end
  local partitioned partition_plan partition_id frozen_partition_plan
  local min_free_gib min_free_kib available_kib
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  requested_out_root="${KPACK_LOCAL_SCRATCH_ROOT:-/root/autodl-tmp}"
  [ -d "$requested_out_root" ] && [ ! -L "$requested_out_root" ] || {
    fail "KPACK_LOCAL_SCRATCH_ROOT must be a regular non-symlink directory"; return $?; }
  out_root="$(realpath -e -- "$requested_out_root")" || {
    fail "KPACK_LOCAL_SCRATCH_ROOT is not a readable directory"; return $?; }
  case "$out_root" in /|/root|/workspace)
    fail "KPACK_LOCAL_SCRATCH_ROOT is too broad"; return $?;; esac
  pilot="${PILOT:-0}"
  case "$pilot" in 0) scope=full ;; 1) scope=pilot ;; *)
    fail "PILOT must be 0 or 1"; return $?;; esac
  out="$(realpath -m -- "${OUT:-$out_root/quactlize-sf-kpack-$scope-$(git -C "$root" rev-parse --short=8 HEAD)}")" || return 2
  case "$out" in "$out_root"/*) ;; *)
    fail "OUT must be a strict configured scratch child"; return $?;; esac
  resume="${RESUME:-0}"; jobs="${JOBS:-16}"
  per_unit="${SCALEFIRST_KPACK_CONFIGS_PER_UNIT:-4}"
  parents_per_binary="${SCALEFIRST_KPACK_PARENTS_PER_BINARY:-32}"
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
  case "$resume" in 0|1) ;; *) fail "RESUME must be 0 or 1"; return $?;; esac
  case "$jobs:$per_unit:$parents_per_binary" in *[!0-9:]*|0:*|*:0:*|*:*:0)
    fail "JOBS/per-unit/parents-per-binary must be positive"; return $?;; esac
  [ "$parents_per_binary" -le 32 ] || {
    fail "SCALEFIRST_KPACK_PARENTS_PER_BINARY may not exceed 32"; return $?; }
  if [ "$partitioned" = 1 ] && {
       [ "$pilot" != 0 ] || [ "$parents_per_binary" != 32 ]; }; then
    fail "distributed partition builds require PILOT=0 and parents-per-binary=32"; return $?
  fi
  preexisting=0
  if [ -e "$out" ]; then
    preexisting=1
    if [ "$resume" != 1 ] || [ ! -d "$out" ]; then
      fail "refusing existing OUT; set RESUME=1"; return $?
    fi
  else
    mkdir -p "$out" || return 2
  fi
  mkdir -p "$out"/{generated,payloads,results,scratch,tmp,inputs} || return 2
  export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"
  export PYTHONDONTWRITEBYTECODE=1
  min_free_gib="${SCALEFIRST_KPACK_MIN_FREE_GIB:-8}"
  case "$min_free_gib" in ''|*[!0-9]*|0)
    fail "SCALEFIRST_KPACK_MIN_FREE_GIB must be a positive integer"; return $?;; esac
  min_free_kib=$((min_free_gib * 1024 * 1024))
  available_kib="$(df -Pk "$out" | awk 'NR==2 {print $4}')" || return 2
  case "$available_kib" in ''|*[!0-9]*) fail "cannot measure OUT free space"; return $?;; esac
  [ "$available_kib" -ge "$min_free_kib" ] || {
    fail "OUT free space ${available_kib}KiB is below one-shard preflight ${min_free_kib}KiB";
    return $?;
  }
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
  host_cxx="$(command -v c++ 2>/dev/null)" || {
    fail "host c++ compiler is unavailable"; return $?; }
  host_cxx="$(realpath -e -- "$host_cxx")" || return 2
  if [ -f "$sdk/release.yaml" ]; then receipt="$sdk/release.yaml";
  elif [ -f "$sdk/VERSION.txt" ]; then receipt="$sdk/VERSION.txt";
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

  python3 -B "$root/tools/scalefirst_kpack_binary_shards.py" self-test || return 2
  plan="$out/shard-plan.json"
  plan_current="$out/.shard-plan.current.$$"
  python3 -B "$root/tools/scalefirst_kpack_binary_shards.py" emit \
    --scope "$scope" --parents-per-binary "$parents_per_binary" \
    --out "$plan_current" || return 2
  python3 -B - "$plan_current" "$plan" "$preexisting" <<'PY' || return 2
import os,pathlib,sys
current,path=map(pathlib.Path,sys.argv[1:3]); preexisting=int(sys.argv[3])
if preexisting:
 if not path.is_file(): raise SystemExit("resumable shard plan is missing")
 if current.read_bytes()!=path.read_bytes(): raise SystemExit("resumable shard plan differs")
 current.unlink()
else:
 os.replace(current,path)
PY

  if [ "$partitioned" = 1 ]; then
    frozen_partition_plan="$out/inputs/build-partition-plan.json"
    shard_rows="$out/selected-shards.tsv"
    python3 -B "$root/tools/kpack_discovery_build_partitions.py" select \
      --plan "$partition_plan" --partition "$partition_id" \
      --route scalefirst --freeze-plan "$frozen_partition_plan" \
      --output "$shard_rows" || return 2
  else
    frozen_partition_plan="-"
  fi

  # This receipt is written before any generated source or binary may be
  # reused.  A partial directory without it has no defensible build lineage;
  # refusing that directory is safer than letting the build system call an
  # old target up-to-date under a new checkout or SDK.
  input_authority="$out/build-input-authority.json"
  python3 -B - "$root" "$sdk" "$receipt" "$input_authority" \
    "$preexisting" "$per_unit" "$min_free_kib" "$available_kib" \
    "$plan" "$host_cxx" "$frozen_partition_plan" "$partition_id" \
    "$global_preflight" <<'PY' || return 2
import hashlib,json,os,pathlib,subprocess,sys
root,sdk,receipt,out=map(pathlib.Path,sys.argv[1:5])
preexisting=int(sys.argv[5]); per_unit=int(sys.argv[6])
min_free_kib=int(sys.argv[7]); available_kib=int(sys.argv[8])
plan_path=pathlib.Path(sys.argv[9]); plan=json.loads(plan_path.read_text())
host_cxx=pathlib.Path(sys.argv[10])
partition_plan_arg=sys.argv[11]; partition_id_arg=sys.argv[12]
global_preflight=pathlib.Path(sys.argv[13])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sdk_file(path):
 return {"path":str(path.relative_to(sdk)),"size":path.stat().st_size,
         "sha256":sha(path),
         "symlink_target":os.readlink(path) if path.is_symlink() else None}
submodules=[]
for line in subprocess.check_output(
    ["git","-C",str(root),"submodule","status","--recursive"],text=True).splitlines():
 if not line or line[0]!=" ": raise SystemExit(f"submodule absent/dirty/conflicted: {line}")
 fields=line[1:].split(); path=fields[1]
 current=subprocess.check_output(
     ["git","-C",str(root/path),"rev-parse","HEAD"],text=True).strip()
 if current!=fields[0]: raise SystemExit(f"submodule gitlink/current differ: {path}")
 if subprocess.check_output(
     ["git","-C",str(root/path),"status","--porcelain"],text=True).strip():
  raise SystemExit(f"submodule worktree dirty: {path}")
 submodules.append({"path":path,"gitlink":fields[0],"current":current})
runtime=[sdk_file(path) for path in sorted(sdk.rglob("libhggc*.so*"))
         if path.is_file() or path.is_symlink()]
if not runtime: raise SystemExit("SDK exposes no libhggc runtime libraries")
doc={"schema":"quactlize.scalefirst_kpack_build_input.v1",
 "source_sha":subprocess.check_output(
     ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
 "source_tree":subprocess.check_output(
     ["git","-C",str(root),"rev-parse","HEAD^{tree}"],text=True).strip(),
 "submodules":submodules,
 "sdk":{"receipt":sdk_file(receipt),"compiler":sdk_file(sdk/"bin/hgcc"),
        "inspector":sdk_file(sdk/"bin/hgobjdump"),"runtime_libraries":runtime},
 "host_probe_compiler":{"path":str(host_cxx),"size":host_cxx.stat().st_size,
                        "sha256":sha(host_cxx)},
 "global_preflight":{"path":"inputs/kpack-global-preflight.json",
                     "sha256":sha(global_preflight),
                     "schema":json.loads(global_preflight.read_text()).get("schema")},
 "configuration":{"ppu_arch":"ppu0010","configs_per_unit":per_unit,
                  "scope":plan["scope"],
                  "parents_per_binary":plan["parents_per_binary"],
                  "formats":sorted({row["qtype"] for row in plan["pairs"]}),
                  "operators":sorted({row["operator"] for row in plan["pairs"]}),
                  "shard_plan_sha256":sha(plan_path),
                  "scratch_policy":"ONE_SHARD_THEN_COMPACT_PAYLOAD"},
 "space_preflight":{"minimum_free_kib":min_free_kib,
                    "observed_free_kib":available_kib,
                    "estimate_scope":"ONE_LARGEST_SHARD"}}
if partition_plan_arg != "-":
 sys.path.insert(0,str(root/"tools"))
 import kpack_discovery_build_partitions as partitions
 partition_path=pathlib.Path(partition_plan_arg)
 partition_doc=partitions.read_plan(partition_path)
 doc["configuration"]["distributed_partition"] = partitions.authority_partition_record(
     partition_path,partition_doc,int(partition_id_arg),"scalefirst")
if preexisting and out.is_file():
 previous=json.loads(out.read_text())
 doc["space_preflight"]["observed_free_kib"]=(
     previous.get("space_preflight",{}).get("observed_free_kib"))
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"
if preexisting:
 if not out.is_file(): raise SystemExit("resumable build input authority is missing")
 if out.read_text()!=encoded: raise SystemExit("resumable build input authority differs")
else:
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
 temporary.write_text(encoded); os.replace(temporary,out)
PY

  build_identity_probe "$root" "$out" "$sdk" "$host_cxx" \
    "$input_authority" || return $?
  python3 -B "$root/tools/scalefirst_internal_matrix.py" self-test || return 2
  python3 -B "$root/tools/scalefirst_grouped_kpack_matrix.py" self-test || return 2
  python3 -B "$root/tools/scalefirst_kpack_binary_shards.py" self-test || return 2
  python3 -B "$root/tools/analyze_scalefirst_kpack_discovery.py" self-test || return 2
  python3 -B "$root/ci/check_scalefirst_grouped_kpack_discovery.py" || return 2
  if [ "$partitioned" = 0 ]; then
    shard_rows="$out/shard-plan.tsv"
    python3 -B - "$root" "$plan" "$shard_rows" <<'PY' || return 2
import importlib.util,json,os,pathlib,sys
root,plan_path,out=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(root/"tools"))
import scalefirst_kpack_binary_shards as planner
doc=json.loads(plan_path.read_text()); planner.validate_plan(doc)
lines=["\t".join(map(str,(row["shard_id"],row["qtype"],row["operator"],
                             row["layout"],row["parent_begin"],row["parent_end"])))
       for row in doc["shards"]]
encoded="\n".join(lines)+"\n"
if out.exists() and out.read_text()!=encoded: raise SystemExit("resumable shard TSV differs")
if not out.exists():
 temporary=out.with_name(f".{out.name}.current.{os.getpid()}")
 temporary.write_text(encoded); os.replace(temporary,out)
PY
  fi
  while IFS=$'\t' read -r shard_id q operator layout begin end; do
    [ -n "$shard_id" ] || { fail "empty row in shard plan"; return $?; }
    build_shard "$root" "$out" "$sdk" "$input_authority" \
      "$global_preflight" "$jobs" "$per_unit" "$min_free_kib" \
      "$shard_id" "$q" "$operator" \
      "$layout" "$begin" "$end" || return $?
  done < "$shard_rows"

  if [ "$partitioned" = 1 ]; then
    PPU_SDK="$sdk" python3 -B \
      "$root/tools/kpack_discovery_build_partitions.py" record \
      --root "$out" --route scalefirst || return 2
    printf '[sf-kpack-bundle] PARTITION_COMPLETE id=%s shards=%s bundle=%s\n' \
      "$partition_id" "$(wc -l < "$shard_rows")" "$out"
    return 0
  fi

  python3 -B - "$root" "$out" "$sdk" "$receipt" <<'PY' || return 2
import hashlib,json,os,pathlib,re,subprocess,sys
root,out,sdk,receipt=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(root/"tools"))
import analyze_scalefirst_kpack_discovery as a
import scalefirst_kpack_binary_shards as planner
status=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--untracked-files=no"],text=True)
dirty=[line for line in status.splitlines() if line[3:] not in {".coord/BOX.md",".coord/INBOX.md"}]
if dirty: raise SystemExit("tracked build source became dirty")
submodules=[]
for line in subprocess.check_output(["git","-C",str(root),"submodule","status","--recursive"],text=True).splitlines():
 if not line or line[0] != " ": raise SystemExit(f"submodule is absent/dirty/conflicted: {line}")
 fields=line[1:].split(); path=fields[1]
 current=subprocess.check_output(["git","-C",str(root/path),"rev-parse","HEAD"],text=True).strip()
 if current!=fields[0]: raise SystemExit(f"submodule gitlink/current differ: {path}")
 if subprocess.check_output(["git","-C",str(root/path),"status","--porcelain"],text=True).strip():
  raise SystemExit(f"submodule worktree dirty: {path}")
 submodules.append({"path":path,"gitlink":fields[0],"current":current})
def sdk_file(path):
 return {"path":str(path.relative_to(sdk)),"size":path.stat().st_size,"sha256":a.sha256(path),
         "symlink_target":os.readlink(path) if path.is_symlink() else None}
runtime=[]
for path in sorted(sdk.rglob("libhggc*.so*")):
 if path.is_file() or path.is_symlink(): runtime.append(sdk_file(path))
if not runtime: raise SystemExit("SDK exposes no libhggc runtime libraries")
input_authority=out/"build-input-authority.json"
if not input_authority.is_file(): raise SystemExit("build input authority disappeared")
plan_path=out/"shard-plan.json"; plan=json.loads(plan_path.read_text())
planner.validate_plan(plan)
probe=out/"payloads/support/box_identity_probe"
probe_receipt=out/"payloads/support/identity-probe-receipt.json"
expected_probe_receipt={
 "schema":"quactlize.scalefirst_kpack_identity_probe_receipt.v1",
 "build_input_authority_sha256":a.sha256(input_authority),
 "source_sha256":a.sha256(root/"tools/box_identity_probe.cpp"),
 "binary_sha256":a.sha256(probe)}
if not probe.is_file() or not probe_receipt.is_file() or \
   json.loads(probe_receipt.read_text())!=expected_probe_receipt:
 raise SystemExit("identity probe receipt chain differs")
probe_host=subprocess.check_output(["readelf","-h",str(probe)],text=True)
probe_machine=next((line.split(":",1)[1].strip() for line in probe_host.splitlines()
                    if "Machine:" in line),"")
doc={"schema":"quactlize.scalefirst_kpack_prebuilt_bundle.v2",
 "scope":plan["scope"],"route":"scalefirst",
 "parents_per_binary":plan["parents_per_binary"],
 "source_sha":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
 "repository":{"tree":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD^{tree}"],text=True).strip(),
               "tracked_dirty_ignored":[".coord/BOX.md",".coord/INBOX.md"],"submodules":submodules},
 "sdk":{"receipt":sdk_file(receipt),"compiler":sdk_file(sdk/"bin/hgcc"),
        "inspector":sdk_file(sdk/"bin/hgobjdump"),"runtime_libraries":runtime},
 "build_input_authority":{"path":str(input_authority.relative_to(out)),
                          "sha256":a.sha256(input_authority)},
 "shard_plan":{"path":str(plan_path.relative_to(out)),"sha256":a.sha256(plan_path),
               "pairs":plan["pairs"]},
 "runtime_probe":{"binary":str(probe.relative_to(out)),
                  "binary_sha256":a.sha256(probe),
                  "receipt":str(probe_receipt.relative_to(out)),
                  "receipt_sha256":a.sha256(probe_receipt),
                  "host_machine":probe_machine},
 "shards":[]}
coverage={}
for planned in plan["shards"]:
 q=int(planned["qtype"]); op=str(planned["operator"]); shard_id=planned["shard_id"]
 binary=("test_scalefirst_internal_sweep" if op=="dense" else
         "test_scalefirst_grouped_kpack_discovery")
 manifest=out/f"generated/{shard_id}/manifest.json"; parsed=a.validate_manifest(op,q,manifest)
 binpath=out/f"payloads/{shard_id}/{binary}"
 binary_receipt=out/f"payloads/{shard_id}/binary-receipt.json"
 parent_range=parsed["parent_range"]
 parent_ids=[row["parent_id"] for row in parsed["compiled_parents"]]
 parent_symbols=[row["symbol"] for row in parsed["compiled_parents"]]
 live_symbols=planner.authority_symbols(op,q)[planned["parent_begin"]:
                                              planned["parent_end"]]
 if (parent_range["begin"]!=planned["parent_begin"] or
     parent_range["end"]!=planned["parent_end"] or
     parent_range["authority_count"]!=planned["authority_parents"] or
     parent_ids!=list(range(planned["parent_begin"],planned["parent_end"])) or
     parent_symbols!=live_symbols):
  raise SystemExit(f"{shard_id}: manifest parent range differs from plan")
 coverage.setdefault((q,op),[]).extend(parent_ids)
 if not binpath.is_file(): raise SystemExit(f"missing binary {binpath}")
 expected_receipt={"schema":"quactlize.scalefirst_kpack_binary_receipt.v1",
  "build_input_authority_sha256":a.sha256(input_authority),
  "manifest_sha256":a.sha256(manifest),"binary_sha256":a.sha256(binpath)}
 if not binary_receipt.is_file() or json.loads(binary_receipt.read_text())!=expected_receipt:
  raise SystemExit(f"{shard_id}: binary receipt chain differs")
 elf=subprocess.check_output([str(sdk/"bin/hgobjdump"),"-lelf",str(binpath)],text=True,stderr=subprocess.STDOUT)
 match=re.search(r"ELF FILE \d+ \((PPU [^)]+)\)",elf)
 if not match: raise SystemExit(f"{binpath}: inspector found no PPU ELF image")
 host=subprocess.check_output(["readelf","-h",str(binpath)],text=True)
 host_machine=next((line.split(":",1)[1].strip() for line in host.splitlines() if "Machine:" in line),"")
 doc["shards"].append({
  "shard_id":shard_id,"route":"scalefirst","qtype":q,"operator":op,
  "layout":a.LAYOUT[q],"mapping_id":a.MAPPING[a.LAYOUT[q]],
  "parent_begin":planned["parent_begin"],"parent_end":planned["parent_end"],
  "parent_ids":parent_ids,"parent_symbols":parent_symbols,
  "authority_parents":planned["authority_parents"],
  "manifest":str(manifest.relative_to(out)),"manifest_sha256":a.sha256(manifest),
  "binary":str(binpath.relative_to(out)),"binary_sha256":a.sha256(binpath),
  "binary_receipt":str(binary_receipt.relative_to(out)),
  "binary_receipt_sha256":a.sha256(binary_receipt),
  "elf":{"host_machine":host_machine,"device_arch":match.group(1),
         "inspector_output_sha256":hashlib.sha256(elf.encode()).hexdigest()}})
for pair in plan["pairs"]:
 key=(int(pair["qtype"]),str(pair["operator"])); ids=coverage.get(key,[])
 expected_end=(pair["authority_parents"] if plan["scope"]=="full" else
               min(pair["authority_parents"],plan["parents_per_binary"]))
 if ids!=list(range(expected_end)) or len(ids)!=len(set(ids)):
  raise SystemExit(f"q{key[0]}/{key[1]}: bundle parent union has gap/overlap")
if plan["scope"]=="full":
 q4_symbols=[symbol for row in doc["shards"]
             if row["qtype"]==12 and row["operator"]=="dense"
             for symbol in row["parent_symbols"]]
 if a.Q4_HISTORICAL_GEOMETRY_ANCHOR not in q4_symbols:
  raise SystemExit("Q4 historical geometry anchor left the full parent union")
encoded=json.dumps(doc,indent=2,sort_keys=True)+"\n"; path=out/"bundle.json"
if path.exists() and path.read_text()!=encoded: raise SystemExit("resumed bundle authority changed")
if not path.exists():
 tmp=path.with_name(f".{path.name}.current.{os.getpid()}"); tmp.write_text(encoded); os.replace(tmp,path)
PY
  printf '[sf-kpack-bundle] COMPLETE scope=%s shards=%s bundle=%s\n' \
    "$scope" "$(wc -l < "$shard_rows")" "$out"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
