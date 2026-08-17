#!/usr/bin/env bash
# Resolve real GGUF shapes, run the two independent sweep components, and
# publish four fail-closed leaderboards.  This file owns orchestration only.
set -uo pipefail

fail() {
  printf '[internal-full-sweep] FAIL: %s\n' "$*" >&2
  return 2
}

atomic_copy() {
  local source=$1 destination=$2 pending="${2}.partial"
  if [ -e "$destination" ] || [ -L "$destination" ] || \
     [ -e "$pending" ] || [ -L "$pending" ]; then
    fail "atomic copy target already exists: $destination"
    return 2
  fi
  cp -- "$source" "$pending" || return 2
  mv -- "$pending" "$destination" || return 2
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

validate_component_summary() {
  python3 -B - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import hashlib, json, pathlib, sys
path, attempt, component = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
spec_path, root_sha, actlize_sha = pathlib.Path(sys.argv[4]), sys.argv[5], sys.argv[6]
try:
    doc = json.loads(path.read_text())
    spec = json.loads(spec_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read component summary {path}: {exc}")
if doc.get("status") != "COMPLETE":
    raise SystemExit(f"{component} summary is not COMPLETE")
if doc.get("component") != component:
    raise SystemExit(f"component summary identity differs: {doc.get('component')!r}")
provenance = doc.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit(f"{component} summary lacks provenance")
if provenance.get("orchestration_attempt_id") != attempt:
    raise SystemExit(
        f"{component} summary is stale: attempt="
        f"{provenance.get('orchestration_attempt_id')!r}, expected={attempt!r}")
expected_provenance = {
    "shape_manifest_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    "gguf_hashes": spec["provenance"]["gguf_hashes"],
    "gguf_set_sha256": spec["provenance"]["gguf_set_sha256"],
    "shape_directory": spec["provenance"]["shape_directory"],
    "root_sha": root_sha,
    "actlize_sha": actlize_sha,
}
for key, expected in expected_provenance.items():
    if provenance.get(key) != expected:
        raise SystemExit(f"{component} summary provenance differs from frozen {key}")

expected = {(row["model_id"], row["shape_id"]): row
            for row in spec["sweep_shapes"]}
cells = doc.get("cells")
if not isinstance(cells, list):
    raise SystemExit(f"{component} summary cells are missing")
actual_keys = {(cell.get("model_id"), cell.get("shape_id")) for cell in cells
               if isinstance(cell, dict)}
if actual_keys != set(expected):
    raise SystemExit(
        f"{component} summary shape membership differs from frozen inventory: "
        f"missing={sorted(set(expected)-actual_keys)} "
        f"extra={sorted(actual_keys-set(expected))}")
for index, cell in enumerate(cells):
    if not isinstance(cell, dict):
        raise SystemExit(f"{component} cell[{index}] is not an object")
    row = expected[(cell["model_id"], cell["shape_id"])]
    shape = cell.get("shape") or {}
    exact = {
        "tp_world": row["tp_world"], "tp_rank": row["tp_rank"],
        "tp_partition": row["tp_partition"],
        "problem_route": row["problem_route"],
        "group_size": row["group_size"], "qtype": row["qtype"],
        "source_tensors": row["source_tensors"],
        "grouped": row["grouped"],
    }
    for key, expected_value in exact.items():
        if cell.get(key) != expected_value:
            raise SystemExit(f"{component} cell[{index}] differs from frozen {key}")
    for key, source_key in (("m", "M"), ("n", "N"), ("k", "K"), ("l", "L")):
        if shape.get(key) != row[source_key]:
            raise SystemExit(f"{component} cell[{index}] differs from frozen shape.{key}")
PY
}

validate_completion() {
  python3 -B - "$@" <<'PY'
import hashlib, json, pathlib, sys
manifest = pathlib.Path(sys.argv[1])
expected_mode, expected_input, expected_identity = sys.argv[2:5]
scale, fully_quantized, results = map(pathlib.Path, sys.argv[5:8])
try:
    doc = json.loads(manifest.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid completion manifest: {exc}")
if doc.get("schema") != "quactlize.internal_full_sweep.completion.v1":
    raise SystemExit("completion schema mismatch")
if doc.get("publication_mode") != expected_mode:
    raise SystemExit("completion publication mode mismatch")
if doc.get("input_state_sha256") != expected_input:
    raise SystemExit("completion input-state mismatch")
if doc.get("orchestration_identity_sha256") != expected_identity:
    raise SystemExit("completion orchestration-identity mismatch")
for name, path in (("scale_first_summary_sha256", scale),
                   ("fully_quantized_summary_sha256", fully_quantized)):
    if not path.is_file():
        raise SystemExit(f"completion member is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if doc.get(name) != digest:
        raise SystemExit(f"completion member hash mismatch: {name}")
if not results.is_dir() or results.is_symlink():
    raise SystemExit(f"completion results tree is missing or unsafe: {results}")
actual = {
    str(path.relative_to(results)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(results.rglob("*")) if path.is_file() and not path.is_symlink()
}
if doc.get("results_manifest") != actual:
    missing = sorted(set(doc.get("results_manifest", {})) - set(actual))
    extra = sorted(set(actual) - set(doc.get("results_manifest", {})))
    raise SystemExit(f"completion results manifest mismatch: missing={missing} extra={extra}")
PY
}

main() {
  if [ "$#" -ne 0 ]; then
    fail 'this runner accepts no positional arguments'
    return 2
  fi

  local root workspace_root sha short stamp out resume preexisting progressed
  local dev_mode publication_mode
  local inputs frozen_catalog frozen_gguf inventory_dir frozen_inventory
  local requested_catalog requested_gguf requested_spec catalog gguf_set spec
  local input_state input_state_file validation_log
  local scale_runner fq_runner scale_summary_rel fq_summary_rel
  local scale_out fq_out scale_summary fq_summary
  local identity identity_current identity_sha provenance provenance_current
  local completion attempt_id attempt_root merge_stage
  local scale_log fq_log scale_rc fq_rc self_rc merge_rc
  local -a pipe_status

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-internal-full-sweep-${short}-${stamp}}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) fail "OUT must be a strict /workspace child: $out"; return 2 ;;
  esac
  resume="${RESUME:-0}"
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return 2;; esac
  dev_mode="${INTERNAL_SWEEP_DEV_MODE:-0}"
  case "$dev_mode" in 0|1) ;; *) fail 'INTERNAL_SWEEP_DEV_MODE must be 0 or 1'; return 2;; esac
  if [ "$dev_mode" = 0 ] && { [ -n "${INTERNAL_SWEEP_CATALOG:-}" ] ||
       [ -n "${GGUF_SET:-}" ] || [ -n "${INTERNAL_SWEEP_SPEC:-}" ] ||
       [ -n "${SCALEFIRST_RUNNER:-}" ] || [ -n "${FULLY_QUANTIZED_RUNNER:-}" ] ||
       [ -n "${SCALEFIRST_SUMMARY_REL:-}" ] ||
       [ -n "${FULLY_QUANTIZED_SUMMARY_REL:-}" ]; }; then
    fail 'catalog/import/runner/summary overrides require INTERNAL_SWEEP_DEV_MODE=1'
    return 2
  fi
  publication_mode=$([ "$dev_mode" = 1 ] && printf development || printf production)

  preexisting=0
  if [ -e "$out" ] || [ -L "$out" ]; then
    if [ "$resume" != 1 ] || [ ! -d "$out" ] || [ -L "$out" ]; then
      fail "refusing existing OUT=$out; set RESUME=1 for this exact directory"
      return 2
    fi
    preexisting=1
  else
    mkdir "$out" || return 2
  fi
  inputs="$out/inputs"
  mkdir -p "$inputs" || return 2
  progressed=0
  for path in "$out/scale-first" "$out/fully-quantized" "$out/results" \
              "$out/completion.json" "$out/attempts"; do
    if [ -e "$path" ] || [ -L "$path" ]; then progressed=1; fi
  done

  # The bundle owns its catalog after the first successful write.  An
  # explicitly supplied external catalog is compared when readable, but a
  # vanished source does not make a completed bundle non-self-contained.
  frozen_catalog="$inputs/catalog.json"
  requested_catalog="${INTERNAL_SWEEP_CATALOG:-$root/benchmarks/internal_sweep_models.json}"
  if [ ! -s "$frozen_catalog" ]; then
    if [ "$progressed" = 1 ]; then
      fail 'measurement state exists but frozen catalog authority is missing'
      return 2
    fi
    requested_catalog="$(realpath -e -- "$requested_catalog")" || return 2
    atomic_copy "$requested_catalog" "$frozen_catalog" || return 2
  elif [ -e "$requested_catalog" ]; then
    requested_catalog="$(realpath -e -- "$requested_catalog")" || return 2
    if ! cmp -s -- "$requested_catalog" "$frozen_catalog"; then
      fail 'requested catalog differs from bundle-frozen catalog'
      return 2
    fi
  else
    printf '[internal-full-sweep] NOTE: external catalog unavailable; using frozen bundle authority\n'
  fi
  catalog="$frozen_catalog"

  frozen_gguf="$inputs/resolved-models.json"
  requested_gguf="${GGUF_SET:-}"
  if [ ! -s "$frozen_gguf" ]; then
    if [ "$progressed" = 1 ]; then
      fail 'measurement state exists but frozen resolved-model authority is missing'
      return 2
    fi
    if [ -n "$requested_gguf" ]; then
      requested_gguf="$(realpath -e -- "$requested_gguf")" || return 2
      atomic_copy "$requested_gguf" "$frozen_gguf" || return 2
    else
      if [ -e "${frozen_gguf}.partial" ] || [ -L "${frozen_gguf}.partial" ]; then
        fail "stale resolved-model partial exists: ${frozen_gguf}.partial"
        return 2
      fi
      python3 -B "$root/tools/resolve_internal_sweep_models.py" resolve \
        --catalog "$catalog" --output "${frozen_gguf}.partial" || return 2
      mv -- "${frozen_gguf}.partial" "$frozen_gguf" || return 2
    fi
  elif [ -n "$requested_gguf" ] && [ -e "$requested_gguf" ]; then
    requested_gguf="$(realpath -e -- "$requested_gguf")" || return 2
    if ! cmp -s -- "$requested_gguf" "$frozen_gguf"; then
      fail 'GGUF_SET differs from bundle-frozen resolved model set'
      return 2
    fi
  fi
  gguf_set="$frozen_gguf"

  inventory_dir="$inputs/inventory"
  frozen_inventory="$inventory_dir/inventory.json"
  requested_spec="${INTERNAL_SWEEP_SPEC:-}"
  if [ ! -s "$frozen_inventory" ]; then
    if [ "$progressed" = 1 ]; then
      fail 'measurement state exists but frozen inventory authority is missing'
      return 2
    fi
    if [ -e "$inventory_dir" ] || [ -L "$inventory_dir" ]; then
      if [ -n "$(find "$inventory_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        fail "partial inventory directory exists: $inventory_dir"
        return 2
      fi
      rmdir "$inventory_dir" || return 2
    fi
    if [ -n "$requested_spec" ]; then
      requested_spec="$(realpath -e -- "$requested_spec")" || return 2
      mkdir "$inventory_dir" || return 2
      atomic_copy "$requested_spec" "$frozen_inventory" || return 2
    else
      if [ -e "${inventory_dir}.partial" ] || [ -L "${inventory_dir}.partial" ]; then
        fail "stale inventory partial exists: ${inventory_dir}.partial"
        return 2
      fi
      python3 -B "$root/tools/gguf_internal_shape_inventory.py" \
        --resolved "$gguf_set" --output-dir "${inventory_dir}.partial" || return 2
      mv -- "${inventory_dir}.partial" "$inventory_dir" || return 2
    fi
  elif [ -n "$requested_spec" ] && [ -e "$requested_spec" ]; then
    requested_spec="$(realpath -e -- "$requested_spec")" || return 2
    if ! cmp -s -- "$requested_spec" "$frozen_inventory"; then
      fail 'INTERNAL_SWEEP_SPEC differs from bundle-frozen inventory'
      return 2
    fi
  fi
  spec="$frozen_inventory"

  python3 -B "$root/tools/resolve_internal_sweep_models.py" self-test || return 2
  python3 -B "$root/tools/gguf_internal_shape_inventory.py" --self-test || return 2
  python3 -B "$root/ci/check_gguf_internal_shape_inventory.py" || return 2
  validation_log="$inputs/authority-validation.current.log"
  python3 -B "$root/tools/validate_internal_sweep_authorities.py" \
    --catalog "$catalog" --resolved "$gguf_set" --inventory "$spec" \
    >"${validation_log}.partial" 2>&1 || {
      local authority_rc=$?
      sed -n '1,200p' "${validation_log}.partial" >&2
      return "$authority_rc"
    }
  mv -f -- "${validation_log}.partial" "$validation_log" || return 2

  input_state="$({
    sha_file "$catalog"
    sha_file "$gguf_set"
    sha_file "$spec"
  } | sha256sum | awk '{print $1}')" || return 2
  input_state_file="$inputs/input-state.sha256"
  if [ -s "$input_state_file" ]; then
    if [ "$(cat "$input_state_file")" != "$input_state" ]; then
      fail 'catalog/resolved-model/inventory authority changed inside bundle'
      return 2
    fi
  else
    if [ "$progressed" = 1 ]; then
      fail 'measurement state exists but input-state authority is missing'
      return 2
    fi
    printf '%s\n' "$input_state" >"${input_state_file}.partial" || return 2
    mv -- "${input_state_file}.partial" "$input_state_file" || return 2
  fi

  scale_runner="${SCALEFIRST_RUNNER:-$root/tools/run_scalefirst_internal_sweep_box.sh}"
  fq_runner="${FULLY_QUANTIZED_RUNNER:-$root/tools/run_fully_quantized_internal_sweep_box.sh}"
  for runner in "$scale_runner" "$fq_runner"; do
    if [ ! -f "$runner" ]; then fail "component runner not found: $runner"; return 2; fi
  done
  scale_summary_rel="${SCALEFIRST_SUMMARY_REL:-results/summary.json}"
  fq_summary_rel="${FULLY_QUANTIZED_SUMMARY_REL:-results/summary.json}"
  case "$scale_summary_rel:$fq_summary_rel" in
    *..*|*//*|/*) fail 'summary paths must be simple relative children'; return 2 ;;
  esac

  identity="$inputs/orchestration.identity.txt"
  identity_current="$inputs/orchestration.identity.current.txt"
  {
    printf 'identity_schema=quactlize.internal_full_sweep.orchestration_identity.v2\n'
    printf 'publication_mode=%s\n' "$publication_mode"
    printf 'root_sha=%s\n' "$sha"
    printf 'actlize_sha=%s\n' "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'top_runner_sha256=%s\n' "$(sha_file "$root/tools/run_internal_full_sweep_box.sh")"
    printf 'authority_validator_sha256=%s\n' "$(sha_file "$root/tools/validate_internal_sweep_authorities.py")"
    printf 'scale_first_runner=%s\n' "$scale_runner"
    printf 'scale_first_runner_sha256=%s\n' "$(sha_file "$scale_runner")"
    printf 'fully_quantized_runner=%s\n' "$fq_runner"
    printf 'fully_quantized_runner_sha256=%s\n' "$(sha_file "$fq_runner")"
    printf 'merger_sha256=%s\n' "$(sha_file "$root/tools/merge_internal_full_sweep.py")"
    printf 'input_state_sha256=%s\n' "$input_state"
    printf 'scale_first_summary_rel=%s\n' "$scale_summary_rel"
    printf 'fully_quantized_summary_rel=%s\n' "$fq_summary_rel"
  } >"${identity_current}.partial" || return 2
  mv -f -- "${identity_current}.partial" "$identity_current" || return 2
  if [ -s "$identity" ]; then
    if ! cmp -s -- "$identity_current" "$identity"; then
      fail 'source/input/runner identity changed inside resumed bundle'
      diff -u -- "$identity" "$identity_current" >&2 || true
      return 2
    fi
  else
    if [ "$progressed" = 1 ]; then
      fail 'measurement state exists but orchestration identity is missing'
      return 2
    fi
    atomic_copy "$identity_current" "$identity" || return 2
  fi
  identity_sha="$(sha_file "$identity")" || return 2

  provenance="$out/orchestration.provenance.txt"
  provenance_current="$inputs/orchestration.provenance.current.txt"
  {
    printf 'schema=quactlize.internal_full_sweep.run.v2\n'
    cat "$identity"
    printf 'catalog=%s\n' "$catalog"
    printf 'catalog_file_sha256=%s\n' "$(sha_file "$catalog")"
    printf 'gguf_set=%s\n' "$gguf_set"
    printf 'gguf_set_file_sha256=%s\n' "$(sha_file "$gguf_set")"
    printf 'internal_sweep_spec=%s\n' "$spec"
    printf 'internal_sweep_spec_file_sha256=%s\n' "$(sha_file "$spec")"
  } >"${provenance_current}.partial" || return 2
  mv -f -- "${provenance_current}.partial" "$provenance_current" || return 2
  if [ -s "$provenance" ]; then
    if ! cmp -s -- "$provenance_current" "$provenance"; then
      fail 'immutable orchestration provenance is truncated or changed'
      return 2
    fi
  else
    if [ "$progressed" = 1 ]; then
      fail 'measurement state exists but orchestration provenance is missing'
      return 2
    fi
    atomic_copy "$provenance_current" "$provenance" || return 2
  fi

  scale_out="$out/scale-first"
  fq_out="$out/fully-quantized"
  scale_summary="$scale_out/$scale_summary_rel"
  fq_summary="$fq_out/$fq_summary_rel"
  completion="$out/completion.json"
  if [ -e "$completion" ] || [ -L "$completion" ]; then
    validate_completion "$completion" "$publication_mode" "$input_state" "$identity_sha" \
      "$scale_summary" "$fq_summary" "$out/results" || return 2
    if [ "$publication_mode" = production ]; then
      printf '[internal-full-sweep] PASS (idempotent completed-bundle resume)\n'
    else
      printf '[internal-full-sweep] DEVELOPMENT-COMPLETE (idempotent resume; not production evidence)\n'
    fi
    printf '[internal-full-sweep] cells: %s\n' "$out/results/cells.tsv"
    printf '[internal-full-sweep] winners: %s\n' "$out/results/winners.tsv"
    printf '[internal-full-sweep] summary: %s\n' "$out/results/summary.json"
    printf '[internal-full-sweep] artifacts: %s\n' "$out"
    return 0
  fi

  mkdir -p "$out/attempts" || return 2
  attempt_id="${stamp}-p$$-${input_state:0:12}"
  attempt_root="$out/attempts/$attempt_id"
  if [ -e "$attempt_root" ] || [ -L "$attempt_root" ]; then
    fail "attempt directory collision: $attempt_root"
    return 2
  fi
  mkdir "$attempt_root" || return 2
  {
    printf 'attempt_id=%s\n' "$attempt_id"
    printf 'attempt_utc=%s\n' "$stamp"
    printf 'resume=%s\n' "$resume"
    printf 'input_state_sha256=%s\n' "$input_state"
  } >"$attempt_root/attempt.txt" || return 2

  scale_log="$attempt_root/scale-first.runner.log"
  printf '[internal-full-sweep] running ScaleFirst component: %s\n' "$scale_runner"
  OUT="$scale_out" RESUME="$resume" GGUF_SET="$gguf_set" INTERNAL_SWEEP_SPEC="$spec" \
    INTERNAL_SWEEP_COMPONENT=scale_first INTERNAL_SWEEP_ATTEMPT_ID="$attempt_id" \
    bash "$scale_runner" 2>&1 | tee "$scale_log"
  pipe_status=("${PIPESTATUS[@]}")
  scale_rc=${pipe_status[0]}
  if [ "${pipe_status[1]}" -ne 0 ]; then fail 'could not persist ScaleFirst runner log'; return 2; fi
  if [ "$scale_rc" -eq 0 ]; then
    validate_component_summary "$scale_summary" "$attempt_id" scale_first \
      "$spec" "$sha" "$(git -C "$root/third_party/actlize" rev-parse HEAD)" || scale_rc=2
  fi

  fq_log="$attempt_root/fully-quantized.runner.log"
  printf '[internal-full-sweep] running FullyQuantized component: %s\n' "$fq_runner"
  OUT="$fq_out" RESUME="$resume" GGUF_SET="$gguf_set" INTERNAL_SWEEP_SPEC="$spec" \
    INTERNAL_SWEEP_COMPONENT=fully_quantized INTERNAL_SWEEP_ATTEMPT_ID="$attempt_id" \
    bash "$fq_runner" 2>&1 | tee "$fq_log"
  pipe_status=("${PIPESTATUS[@]}")
  fq_rc=${pipe_status[0]}
  if [ "${pipe_status[1]}" -ne 0 ]; then fail 'could not persist FullyQuantized runner log'; return 2; fi
  if [ "$fq_rc" -eq 0 ]; then
    validate_component_summary "$fq_summary" "$attempt_id" fully_quantized \
      "$spec" "$sha" "$(git -C "$root/third_party/actlize" rev-parse HEAD)" || fq_rc=2
  fi
  {
    printf 'scale_first_rc=%d\n' "$scale_rc"
    printf 'fully_quantized_rc=%d\n' "$fq_rc"
    printf 'scale_first_log_sha256=%s\n' "$(sha_file "$scale_log")"
    printf 'fully_quantized_log_sha256=%s\n' "$(sha_file "$fq_log")"
  } >>"$attempt_root/attempt.txt" || return 2
  if [ "$scale_rc" -ne 0 ] || [ "$fq_rc" -ne 0 ]; then
    printf '[internal-full-sweep] INCOMPLETE: scale_first_rc=%d fully_quantized_rc=%d; no merged winner published\n' \
      "$scale_rc" "$fq_rc" >&2
    printf '[internal-full-sweep] artifacts: %s\n' "$out" >&2
    return 3
  fi

  python3 -B "$root/tools/merge_internal_full_sweep.py" self-test \
    | tee "$attempt_root/merger-self-test.log"
  pipe_status=("${PIPESTATUS[@]}")
  self_rc=${pipe_status[0]}
  if [ "${pipe_status[1]}" -ne 0 ]; then fail 'could not persist merger self-test log'; return 2; fi
  if [ "$self_rc" -ne 0 ]; then fail "merger self-test returned rc=$self_rc"; return "$self_rc"; fi

  merge_stage="$attempt_root/merged"
  python3 -B "$root/tools/merge_internal_full_sweep.py" merge \
    --scale-first "$scale_summary" --fully-quantized "$fq_summary" \
    --out "$merge_stage" 2>&1 | tee "$attempt_root/merge.log"
  pipe_status=("${PIPESTATUS[@]}")
  merge_rc=${pipe_status[0]}
  if [ "${pipe_status[1]}" -ne 0 ]; then fail 'could not persist merge log'; return 2; fi
  if [ "$merge_rc" -ne 0 ]; then
    printf '[internal-full-sweep] INCOMPLETE: merger rejected component results\n' >&2
    return "$merge_rc"
  fi
  if [ -e "$out/results" ] || [ -L "$out/results" ]; then
    mv -- "$out/results" "$attempt_root/preexisting-results" || return 2
  fi
  mv -- "$merge_stage" "$out/results" || return 2

  if [ -e "${completion}.partial" ] || [ -L "${completion}.partial" ]; then
    fail 'stale or unsafe completion partial exists'
    return 2
  fi
  python3 -B - "${completion}.partial" "$publication_mode" "$input_state" "$identity_sha" \
    "$scale_summary" "$fq_summary" "$out/results" <<'PY' || return 2
import hashlib, json, pathlib, sys
output = pathlib.Path(sys.argv[1])
publication_mode, input_state, identity = sys.argv[2:5]
scale, fully_quantized, results = map(pathlib.Path, sys.argv[5:8])
doc = {
    "schema": "quactlize.internal_full_sweep.completion.v1",
    "publication_mode": publication_mode,
    "input_state_sha256": input_state,
    "orchestration_identity_sha256": identity,
    "scale_first_summary_sha256": hashlib.sha256(scale.read_bytes()).hexdigest(),
    "fully_quantized_summary_sha256": hashlib.sha256(
        fully_quantized.read_bytes()).hexdigest(),
    "results_manifest": {
        str(path.relative_to(results)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(results.rglob("*")) if path.is_file() and not path.is_symlink()
    },
}
output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
PY
  if [ -e "$completion" ] || [ -L "$completion" ]; then
    fail 'completion manifest appeared concurrently'
    return 2
  fi
  mv -- "${completion}.partial" "$completion" || return 2
  validate_completion "$completion" "$publication_mode" "$input_state" "$identity_sha" \
    "$scale_summary" "$fq_summary" "$out/results" || return 2

  if [ "$publication_mode" = production ]; then
    printf '[internal-full-sweep] PASS\n'
  else
    printf '[internal-full-sweep] DEVELOPMENT-COMPLETE (not production evidence)\n'
  fi
  printf '[internal-full-sweep] cells: %s\n' "$out/results/cells.tsv"
  printf '[internal-full-sweep] winners: %s\n' "$out/results/winners.tsv"
  printf '[internal-full-sweep] summary: %s\n' "$out/results/summary.json"
  printf '[internal-full-sweep] artifacts: %s\n' "$out"
}

main "$@"
