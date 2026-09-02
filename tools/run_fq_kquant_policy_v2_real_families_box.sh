#!/usr/bin/env bash
# Execute the five-family Q4 policy sweep from the already-built A04 payload.
set -euo pipefail

fail() {
  printf '[fq-kquant-policy-real-box] FAIL: %s\n' "$*" >&2
  exit 2
}

main() {
  [[ $# = 0 ]] || fail 'no positional arguments are accepted'
  local root artifact_root out bundle bundle_input build_source sdk manifest plan binary library
  local runner_source build_source_head expected_build_source family identity n k round log
  local iterations warmups rounds
  local -a runner_inputs families dense_args

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  artifact_root="$(realpath -e -- "${FQ_KQUANT_POLICY_REAL_ROOT:-/workspace}")" || fail 'artifact root is absent'
  out="$(realpath -m -- "${OUT:-$artifact_root/fq-kquant-policy-real-$(date -u +%Y%m%dT%H%M%SZ)-$$}")"
  case "$out" in "$artifact_root"/*) ;; *) fail 'OUT must be a strict artifact-root child';; esac
  [[ ! -e "$out" && ! -L "$out" ]] || fail "refusing existing OUT: $out"
  case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*|*[!0-9]*) fail 'CUDA_VISIBLE_DEVICES must name exactly one numeric ordinal';; esac
  [[ "${RESUME:-0}" = 0 ]] || fail 'RESUME is not implemented'
  [[ -z "${PPU_DEFS:-}${PPU_EXTRA_DEFS:-}" ]] || fail 'ambient build definitions are forbidden'

  iterations="${PERF_ITERATIONS:-11}"
  warmups="${PERF_WARMUPS:-3}"
  rounds="${PERF_ROUNDS:-3}"
  [[ "$iterations:$warmups:$rounds" = "11:3:3" ]] || fail 'the admitted timing denominator is exactly 3 rounds x 11 samples x 3 warmups'
  [[ "${SWEEP_PROFILE:-kpack-policy-v2}" = kpack-policy-v2 ]] || fail 'profile identity differs'
  [[ "${SWEEP_CONFIGS:-1}" = 1 ]] || fail 'all five categorical candidates are required'

  bundle_input="${FQ_KQUANT_POLICY_V2_BUNDLE:-/nonexistent}"
  [[ -d "$bundle_input" && ! -L "$bundle_input" ]] || fail 'bundle must be a regular non-symlink directory'
  bundle="$(realpath -e -- "$bundle_input")" || fail 'exact prebuilt bundle is required'
  case "$bundle" in "$artifact_root"/*) ;; *) fail 'bundle must be a strict artifact-root child';; esac
  manifest="$bundle/manifest.json"
  [[ -f "$manifest" && ! -L "$manifest" ]] || fail 'bundle manifest is missing or symlinked'

  build_source="${FQ_KQUANT_POLICY_V2_BUILD_SOURCE:-/nonexistent}"
  [[ -d "$build_source" && ! -L "$build_source" ]] || fail 'the frozen bundle-source worktree is required'
  build_source="$(realpath -e -- "$build_source")"
  expected_build_source="425198f5d52377faf85eae4160cd44826e7f4388"
  build_source_head="$(git -C "$build_source" rev-parse HEAD)"
  [[ "$build_source_head" = "$expected_build_source" ]] || fail 'bundle-source worktree is not exact 425198f'
  [[ -f "$build_source/tools/fq_kquant_policy_v2_prebuilt.py" && ! -L "$build_source/tools/fq_kquant_policy_v2_prebuilt.py" ]] || fail 'frozen bundle verifier is absent'

  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  [[ -n "$sdk" ]] || fail 'PPU_SDK is required'
  sdk="$(realpath -e -- "$sdk")"
  [[ -f "$sdk/release.yaml" && ! -L "$sdk/release.yaml" ]] || fail 'SDK release receipt is absent'

  runner_inputs=(
    tools/plan_fq_kquant_policy_v2_real_families.py
    tools/analyze_fq_kquant_policy_v2_real_families.py
    tools/adjudicate_fq_kquant_policy_v2.py
    tools/run_fq_kquant_policy_v2_real_families_box.sh
    tools/probe_box_identity.py
  )
  [[ -z "$(git -C "$root" status --porcelain -- "${runner_inputs[@]}")" ]] || fail 'current runner inputs are dirty or untracked'
  runner_source="$(git -C "$root" rev-parse HEAD)"
  [[ "$runner_source" =~ ^[0-9a-f]{40}$ ]] || fail 'current runner source identity is malformed'

  mkdir -p "$out/inputs" "$out/runs" "$out/results"
  python3 -B "$root/tools/plan_fq_kquant_policy_v2_real_families.py" self-test
  python3 -B "$root/tools/analyze_fq_kquant_policy_v2_real_families.py" self-test
  python3 -B "$build_source/tools/fq_kquant_policy_v2_prebuilt.py" self-test
  python3 -B "$build_source/tools/fq_kquant_policy_v2_prebuilt.py" verify \
    --bundle "$bundle" --source-root "$build_source" --sdk "$sdk" \
    --execution-sdk-compatible

  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$out/inputs/box-identity.json" || fail 'runtime one-device probe failed'
  python3 -B - "$out/inputs/box-identity.json" <<'PY' || fail 'measured one-device evidence is required'
import json, sys
probe = json.load(open(sys.argv[1]))["device_probe"]
assert probe["status"] in ("measured", "properties-unavailable")
assert probe["device_count"] == 1
PY

  plan="$out/inputs/plan.json"
  python3 -B "$root/tools/plan_fq_kquant_policy_v2_real_families.py" materialize --output "$plan"
  cp -- "$manifest" "$out/inputs/bundle-manifest.json"
  binary="$bundle/test_fq_kquant_layout_perf"
  library="$bundle/libquactlize_ppu.so"
  [[ -x "$binary" && -f "$binary" && ! -L "$binary" ]] || fail 'verified executable changed'
  [[ -f "$library" && ! -L "$library" ]] || fail 'verified library changed'

  python3 -B - \
    "$root" "$build_source" "$bundle" "$plan" "$out/inputs/box-identity.json" \
    "$out/inputs/result-authority.json" "$iterations" "$warmups" "$rounds" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys

root, build_root, bundle, plan, box, output = map(pathlib.Path, sys.argv[1:7])
iterations, warmups, rounds = map(int, sys.argv[7:10])
manifest_path = bundle / "manifest.json"
manifest = json.loads(manifest_path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

runner_names = [
    "tools/plan_fq_kquant_policy_v2_real_families.py",
    "tools/analyze_fq_kquant_policy_v2_real_families.py",
    "tools/adjudicate_fq_kquant_policy_v2.py",
    "tools/run_fq_kquant_policy_v2_real_families_box.sh",
    "tools/probe_box_identity.py",
]
runner_files = []
for name in runner_names:
    path = root / name
    assert path.is_file() and not path.is_symlink()
    runner_files.append({"path": name, "size": path.stat().st_size, "sha256": sha(path)})
frozen = build_root / "tools/fq_kquant_policy_v2_prebuilt.py"
assert manifest["source"]["commit"] == git_head(build_root) == "425198f5d52377faf85eae4160cd44826e7f4388"
value = {
    "schema": "quactlize.fq-kquant-policy-real-result-authority.v1",
    "runner_source": {"commit": git_head(root), "files": runner_files},
    "bundle_build_source": {
        "commit": git_head(build_root),
        "frozen_verifier": {"path": "tools/fq_kquant_policy_v2_prebuilt.py", "size": frozen.stat().st_size, "sha256": sha(frozen)},
    },
    "bundle": {
        "manifest_sha256": sha(manifest_path),
        "binary_sha256": sha(bundle / "test_fq_kquant_layout_perf"),
        "library_sha256": sha(bundle / "libquactlize_ppu.so"),
    },
    "inputs": {"plan_sha256": sha(plan), "box_identity_sha256": sha(box)},
    "measurement": {
        "families": 5, "m_min": 1, "m_max": 64, "candidates": 5,
        "rounds": rounds, "iterations": iterations, "warmups": warmups,
        "execution_unit": "one-family-one-round", "raw_bad_required": 0,
    },
}
temporary = output.with_name(f".{output.name}.current.{os.getpid()}")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY

  mapfile -t families < <(python3 -B - "$plan" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["families"]:
    print(row["identity"], row["n"], row["k"])
PY
  )
  [[ ${#families[@]} = 5 ]] || fail 'family denominator differs'
  for family in "${families[@]}"; do
    read -r identity n k <<<"$family"
    [[ "$identity" = "q4-dense-n${n}-k${k}" ]] || fail 'family identity differs'
    dense_args=()
    for m in $(seq 1 64); do
      dense_args+=("--dense=$m,$n,$k")
    done
    [[ ${#dense_args[@]} = 64 ]] || fail 'per-family M denominator differs'
    for round in 1 2 3; do
      log="$out/runs/q12-n${n}-k${k}-round${round}.log"
      LD_LIBRARY_PATH="$bundle${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$binary" --iterations=11 --warmups=3 --round="$round" \
        --order=kpack-first --all-configs=1 --profile=kpack-policy-v2 \
        "${dense_args[@]}" >"$log" 2>&1 || {
          tail -n 160 "$log" >&2
          fail "family=$identity round=$round failed"
        }
      grep -Fqx \
        "FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=12 round=$round layout=kpack order=kpack-first iterations=11 warmups=3 all_configs=1 dense_cases=64 grouped_cases=0 status=PASS" \
        "$log" || fail "family=$identity round=$round completion differs"
      sha256sum "$log" >"$log.sha256"
      printf '[fq-kquant-policy-real-box] PASS family=%s round=%s dense_cases=64\n' "$identity" "$round"
    done
  done

  python3 -B "$root/tools/analyze_fq_kquant_policy_v2_real_families.py" analyze \
    --input "$out" --output "$out/results"
  sha256sum "$out/results/adjudication.json" >"$out/results/adjudication.json.sha256"
  sha256sum "$out/results/adjudication.tsv" >"$out/results/adjudication.tsv.sha256"
  sha256sum "$out/inputs/result-authority.json" >"$out/inputs/result-authority.json.sha256"
  printf '[fq-kquant-policy-real-box] DIAGNOSTIC_COMPLETE runner=%s build_source=%s artifacts=%s\n' \
    "$runner_source" "$build_source_head" "$out"
}

main "$@"
