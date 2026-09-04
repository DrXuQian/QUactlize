#!/usr/bin/env bash
# Verify the pinned artifact worktree, then invoke the compile-free source gate.
set -u -o pipefail

EXPECTED_PARENT=92e9dcaca91b362019354e77ac21536bbc1b51ac
EXPECTED_BRANCH=artifacts/ppu0010/92e9dca-m8-epilogue-perf-ab-eaf274676ac0
EXPECTED_REL=prebuilt/ppu0010/92e9dca/m8-epilogue-perf-ab-eaf274676ac0
EXPECTED_MANIFEST=eaf274676ac0405cc1f3de663ab3d4da06049c730b2ac48a440c7a89c6e7ea8c

usage() {
  printf 'usage: CUDA_VISIBLE_DEVICES=N bash %s --artifact-commit SHA --ppu-sdk DIR --output NEW_DIR\n' "$0" >&2
}

fail() {
  printf '[m8-epilogue-perf-ab-artifact] FAIL: %s\n' "$*" >&2
  return 2
}

main() {
  local artifact_commit= sdk= output= source_root bundle_root bundle actual_parent
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    fail 'run-prebuilt.sh must be executed with bash, not sourced'; return 2
  fi
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --artifact-commit) [ "$#" -ge 2 ] || { usage; return 2; }; artifact_commit=$2; shift 2 ;;
      --ppu-sdk) [ "$#" -ge 2 ] || { usage; return 2; }; sdk=$2; shift 2 ;;
      --output) [ "$#" -ge 2 ] || { usage; return 2; }; output=$2; shift 2 ;;
      *) usage; return 2 ;;
    esac
  done
  [[ "$artifact_commit" =~ ^[0-9a-f]{40}$ ]] || {
    fail 'exact artifact commit is required'; return 2;
  }
  [ -n "$sdk" ] && [ -n "$output" ] || { usage; return 2; }
  source_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)" || {
    fail 'artifact is not in a Git worktree'; return 2;
  }
  [ "$(git -C "$source_root" rev-parse HEAD)" = "$artifact_commit" ] || {
    fail 'artifact worktree HEAD differs'; return 2;
  }
  actual_parent="$(git -C "$source_root" rev-parse HEAD^ 2>/dev/null)" || {
    fail 'artifact commit has no source parent'; return 2;
  }
  [ "$actual_parent" = "$EXPECTED_PARENT" ] || {
    fail 'artifact source parent differs'; return 2;
  }
  if ! git -C "$source_root" diff --quiet HEAD -- .gitattributes "$EXPECTED_REL" ||
      [ -n "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]; then
    fail 'artifact worktree is dirty'; return 2
  fi
  if git -C "$source_root" diff-tree --no-commit-id --name-only -r HEAD | \
      awk -v rel="$EXPECTED_REL/" '$0 != ".gitattributes" && index($0, rel) != 1 {bad=1} END {exit bad}'; then
    :
  else
    fail 'artifact commit changes files outside its payload'; return 2
  fi
  bundle_root="$source_root/$EXPECTED_REL"
  bundle="$bundle_root/bundle"
  [ "$(sha256sum "$bundle/manifest.json" | awk '{print $1}')" = "$EXPECTED_MANIFEST" ] || {
    fail 'bundle manifest hash differs'; return 2;
  }
  python3 -B - "$bundle_root/artifact.json" "$source_root" "$bundle_root" \
      "$EXPECTED_PARENT" "$EXPECTED_BRANCH" "$EXPECTED_REL" "$EXPECTED_MANIFEST" <<'PY' || {
import hashlib
import json
import pathlib
import subprocess
import sys

meta_path, source, root = map(pathlib.Path, sys.argv[1:4])
parent, branch, rel, manifest_hash = sys.argv[4:]
meta = json.loads(meta_path.read_text())
assert meta == {
    "schema": "quactlize.m8-epilogue-perf-ab-artifact.v1",
    "source_parent": parent,
    "artifact_branch": branch,
    "artifact_rel": rel,
    "bundle_manifest_sha256": manifest_hash,
    "source_tools": {
        "tools/analyze_m8_epilogue_perf_ab.py": {
            "git_blob": "7bf9232d090b4e76041a384439d6332a69e9db46",
            "sha256": "8fd53bf5cb61cc0bbe88c70392bfee1c90bad2cacfb362e46b62257889acd300",
        },
        "tools/run_m8_epilogue_perf_ab_prebuilt_box.sh": {
            "git_blob": "cf9531671f7b2fa59fbfb8464f76423567535d5a",
            "sha256": "1dbf1f17ccb5cdbfe8791dc0de357c7003828f5a9038a2e9d9703e97700f5660",
        },
    },
    "payloads": {
        "bundle/bin/baseline": {
            "size": 4877888,
            "sha256": "411e74f2bdf2728266f0dabebec69d9786159ecfa60aef553a534b1556ba5556",
        },
        "bundle/bin/candidate": {
            "size": 4868640,
            "sha256": "3d7e20e87a9edb564b6524e4a63b189235ab4ca8350d5b016d75f1cbfc1e11f4",
        },
    },
}
for name, row in meta["source_tools"].items():
    path = source / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    blob = subprocess.check_output(
        ["git", "-C", str(source), "hash-object", str(path)], text=True).strip()
    assert blob == row["git_blob"]
for name, row in meta["payloads"].items():
    path = root / name
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == row["size"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
PY
    fail 'artifact metadata/payload binding differs'; return 2
  }
  M8_EPILOGUE_PERF_AB_BUNDLE="$bundle" PPU_SDK="$sdk" OUT="$output" \
    bash "$source_root/tools/run_m8_epilogue_perf_ab_prebuilt_box.sh"
}

main "$@"
