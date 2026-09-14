#!/usr/bin/env bash
# Fetch the pinned repair through Git LFS; never rebuild on the box.
(
    set -Eeuo pipefail
    trap 'rc=$?; printf "decode_repair_launcher_rc=%s\nCurrent Docker shell is preserved.\n" "$rc"' EXIT
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_decode_io_ppu_box.sh"
    cd "$ROOT"
    SHA=11d0f34be65b8c61997b47f4ca6000e6da8c489b
    BRANCH=artifacts/kpack-decode-io-device-v2
    DEST_ROOT=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$DEST_ROOT" && test -d "$DEST_ROOT"
    ART="$DEST_ROOT/quactlize-decode-artifact-11d0f34"
    git fetch origin "$BRANCH"
    git cat-file -e "$SHA^{commit}"
    if [[ -e "$ART" ]]; then
        test -d "$ART" && test "$(git -C "$ART" rev-parse HEAD)" == "$SHA"
    else
        GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "$SHA"
    fi
    git -C "$ART" lfs pull origin --include="prebuilt/ppu0010/kpack-decode-io-device-v2/**" --exclude=""
    git lfs pull origin --include="prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so" --exclude=""
    export BUNDLE="$ART/prebuilt/ppu0010/kpack-decode-io-device-v2"
    export FETCH_PAYLOADS=0
    printf 'Typed decode repair fetched: %s\ncompile=NONE jit=NONE\n' "$SHA"
    bash "$ROOT/tools/run_kpack_decode_io_ppu_box.sh"
)
