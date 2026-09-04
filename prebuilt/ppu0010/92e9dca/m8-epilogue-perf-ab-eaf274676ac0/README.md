# TM8 epilogue performance A/B

This compile-free PPU0010 bundle compares one exact Q4 K-pack S1 kernel across
the epilogue ownership fix. Both builds use the same parent source tree. The
baseline is a deterministic commit-tree whose only change is the actlize
gitlink (`9d063e4c`); the candidate uses actlize `423253c`.

The measured cell is `M8xN3072xK512`, AP0, S1, TM8/TN64/TK256,
WM8/WN16, DN16. The runner performs six alternating AB/BA rounds with 31
samples per arm per round and seven raw-bit correctness repetitions. It also
requires the exact S1 resource and ISA transition before timing is accepted.

Use a detached artifact worktree and hydrate the two LFS payloads. Pass the
exact artifact commit printed by `git rev-parse` back into the wrapper so the
execution cannot silently move to another branch tip.

```bash
ART_BRANCH=artifacts/ppu0010/92e9dca-m8-epilogue-perf-ab-eaf274676ac0
ART_REL=prebuilt/ppu0010/92e9dca/m8-epilogue-perf-ab-eaf274676ac0
WT=/workspace/quactlize-m8-epilogue-perf-artifact
OUT=/workspace/quactlize-m8-epilogue-perf-result-$(date -u +%Y%m%dT%H%M%SZ)-$$

git fetch origin "$ART_BRANCH"
ART_COMMIT=$(git rev-parse FETCH_HEAD)
git worktree add --detach "$WT" "$ART_COMMIT"
git -C "$WT" lfs pull --include="$ART_REL/bundle/bin/*" --exclude='' origin

CUDA_VISIBLE_DEVICES=0 bash "$WT/$ART_REL/run-prebuilt.sh" \
  --artifact-commit "$ART_COMMIT" \
  --ppu-sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --output "$OUT"
```

Invoke the wrapper with `bash`; do not source it. A clean result ends with
`FQ_M8_EPILOGUE_PERF_AB_GATE verdict=PASS`.
