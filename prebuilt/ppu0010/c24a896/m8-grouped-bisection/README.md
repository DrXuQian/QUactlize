# Grouped Q4 K-pack TM8 bisection

This compile-free diagnostic crosses the exact failing K-pack geometry over
logical TM8/TM16 and nonpersistent/persistent schedulers.  Each binary contains
only two generated parent rows.  Raw-FP16 correctness is checked before timing,
and a mismatch reports expert/local-M/N coordinates plus M-tile and N16 cohort
histograms.

Run on one PPU0010 device:

```bash
git fetch origin artifacts/ppu0010/c24a896-m8-grouped-bisection
git worktree add --detach /workspace/quactlize-m8-grouped-bisection FETCH_HEAD
git -C /workspace/quactlize-m8-grouped-bisection lfs pull

CUDA_VISIBLE_DEVICES=0 bash \
  /workspace/quactlize-m8-grouped-bisection/prebuilt/ppu0010/c24a896/m8-grouped-bisection/run-prebuilt.sh \
  --ppu-sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --output /workspace/quactlize-m8-grouped-result
```

An existing output base is safe: every invocation creates a fresh child run.
The runner never builds source and always returns zero after printing a terminal
diagnostic line, so a red kernel cannot terminate an interactive container.
