# TM8 second-tile A-descriptor A/B

This compile-free bundle compares the historical TM8 A descriptor with one
coordinate-normalized form.  Both Xplane and canonical Q4 K-pack arms come
from clean source `b4436c9befcc3387e371e38c58852c8f8ce768cf`; the only device
definition difference is `PPU_M8_A_GMEM_TILE_REBASE=1`.

Run on one PPU0010 device:

```bash
git fetch origin artifacts/ppu0010/b4436c9-m8-second-tile-rebase-ab
git worktree add --detach /workspace/quactlize-m8-rebase-ab FETCH_HEAD
git -C /workspace/quactlize-m8-rebase-ab lfs pull

CUDA_VISIBLE_DEVICES=0 bash \
  /workspace/quactlize-m8-rebase-ab/prebuilt/ppu0010/b4436c9/m8-second-tile-rebase-ab/run-prebuilt.sh \
  --ppu-sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --output /workspace/quactlize-m8-rebase-result
```

The runner always returns zero and prints one terminal
`M8_SECOND_TILE_REBASE_AB verdict=...` line.  It never builds source and does
not terminate the invoking container shell on a failed diagnostic.
