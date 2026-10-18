# Completed-screen closure

The R26 supplement has 430 complete points, not another unfinished sweep.
R27 adds **observed delivery peers of an already selected exact geometry**.
Donors exclude the entire target N/K family and are not ranked by latency.
Host replay of the reviewed archive produces 1,138 distinct measured
compositions, with zero remaining rule gaps (430 points, at most four calls
each). Some new proposals have not been measured; this confirmation does not
claim global optimality over those proposals or all legal kernels.

The next Box run confirms the best two rule calls, best historical control,
and original wiring, deduplicating identical measured stages. There are six
alternating rounds. TC producer/reducer/adapters and dequant remain separate
single-kernel timers. Dense cuBLAS uses ordinary provider events, including
its internal reduction; it never uses CUDA Graph or Asys. Setup/JIT/first-use
warmup is excluded. No old timings enter the new comparison.

Use the same seven physical cards and SDK as the completed supplement. The
runner resolves PCI identity, not just visible ordinal. Every family remains
on its original card. TC modules are verified and reused, not recompiled;
only the small helper/dequant libraries are built. Do not run other GPU work
on these cards during timing. CUDA Graph does not isolate competing work.

```bash
PREVIOUS_RUN=/workspace/kpack-prefill-supplement-uedmwhxm \
PPU_SDK=/usr/local/PPU_SDK DEVICES=1,2,3,4,5,6,7 IDLE_CONFIRMED=1 \
bash tools/run_kpack_prefill_confirm_box.sh
```

The reviewed archive defaults to
`$PREVIOUS_RUN/review-1792311707684248419.tgz`. Set `REVIEW_ARCHIVE` if it was
moved. Its SHA256 must match the reviewed receipt; source control contains no
uploaded binaries, weights or raw archives. `PLAN_ONLY=1` performs host replay
only. `FINALIST_PLAN` may reuse the generated `finalists.json` from that replay.

Resume an interrupted confirmation with `CONFIRM_RUN=<new run printed above>
RESUME=1` and the same other settings. Complete rounds remain immutable.
Incomplete rounds restart fresh, not from a mixture of partial samples.
Failures leave peer points and later families running in fresh processes; only affected incomplete rounds need
another invocation. There is no 24-hour automatic kill or low-utilization
abort. Ctrl-C stops this runner's worker process groups, never unrelated work.

The small `confirmation-*.tgz` contains selected configs, accuracy summaries,
raw component samples and bounded error logs. Raw run directories remain on
Box. `PASS` means no-regressing component sums among the measured finalists,
not a model performance admission. A regression/unstable comparison has no
winner; it never installs the old selector as a hidden fallback. Production
table promotion, actual selected-call validation and warmed native/model A/B
are separate gates. Inference contains no online tuning.
