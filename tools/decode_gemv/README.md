# Decode GEMV / small-M investigation tooling (ZW610)

Snapshot of the scripts used for the decode GEMV study in `docs/decode_gemv/`. They are
experiment runners, not part of the product build: nothing under `quactlize/`, `ci/` or the
package build imports them, and they import only `quactlize.runtime.native`, `ci.support.*`,
`reference.gguf_kpack` and `gguf`.

Layout they expect (`env.sh` sets it): an investigation root `QGEMV_ROOT` with this directory
as `$QGEMV_ROOT/tools`, the QUactlize worktree as `$QGEMV_ROOT/source`, the workload manifest as
`$QGEMV_ROOT/inventory/workload_manifest.json`, results under `$QGEMV_ROOT/runs` and
`$QGEMV_ROOT/profile`, scratch in `/dev/shm/qgemv`. Edit `env.sh` (SDK path, `QGEMV_ROOT`)
before use; the scripts were run with SDK 2.2.1-184ee6 (cuda-12.9 variant).

| purpose | entry |
|---|---|
| workload manifest from GGUF headers (range requests, no weights) | `fetch_gguf_headers.py`, `make_manifest.py`, `make_cases.py` |
| cold-weight benchmark through the public dense/MoE ABI, `--check` against gguf-py | `qgemv_bench.py` |
| Asys capture with idle-card wait, contamination check and alternating arms | `ab_rounds.py`, `parse_asys.py`, `summarize.py`, `device_idle.py`, `when_idle.sh` |
| MMA-reader prototypes (`quactlize/execution/mma_reader.cuh`, `mma_staged.cuh`) | `build_mma_proto.sh` (`STAGED=1` for the staged reader), `mma_proto_tu.cu`, `mma_staged_tu.cu`, `mma_proto_router.cpp`, `make_proto_cases.py`, `screen_report.py`, `resources.py` |
| access-pattern probes (step 1) | `pattern_probe.cu`, `pattern_sweep.sh`, `pattern_asys.py`, `pattern_acu.sh`, `acu_summary.py` |
| device probes | `bw_probe.cu` (streaming read), `mma_probe.cu` (m8n16k16 fragment maps) |
| optional actlize extension, compile only | `check_extension.py --sdk "$PPU_SDK"` (portable; does not source `env.sh`, no GPU launches) |
| extension device retest | `retest_extension.py --sdk "$PPU_SDK" --device 1` (46 direct + 29 staged; packer-only build, GGUF oracle, graph smoke, no timing/sweep) |

Typical use:

```bash
source tools/env.sh
bash tools/build_mma_proto.sh /dev/shm/qgemv/proto/libqgemv_mma.so
python3 tools/qgemv_bench.py --package PKG --cases CASES.json --out out.jsonl --check   # correctness
python3 tools/ab_rounds.py --arm base=PKG --cases CASES.json --out runs/X --device 0,1 --rounds 3
bash tools/pattern_sweep.sh 0,1 profile/pattern/sweepN CONFIGS.txt                       # probes
```
