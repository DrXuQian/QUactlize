# Native K-pack execution package

Development validation package, not a device-admitted release.

- `libquactlize_kpack_dispatch.so`: SDK-free C++ heuristic selection and cached
  module binding (607,400 bytes).
- `libquactlize_ppu_execution.so`: five-format GEMV and GPU scale prepass
  (560,232 bytes), identical to the execution-v1 payload.
- `modules/<build-key>/kernel.so`: 216 selected FQ/ScaleFirst dense/grouped
  parents; complete parent and compiler identities are in `manifest.json`.
- Total DSO payload: 35,866,304 bytes. Binaries are Git LFS objects.

The measured single-token grouped overrides select Q4 N512/K2048 compact
TM8/S4 and Q5 N2048/K512 compact TM8/S1, only for M8/E256/max_rows1.
They use the two already-built postops candidate modules. All 214 previous
module payloads and the execution DSO are unchanged; only the host selector
was rebuilt. Policy value `QKS_MEASURED_GROUPED` identifies these choices.
The independent PPU postops gate passes, but full-adapter model performance
with the new choices still needs measurement.

The package is a selected closure, not a full Cartesian sweep or runtime JIT.
A missing parent returns an explicit miss. The existing six libraries remain
required by llama.cpp for weight intake and canonical K-pack FQ fallback.
No Xplane bytes are introduced by this execution path.

Validation:

```sh
python3 tools/verify_kpack_dispatch.py prebuilt/ppu0010/kpack-native-v1
bash tools/run_kpack_native_box.sh --help
```

The box entry validates selected arithmetic, creates a measured GEMV table,
rebuilds the llama.cpp adapter, checks changing-router graph replay, measures
the real model with ABBA ordering, and captures a separate short device trace.
All prebuilt manifests intentionally retain `device_validated=false` and
`heuristic_admitted=false` until those results have been reviewed.

See `docs/LLAMA_CPP_KPACK_HANDOFF.md` for environment variables and scopes.
