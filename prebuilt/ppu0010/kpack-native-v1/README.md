# Native K-pack execution package

Development validation package, not a device-admitted release.

- `libquactlize_kpack_dispatch.so`: SDK-free C++ heuristic selection and cached
  module binding (607,336 bytes).
- `libquactlize_ppu_execution.so`: five-format GEMV and GPU scale prepass
  (560,232 bytes), identical to the execution-v1 payload.
- `modules/<build-key>/kernel.so`: 214 selected FQ/ScaleFirst dense/grouped
  parents; complete parent and compiler identities are in `manifest.json`.
- Total DSO payload: 35,162,144 bytes. Binaries are Git LFS objects.

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
