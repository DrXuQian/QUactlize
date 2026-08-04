# HANDOFF — wire quactlize's k-quant GEMV into llama.cpp (ggml-cuda)

**For:** a Claude with no context on this project.
**Patch:** `/root/quactlize-kquant-gemv.patch` (183 lines).
**Status:** written and reviewed against the .so's real ABI. **NEVER COMPILED.** Assume it does not build until
you have built it. Everything below that is a claim rather than a check is marked.

---

## 1. What this does, in one paragraph

llama.cpp loads a GGUF checkpoint and keeps the quantised blocks in VRAM in their native layout. For a
single-token decode (`M=1`) of a k-quant tensor, ggml routes to `mul_mat_vec_q`. This patch inserts, ahead of
that, a call into an external `libquactlize_ppu.so` that reads **those same raw GGUF blocks** — no repack, no
checkpoint conversion, no extra VRAM. If the .so is absent or declines, ggml's own kernel runs unchanged.

The five formats it covers are the k-quants: **Q2_K, Q3_K, Q4_K, Q5_K, Q6_K** (ggml types 10–14).

---

## 2. What you are touching

| file | state | lines |
|---|---|---|
| `ggml/src/ggml-cuda/ppu-quactlize-so.h` | NEW | 43 |
| `ggml/src/ggml-cuda/mmvq-quactlize-ppu.cuh` | NEW | 40 |
| `ggml/src/ggml-cuda/mmvq-quactlize-ppu.cu` | NEW | 43 |
| `ggml/src/ggml-cuda/ppu-so.cu` | modified | +45 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | modified | +12 |

**`CMakeLists.txt` needs NO change** — `ggml/src/ggml-cuda/CMakeLists.txt:105` is `file(GLOB GGML_SOURCES_CUDA
"*.cu")`, so the new `.cu` is picked up automatically. This has a consequence you must respect: **the new file
compiles in EVERY cuda build, including plain NVIDIA ones with no PPU anywhere.** It stays linkable because
`ppu-so.cu:194` has an `#else` branch of inert stubs, so `ggml_ppu_quactlize_*` always resolves. Do not add an
`#ifdef` that removes those stubs.

---

## 3. Apply

```bash
cd /root/llama.cpp            # or wherever your checkout is
git checkout -b quactlize-kquant-gemv
git apply --check /root/quactlize-kquant-gemv.patch    # must print nothing
git apply /root/quactlize-kquant-gemv.patch
```

If `--check` complains, the base has moved. The patch was cut against `be37a0d89`. Re-cut rather than
force-apply: the `ggml-cuda.cu` hunk lands inside a specific `else if` chain and a fuzzy match can put it in the
wrong branch, which fails **silently** (the route never fires, and everything still works via ggml).

---

## 4. Build

Two builds matter, and the first is the one people forget.

### 4a. The build that proves you broke nothing

```bash
cmake -B build-plain -DGGML_CUDA=ON
cmake --build build-plain -j$(nproc) --target ggml-cuda
```

This has `GGML_PPU_SO` **off** and `GGML_USE_PPU` **off**. It must succeed. It is the only check that the
globbed-in file does not break every ordinary CUDA user.

### 4b. The build that actually enables the route

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_PPU_SO=ON -DGGML_USE_PPU=ON
cmake --build build -j$(nproc)
```

**Both flags are required and they do different things**, which is a real trap:

- `GGML_PPU_SO` (cmake option, `CMakeLists.txt:165`) → compiles the dlopen loader body and links `${CMAKE_DL_LIBS}`
- `GGML_USE_PPU` → guards the **dispatch site** in `ggml-cuda.cu:2643`

With only the first, the .so loads and nothing ever calls it. With only the second, the call site exists and
reaches inert stubs that return `-1`. Neither combination errors. **Verify both are actually on the compile
line** rather than trusting that you passed them:

```bash
grep -c "GGML_PPU_SO" build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make    # expect >= 1
nm -C build/bin/libggml-cuda.so | grep quactlize                                       # expect the symbols
```

---

## 5. The .so it dlopens

Nothing in llama.cpp builds it. It comes from the quactlize repo and is built on the PPU box:

```bash
export GGML_PPU_QUACTLIZE_SO=/path/to/libquactlize_ppu.so   # else it tries "libquactlize_ppu.so" via ld.so
```

The loader (`ppu-so.cu:88-97`) resolves **two** symbols and is deliberately all-or-nothing: if either
`quactlize_ppu_vecdot_dense` or `quactlize_ppu_vecdot_moe` is missing it clears **both** pointers and reports
unavailable. A half-loaded library that serves dense and aborts on MoE is worse than no library.

The ABI, verified against `quactlize/csrc/device/ppu_backend.cu:233,242` — if you change one side, change both:

```c
int quactlize_ppu_vecdot_dense(const uint8_t *b, int64_t block_bytes, const uint16_t *x, float *out,
                               int rows, int bpr, int qtype);
int quactlize_ppu_vecdot_moe  (const uint8_t *b, int64_t block_bytes, const uint16_t *x, const int *offsets,
                               float *out, int n, int bpr, int experts, int total_rows, int max_rows, int qtype);
```

`qtype` is the **ggml type enum** (10=Q2_K … 14=Q6_K), passed straight through. `x` is `uint16_t*` holding
**fp16** — ggml's activation is f32, so the patch converts it (`quactlize_f32_to_f16` in the new `.cu`). That
conversion is a real kernel launch on every call; if you are profiling and see an unexplained small kernel, it
is that.

---

## 6. How to tell it actually ran

**This is the part that matters.** The route can decline for a dozen reasons and a declining route looks exactly
like a working one — the model still produces correct output, because ggml's kernel takes over.

The patch therefore never declines silently. `ggml_cuda_mmvq_quactlize_why()` returns `NULL` to accept, or a
**string naming the reason** to decline, and `ggml_ppu_route_log()` (`ppu-route.cuh:35`) prints it. The switch is
`GGML_PPU_ROUTE` — note that it tests the VALUE, not presence, so `GGML_PPU_ROUTE=0` correctly means off. (An
earlier switch in this file tested `getenv() != nullptr`, so `=0` turned the path ON and cost someone a bisect;
the comment at `ppu-route.cuh:27` is that scar.)

```bash
GGML_PPU_ROUTE=1 GGML_PPU_QUACTLIZE_SO=/path/to/libquactlize_ppu.so \
  ./build/bin/llama-cli -m model.gguf -p "hello" -n 8 2>&1 | grep quactlize
```

- lines saying `quactlize-kquant-gemv` with **no** reason → it ran
- lines with a reason (`not a k-quant`, `library unavailable`, `M != 1`, …) → it declined, and the reason is the
  next thing to fix
- **no lines at all** → the dispatch site was never compiled. `GGML_USE_PPU` is off, or the `ggml-cuda.cu` hunk
  landed in the wrong branch. This is the failure mode the fuzzy-apply warning in §3 is about.

### Correctness

```bash
./build/bin/test-backend-ops -o MUL_MAT   # the standard oracle; k-quant M=1 cases go through the route
```

`rc != 0` from the .so is a **`GGML_ABORT`, not a fallback**, and that is deliberate: by the time it returns,
`dst` has not been written, so falling through would feed the model uninitialised VRAM. A crash is the correct
outcome of a wrong answer.

---

## 7. What is NOT done — do not report these as finished

1. **Never compiled.** Neither build in §4 has been run. Expect ordinary first-compile errors.
2. **The MoE arm is wired but unexercised.** `ggml_ppu_quactlize_vecdot_moe` is loaded and forwarded, but the
   ggml side only calls the **dense** path today; nothing constructs the `offsets` array from a
   `mul_mat_id`. MoE decode still runs on ggml.
3. **No dispatch threshold.** The route takes every k-quant `M==1` case. There is no measurement behind that
   choice on this hardware — it is where `use_mul_mat_vec_q` already sits, not a tuned crossover.
4. **No perf number in llama.cpp.** The kernel is measured in its own repo; end-to-end tok/s here is unmeasured.
   Do not quote a speedup.

---

## 8. If you need to go deeper

The kernel side lives in `github.com/DrXuQian/quactlize` (branch `develop`). Relevant files:

- `quactlize/csrc/device/ppu_backend.cu` — the C ABI above, and the GEMV behind it
- `quactlize/routes.py` — the four routes and how M selects among them
- `.coord/INBOX.md` — the running work log with the kernel author

Report back: which of the two builds succeeded, the exact `grep quactlize` output from §6, and the
`test-backend-ops` result. If a build failed, the first error verbatim — not a summary of it.
