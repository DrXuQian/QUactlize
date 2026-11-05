# PPU wheel

Build on a machine with a complete PPU SDK, the pinned actlize checkout, Python,
setuptools, wheel, and a C++ toolchain. A device is not required for compilation.
Use the platform's existing Torch; do not install a generic CUDA Torch to build
or test this package. The default PPU wheel uses ctypes and does not link Torch.

```bash
bash build.sh --wheel --sdk /usr/local/PPU_SDK --output dist --jobs 8
```

This runs the existing complete public-library builder, then creates a
non-editable `quactlize-0.2.0-py3-none-linux_x86_64.whl`. No device code is built
during wheel installation. `--bundle /path/to/package` reuses a complete bundle
only after checking all payload hashes, current Python/native source hashes,
actual ELF exports, private libraries, and relative loader paths. Older bundles
without build provenance must be rebuilt. `--work /new/path` retains build logs
and intermediates for diagnostics. No source checkout/submodule fetch occurs.

The current builder emits **ppu10** (`-arch=ppu_10`), not ppu15, ppu17 or NVIDIA.
Publish this wheel only in the corresponding PPU artifacts channel. Linux's pip
platform tag does not encode accelerator generation. The SDK release, dependency
SONAMEs, source revision/content hashes and device-validation status are recorded
in `quactlize/_native/wheel.json` and `manifest.json`. SDK/driver libraries and
libc are not bundled. Install the matching platform runtime before loading the
DSO; the default Flash-Next path is AOT, while other existing APIs may use the
bundled JIT source closure during their explicit preparation stage.

To retain the optional legacy `_C` host preprocessing ops as well, add
`--with-host` **on the destination platform Torch build environment**. That wheel
gets a CPython/ABI-specific tag. The default C-ABI wheel keeps Python entrypoints
but does not contain `_C`; those legacy ops require this optional build. Import,
discovery, packing ABI declarations and native16/serving do not load `_C`.

## Installation and framework API

After the PPU artifact channel is configured and the wheel is published there:

```bash
python -m pip install --only-binary=quactlize quactlize==0.2.0
# Qualification of an unpublished local artifact, without changing platform dependencies:
python -m pip install --no-deps /path/to/quactlize-0.2.0-py3-none-linux_x86_64.whl
```

```python
import quactlize
path = quactlize.get_library_path(target="ppu10")  # absolute pathlib.Path
info = quactlize.get_library_info(target="ppu10")  # JSON-serializable dict

from quactlize.compat.native import Library
native = Library(target="ppu10")
q5_1 = Library(target="ppu10", packed_route=0)     # packed AUTO, unchanged arithmetic
# Library("/explicit/developer/library.so") remains supported; do not combine path and target.

from quactlize.compat.serving_native import Library as ServingLibrary
helpers = ServingLibrary(target="ppu10")         # bind only, no prepare/device allocation
from quactlize.serving import Serving
# serving = Serving(device=0, stream=explicit_stream, target="ppu10")
# Construct outside graph capture; retain it, buffers and prepared plans through replay.
```

Signatures: `get_library_path(target=None)`, `get_library_info(target=None)`,
`Library(path=None, *, target=None, packed_route=None)`,
`ServingLibrary(path=None, *, target=None)`,
`Serving(library=None, device=0, stream=None, *, target=None)`.
Omitted target uses the package's explicit default only. Discovery reads files,
never probes hardware, dlopens a DSO, compiles, or creates a CUDA context. Wrong
target, missing resources and inconsistent manifests raise a path-specific error;
there is no search of environment variables, another checkout or system DSOs.
Flare must still bind the actual DSO content identity and validate its ABI.

## Installed-wheel qualification

Use a **non-editable** isolated environment with platform Torch and test
dependencies already available. Run outside the checkout with `PYTHONPATH`
unset. The following script never inserts production source into `sys.path`:

```bash
python -I /path/to/quactlize/tools/verify_installed_wheel.py --target ppu10
```

It verifies all required Python modules are under the environment prefix, no
device initialization during import/discovery, public symbols, exact plane sizes,
and field-wise descriptor rejection for E512/top10 Q5_1 down, Q8 down/gate-up,
Q4_K/Q5_K gate-up at M1/8/9/128 in both native dtypes. It does not allocate device
memory. Repeat in a second installation prefix; `--deny-root SOURCE --deny-root
BUILD_OUTPUT` denies Python file access to the original source/build directories.
Do not remove user repositories to simulate isolation.

On a real PPU, using the SDK release recorded by the wheel:

```bash
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  python -I /path/to/quactlize/tools/verify_installed_wheel.py --target ppu10 \
  --device-tests --test-source /path/to/quactlize \
  --flare /path/to/spark-local-inference
```

Only test/oracle files are copied to a temporary directory; production Python and
DSOs come from the installed wheel. This runs the existing canonical packing,
native16/changing-ID/shared-scratch/graph tests and serving router/SwiGLU/row/merge
tests without changing tolerances. For Flare use `feat/qwen38-gguf >= 3f2083b`;
its `tests/preparation/test_gguf_packing_cuda.py` gate receives
`FLARE_TEST_PPU_MATMUL_LIBRARY` from package discovery. Install Flare and its test
dependencies first. Omitting `--flare` does not qualify Flare integration.

Report host packaging/queries, PPU numerical/graph tests and performance
separately. No PPU device means device tests remain **NOT RUN**, regardless of
successful compilation, host fixtures or installation. This task changes no
kernel layouts, rounding, tolerance, route selection or MTP behavior.
