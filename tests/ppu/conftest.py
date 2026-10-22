import ctypes as C
import os
from pathlib import Path

import pytest

from quactlize.runtime.native import SDK, sdk_identity
from tools.build_ppu_library import verify
from tools.run_kpack_pack_gate import bind, device_identity
from tools.run_kpack_grouped_device_gate import graph_bind


@pytest.fixture(scope="session")
def package():
    path = os.environ.get("QUACTLIZE_CI_PACKAGE")
    if not path:
        pytest.fail("set QUACTLIZE_CI_PACKAGE or use run_tests.sh --suite accuracy --package PATH")
    root = Path(path).resolve(strict=True)
    manifest = verify(root)  # Real hashes/exports/JIT closure; no fake device libraries.
    current = sdk_identity(Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")))
    if current != manifest["sdk"]:
        pytest.fail("installed public package SDK differs; rebuild with the selected PPU_SDK")
    return root


@pytest.fixture(scope="session")
def sdk(package):
    sdk = SDK(Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")))
    print("QUACTLIZE_CI_DEVICE", device_identity(sdk), flush=True)
    graph_bind(sdk)
    # Also declares stream, event and memset functions used by the harness.
    bind(sdk, C.CDLL(str(package / "libquactlize_ppu.so"), mode=C.RTLD_LOCAL))
    return sdk
