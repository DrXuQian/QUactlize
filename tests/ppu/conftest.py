import ctypes as C
import os
from pathlib import Path

import pytest

from quactlize.runtime.native import SDK, sdk_identity
from ci.support.packing import bind, device_identity
from ci.support.graph_api import graph_bind


@pytest.fixture(scope="session")
def package():
    path = os.environ.get("QUACTLIZE_CI_PACKAGE")
    if not path:
        from quactlize import get_library_path, get_library_info
        root = get_library_path().parent
        info = get_library_info()
        sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
        release = (sdk / 'release.yaml').read_text().strip().removeprefix('version:').strip()
        if release != info['sdk_requirements']['build_sdk']:
            pytest.fail('installed wheel device gate requires the recorded build SDK release')
        return root
    from tools.build_ppu_library import verify
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
    bind(sdk, C.CDLL(str(package / "libncp_quactlize.so"), mode=C.RTLD_LOCAL))
    return sdk
