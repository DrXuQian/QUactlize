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
    sdk = Path(os.environ.get("PPU_SDK") or "/usr/local/PPU_SDK").resolve()
    if not path:
        from quactlize import get_library_info
        info = get_library_info()
        root = Path(info['library_path']).parent
        expected = info['sdk_requirements']['build_sdk']
    else:
        from tools.build_ppu_library import verify
        root = Path(path).resolve(strict=True)
        manifest = verify(root)  # Real hashes/exports/JIT closure; no fake device libraries.
        expected = manifest.get('build_info', {}).get('sdk_release')
        if not expected:
            # Old manifests have no runtime release contract; retain their exact check.
            current = sdk_identity(sdk)
            if current != manifest["sdk"]:
                pytest.fail(f"public package SDK fingerprint differs: built={manifest['sdk']} "
                            f"selected={current} PPU_SDK={sdk}; rebuild this legacy package")
            return root
    # Match the installed-wheel contract for explicit package paths too.
    # The build fingerprint includes hgcc/hgobjdump: host ELF loader/RPATH
    # adjustments change it without changing the declared runtime release.
    release = (sdk / 'release.yaml').read_text().strip().removeprefix('version:').strip()
    if release != expected:
        pytest.fail(f"public package SDK release differs: built={expected!r} selected={release!r} "
                    f"PPU_SDK={sdk} package={root}; select the matching SDK or rebuild the package")
    print(f"QUACTLIZE_CI_PACKAGE path={root} SDK={release} PPU_SDK={sdk}", flush=True)
    return root


@pytest.fixture(scope="session")
def sdk(package):
    sdk = SDK(Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")))
    print("QUACTLIZE_CI_DEVICE", device_identity(sdk), flush=True)
    graph_bind(sdk)
    # Also declares stream, event and memset functions used by the harness.
    bind(sdk, C.CDLL(str(package / "libncp_quactlize.so"), mode=C.RTLD_LOCAL))
    return sdk
