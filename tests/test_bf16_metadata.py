import os
from pathlib import Path
import subprocess
import ctypes as C

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_direct_bf16_metadata_and_full_dequant(tmp_path):
    sdk = Path(os.environ.get("PPU_SDK", "/root/ppu-sdk/2.1.1"))
    exe = tmp_path / "bf16-metadata"
    command = ["g++", "-O2", "-fno-strict-aliasing", "-ffp-contract=off", "-std=c++17",
               f"-I{ROOT}", f"-I{ROOT}/quactlize/include", f"-I{ROOT}/third_party/actlize/include",
               f"-I{sdk}/include", f"-I{sdk}/targets/x86_64-linux/include",
               str(ROOT / "tests/bf16_metadata_host.cpp"), "-o", str(exe)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([exe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BF16_METADATA_HOST PASS" in result.stdout


def test_metadata_abi_layout(tmp_path):
    from quactlize.decode.native_compute import GroupedMetadataCall, ComputeMetadataIdentity
    from quactlize.dequant.native import TypedCall
    source, binary = tmp_path / "sizes.cpp", tmp_path / "sizes"
    source.write_text('#include "quactlize/runtime/abi.h"\n#include "quactlize/dequant/api.h"\n'
        '#include <cstdio>\nint main(){printf("%zu %zu %zu",sizeof(qk_compute_device_call_v4),'
        'sizeof(qk_compute_identity_v4),sizeof(qzd_call_v2));}')
    subprocess.run(["g++","-std=c++17",f"-I{ROOT}",str(source),"-o",str(binary)],check=True)
    assert list(map(int,subprocess.check_output([binary],text=True).split())) == [
        C.sizeof(GroupedMetadataCall),C.sizeof(ComputeMetadataIdentity),C.sizeof(TypedCall)]


@pytest.mark.parametrize("q", range(10,15))
def test_device_metadata_fixture_rejects_decoded_half(q):
    from dev.bf16_compute.check_metadata import fixture
    from dev.bf16_compute.fixture import bf16_bits, bf16_float
    _, typed, full, half = fixture(q,n=256,k=512,experts=1)
    assert typed.shape == half.shape
    assert np.isfinite(bf16_float(typed)).all() and np.isfinite(bf16_float(full)).all()
    assert np.any(typed[0] != bf16_bits(half[0].view("<f2").astype("f4")))


def test_matched_prefill_inventory_has_identical_geometry():
    from dev.bf16_compute.matched import parents
    from quactlize.runtime.compiler import validate_parent
    inventory = parents()
    assert len(inventory) == 8 and len({key for key, _, _ in inventory}) == 8
    pairs = {}
    for key, compute, parent in inventory:
        validate_parent(parent)
        assert parent["ap"] == 0 and parent["symbol"] == key
        assert (parent["tm"], parent["tn"], parent["wm"], parent["wn"], parent["stages"], parent["dn"]) == (64,128,64,32,2,16)
        pairs.setdefault((parent["qtype"], parent["route"]), {})[compute] = {
            field: value for field, value in parent.items() if field != "symbol"}
    assert set(pairs) == {(q, route) for q in (12,13) for route in ("fq-grouped", "sf-grouped")}
    for pair in pairs.values():
        assert pair["f16"] == pair["bf16"]
