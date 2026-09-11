"""Host checks for the development-only historical Xplane comparison."""

import hashlib
import json
from pathlib import Path

import pytest

from dev.gemv_cuda.compare_xplane import n2_authority


@pytest.mark.parametrize("plant", (None, "reader", "hash", "columns", "name"))
def test_n2_comparison_binds_the_actual_reader(tmp_path, plant):
    library = tmp_path / "libkpack_gemv_cuda.so"
    library.write_bytes(b"test fixture, not an ELF")
    manifest = dict(reader="cuda-n2", library=library.name,
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    pair_column_values_per_thread=2)
    if plant == "reader":
        manifest["reader"] = "production"
    elif plant == "hash":
        manifest["library_sha256"] = "0" * 64
    elif plant == "columns":
        manifest["pair_column_values_per_thread"] = 1
    elif plant == "name":
        manifest["library"] = "some_other_library.so"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if plant:
        with pytest.raises(ValueError, match="manifest-bound N2"):
            n2_authority(library)
    else:
        assert n2_authority(library)["reader"] == "cuda-n2"


def test_historical_arm_uses_the_original_kernel_and_producer():
    root = Path(__file__).resolve().parents[1]
    source = (root / "dev/gemv_cuda/xplane_compare.cu").read_text()
    assert '#include "gguf_bc_q4_gemv.hpp"' in source
    assert "gguf_scale::bc_q4_gemv::launch<C,W,1>" in source
    assert "xplane::place_derived<4,64,64,64,32,32,1,64>" in source
    assert "xplane::recover_derived<4,64,64,64,32,32,1,64>" in source
    assert "__global__" not in source
    assert "cudaDevAttrL2CacheSize" in source
