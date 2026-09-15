"""Compile-only typed decode endpoints for an existing selected TC parent.

The parent/configuration stays owned by the production selector. Endpoint
storage is a distinct module identity; old FP16 modules are never overwritten.
"""

from pathlib import Path

from quactlize.runtime.compiler import Compiler, ROOT, sha, validate_parent, source_contract
from quactlize.runtime.tuning import digest


class DecodeCompiler(Compiler):
    def __init__(self, sdk, cache, jobs=1, compute_type="f16"):
        if compute_type not in ("f16", "bf16"):
            raise ValueError("decode compute must be f16 or bf16")
        self.compute_type = compute_type
        super().__init__(sdk, cache, jobs)
        base_source = source_contract(self.identity)
        directory = ROOT / "quactlize/decode"
        paths = sorted(p for p in directory.iterdir() if p.suffix in (".h", ".cuh"))
        self.identity["kernel"] = digest({
            "base": self.kernel_identity,
            "decode": [(str(p.relative_to(ROOT)), sha(p)) for p in paths],
            "compute_type": compute_type,
        })
        self.identity["generator"] = digest({
            "base": self.identity["generator"], "decode": sha(__file__),
        })
        self.identity["endpoints"] = "decode-m1-8-f32-bf16-v1"
        self.identity["compute_type"] = compute_type
        self.identity["base_source_contract"] = base_source
        for path in paths + [Path(__file__)]:
            self.input_stats[path] = (path.stat().st_mtime_ns, path.stat().st_size)

    def source(self, parent, key):
        # Source-only audits call this unbound for the legacy F16 generator.
        # A BF16 build always has an explicit compiler instance and identity.
        compute_type = getattr(self, "compute_type", "f16")
        validate_parent(parent)
        if not parent["route"].endswith("dense"):
            raise ValueError("typed dense decode module requires a dense parent")
        if compute_type == "bf16" and parent["ap"]:
            raise ValueError("BF16 packed-A provider is not admitted")
        return f"#define QKD_USE_BF16_COMPUTE {int(compute_type == 'bf16')}\n" + Compiler.source(self, parent, key).replace(
            '#include "module.cuh"', '#include "dense.cuh"')
