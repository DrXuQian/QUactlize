"""Explicit-compute grouped modules; canonical artifact bytes are unchanged."""
from pathlib import Path
from quactlize.runtime.compiler import Compiler, sha, source_contract, validate_parent
from quactlize.runtime.tuning import digest


class GroupedComputeCompiler(Compiler):
    def __init__(self, sdk, cache, jobs=1, compute_type="bf16"):
        if compute_type not in ("f16", "bf16"):
            raise ValueError("grouped compute must be f16 or bf16")
        super().__init__(sdk, cache, jobs)
        self.compute_type = compute_type
        self.identity["base_source_contract"] = source_contract(self.identity)
        self.identity["kernel"] = digest({"base": self.kernel_identity, "compute_type": compute_type})
        self.identity["generator"] = digest({"base": self.identity["generator"], "grouped_compute": sha(__file__)})
        self.identity["compute_type"] = compute_type
        self.identity["endpoints"] = "grouped-explicit-compute-v3"
        path = Path(__file__)
        self.input_stats[path] = (path.stat().st_mtime_ns, path.stat().st_size)

    def source(self, parent, key):
        validate_parent(parent)
        if not parent["route"].endswith("grouped"):
            raise ValueError("explicit-compute grouped module requires a grouped parent")
        return f"#define QK_USE_BF16_COMPUTE {int(self.compute_type == 'bf16')}\n" + super().source(parent, key)
