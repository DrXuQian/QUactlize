"""Experiment axes over the same SIMT inventory compiled by production."""
from quactlize.execution.simt_codegen import (
    Config, QTYPES, SPLITS, inventory, runtime_inventory, source,
)

SCHEMA = "quactlize.simt-register-reuse.v1"


def sweep_workloads():
    # Keep the historical Q4 registry and receipts intact.
    from dev.gemv_ppu.decode_sweep import workloads
    return [w | {"qtype": q, "id": f"q{q}-" + w["id"]}
            for q in QTYPES for w in workloads()]


def plan(profile="full"):
    return dict(schema=SCHEMA, profile=profile, formats=list(QTYPES),
                configs={str(q): [c.record() for c in runtime_inventory(q, profile)] for q in QTYPES},
                pruned=["C*P>32: warp reduction/metadata ownership",
                        "Q8 metadata sharing: different K32 scale in each worker"],
                comparison=["NEW_SIMT", "PREVIOUS_SIMT_WINNERS", "CURRENT_TYPED_TC"],
                input_storage=["F16", "F32"], activation_arithmetic="F16_REGISTER_ROUNDING",
                dequant_arithmetic="FP32_GROUP_AFFINE", accumulator="F32", output="F32",
                timing="COMPLETE_CALL_INCLUDING_SPLIT_REDUCER_AND_REQUIRED_ENDPOINTS",
                dense_m=list(range(1, 9)), grouped_tokens=list(range(1, 9)),
                offline_change=False, production_policy_changed=False,
                device_validated=False, performance_admitted=False)
