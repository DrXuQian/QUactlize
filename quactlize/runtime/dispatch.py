"""Prepare one selected tactic. No online profiling, shortlist or hidden JIT."""

from dataclasses import asdict

from .candidates import PARENT_FIELDS, from_config
from .tuning import UnsupportedTactic, digest


def prepare_selected(backend, request, selection, *, allow_prediction=False):
    """Bind a host selection to its exact prebuilt/cached parent module.

    Load/compile the one parent before this call, outside inference/capture.
    Unknown-shape predictions need separate numerical admission. No alternate
    config or arbitrary compiled default is selected on a rejection. The
    returned handle is reused with backend.run(), and closed by its owner.
    """
    status = selection.get("status")
    allowed = {"MEASURED_RECENT", "MEASURED_HISTORICAL"}
    if allow_prediction:
        allowed.add("HEURISTIC_UNVALIDATED")
    if status not in allowed or digest(selection.get("request")) != digest(
        asdict(request)
    ):
        raise UnsupportedTactic("selection is not admitted for this exact request")
    config = selection["config"]
    tactic = from_config(config)
    module = backend.modules.get(tactic.parent)
    if module is None:
        raise UnsupportedTactic("load/compile the selected parent before preparation")
    parent = {k: config[k] for k in PARENT_FIELDS}
    if (
        module.record["parent"] != parent
        or module.record["identity"] != selection["module_contract"]
    ):
        raise UnsupportedTactic("selected module parent/compiler contract differs")
    if (
        backend.identity["device"] != "PPU-ZW810"
        or backend.identity["compute_units"] != 72
        or any(
            backend.identity[k] != selection["module_contract"][k]
            for k in ("sdk", "kernel")
        )
    ):
        raise UnsupportedTactic("selected module device/SDK/kernel differs")
    if backend.is_capturing():
        raise UnsupportedTactic("prepare selected handles before graph capture")
    # NativeBackend.prepare queries actual resources/occupancy/legality and
    # resolves the grid from request.rows. The historic grid is never reused.
    return backend.prepare(request, tactic)
