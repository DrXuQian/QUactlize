import pytest
from tools.export_kpack_prefill_policy import export


def fixture():
    rows = []
    items = [(q, 1024, 5120, 1) for q in range(10, 15)] + [
        (12, 512, 2048, 256),
        (13, 2048, 512, 256),
    ]
    for q, n, k, e in items:
        for t in (1, 128):
            for sf in (0, 1):
                m = t if e == 1 else t * 8
                rows.append(
                    dict(
                        q=q,
                        n=n,
                        k=k,
                        experts=e,
                        m=m,
                        max_rows=t,
                        route=(0 if e == 1 else 2) + sf,
                        status="PASS",
                        graph_replays=3,
                        rows_host=False,
                        rows_device=False,
                        sf_metadata_mode="PER_CALL_GPU_PREPASS" if sf else "PACKED_UNITS",
                        profiles=[
                            dict(
                                rows=[m] if e == 1 else [t] * 8 + [0] * (e - 8),
                                error=0.0001,
                                samples_us=[8.0 if sf else 10.0] * 5,
                            )
                            for _ in range(3)
                        ],
                        prepass_samples_us=[5.0] * 3,
                        selection=dict(parent=f"p{sf}"),
                    )
                )
    return dict(status="PASS", results=rows, failures=[])


def test_prefill_choices_use_complete_call_not_amortization():
    text, records = export(fixture())
    assert len(text.splitlines()) == 15
    assert all(
        r["selected"] == "SF" and "kernel_break_even_calls" not in r for r in records
    )
    assert text.splitlines()[0] == "KPACK_PREFILL_POLICY_V2_PER_CALL"


def test_old_resident_timings_cannot_be_silently_reused():
    s = fixture()
    for r in s["results"]:
        r.pop("sf_metadata_mode")
    with pytest.raises(ValueError, match="resident-only"):
        export(s)


def test_prepass_is_in_sample_not_added_again():
    s = fixture()
    for r in s["results"]:
        if r["route"] % 2:
            for p in r["profiles"]:
                p["samples_us"] = [13.0]*5  # Core 8 + prepass 5, FQ is 10.
    assert all(r["selected"] == "FQ" for r in export(s)[1])


def test_marginal_sf_does_not_add_metadata():
    s = fixture()
    for r in s["results"]:
        if r["route"] % 2:
            for p in r["profiles"]:
                p["samples_us"] = [9.9] * 5
    assert all(r["selected"] == "FQ" for r in export(s)[1])


@pytest.mark.parametrize("plant", range(5))
def test_invalid_prefill_evidence(plant):
    s = fixture()
    r = s["results"][0]
    if plant == 0:
        r["profiles"][0]["error"] = float("nan")
    if plant == 1:
        r["profiles"][0]["samples_us"] = [0.0] * 5
    if plant == 2:
        r["n"] = 256
    if plant == 3:
        r["rows_host"] = True
    if plant == 4:
        r["profiles"][0]["rows"] = [0]
    with pytest.raises(ValueError):
        export(s)
