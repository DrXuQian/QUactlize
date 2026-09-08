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


def test_prefill_choices_and_break_even():
    text, records = export(fixture())
    assert len(text.splitlines()) == 15
    assert all(
        r["selected"] == "SF" and r["kernel_break_even_calls"] == 3 for r in records
    )


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
