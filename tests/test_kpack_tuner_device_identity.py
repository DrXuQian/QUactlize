from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_kpack_tuner as tuner


def driver_result(name="PPU-ZW810"):
    return SimpleNamespace(
        returncode=0,
        stdout=(
            f"KPACK_TUNER_DEVICE visible=1 device=0 name={name} pci=0000:00:00 cu=72\n"
        ),
    )


def mock_probes(monkeypatch, same=False):
    monkeypatch.setattr(tuner, "build_identity_probe", lambda *_: None)
    monkeypatch.setattr(tuner.subprocess, "run", lambda *_, **__: driver_result())

    def query(_binary, env):
        ordinal = int(env["CUDA_VISIBLE_DEVICES"])
        return {
            "name": "PPU-ZW810",
            "ordinal": 0,
            "pci_identity": f"0000:08:00.{0 if same else ordinal}",
            "pci_method": "hggcDeviceGetPCIBusId",
        }

    monkeypatch.setattr(tuner, "query_identity_probe", query)


def test_duplicate_property_pci_does_not_hide_unique_api_bdf(monkeypatch):
    mock_probes(monkeypatch)
    rows = tuner.probe_devices(
        {"pairs": {"q12": {"driver": "driver"}}}, [0, 1], Path("/sdk")
    )
    assert [r["pci"] for r in rows] == ["0000:08:00.0", "0000:08:00.1"]
    assert all(r["property_pci"] == "0000:00:00" for r in rows)
    assert all(r["pci_method"] == "hggcDeviceGetPCIBusId" for r in rows)


def test_real_duplicate_api_bdf_still_rejected(monkeypatch):
    mock_probes(monkeypatch, same=True)
    with pytest.raises(ValueError, match="same physical device via SDK PCI query"):
        tuner.probe_devices(
            {"pairs": {"q12": {"driver": "driver"}}}, [0, 1], Path("/sdk")
        )


def test_mismatched_driver_sidecar_rejected(monkeypatch):
    mock_probes(monkeypatch)
    monkeypatch.setattr(
        tuner.subprocess, "run", lambda *_, **__: driver_result("OTHER")
    )
    with pytest.raises(ValueError, match="disagrees"):
        tuner.probe_devices({"pairs": {"q12": {"driver": "driver"}}}, [0], Path("/sdk"))


@pytest.mark.parametrize(
    "pci,method,allowed",
    (
        ("0000:08:00.0", "hggcDeviceGetPCIBusId", True),
        ("-", "hggcDeviceGetPCIBusId", False),
        ("0000:08:00.0", "-", False),
    ),
)
def test_sdk_wire_requires_measured_pci(monkeypatch, pci, method, allowed):
    wire = (
        "QZ_HGGC_DEVICE_PROBE_V1\ncount\t1\n"
        f"device\t0\t5050552d5a57383130\t0\t0\t1\t{pci}\n"
        f"pci_method\t{method}\ndriver\t13000\thggcDriverGetVersion\n"
    )
    monkeypatch.setattr(
        tuner.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(returncode=0, stdout=wire),
    )
    if allowed:
        assert tuner.query_identity_probe(Path("probe"), {})["pci_identity"] == pci
    else:
        with pytest.raises(ValueError, match="measured PCI"):
            tuner.query_identity_probe(Path("probe"), {})


def test_probe_and_driver_share_exact_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited")
    env = tuner.device_environment(2, Path("/sdk"))
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["LD_LIBRARY_PATH"].startswith("/sdk/lib:")
    assert env["LD_LIBRARY_PATH"].endswith(":/inherited")
