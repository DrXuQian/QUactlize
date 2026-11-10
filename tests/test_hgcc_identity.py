"""Host-only HGGC identity contracts; no compilation or device execution."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.hgcc_identity import compiler_identity

ROOT = Path(__file__).resolve().parents[1]
BANNER = "hgcc(HGGC C/C++ Compiler)\nRelease version 2.2.1-test\n"


@pytest.mark.parametrize("supported", ["-V", "--version"])
@pytest.mark.parametrize("stderr_banner", [False, True])
def test_real_process_accepts_sdk_dialect_and_records_identity(tmp_path, supported, stderr_banner):
    compiler = tmp_path / "hgcc"
    compiler.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"if sys.argv[1] == {supported!r}:\n"
        f"    print({BANNER!r}, file=sys.stderr if {stderr_banner!r} else sys.stdout)\n"
        "else:\n"
        "    print('hgcc error: unknown argument ' + repr(sys.argv[1]), file=sys.stderr)\n"
        "    raise SystemExit(3)\n"
    )
    compiler.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/hgcc_identity.py"), "--compiler", str(compiler)],
        text=True, capture_output=True, check=True,
    )
    record = json.loads(result.stdout)
    assert record["version"] == BANNER.strip()
    assert record["version_flag"] == supported
    assert record["sha256"] == hashlib.sha256(compiler.read_bytes()).hexdigest()
    expected = ["-V"] if supported == "-V" else ["-V", "--version"]
    assert [probe["argv"][-1] for probe in record["probes"]] == expected
    if supported == "--version":
        assert record["probes"][0]["returncode"] == 3
        assert "unknown argument" in record["probes"][0]["output"]


@pytest.mark.parametrize("code,output", [
    (0, ""),
    (0, "hgcc usage: compile files"),
    (0, "gcc version 13.0"),
    (3, BANNER),
    (3, "hgcc error: unknown argument"),
])
def test_empty_unrelated_or_failed_output_never_becomes_identity(tmp_path, code, output):
    compiler = tmp_path / "hgcc"
    compiler.write_bytes(b"host-fixture-not-a-real-compiler")
    response = subprocess.CompletedProcess([], code, stdout=output)
    with patch("tools.hgcc_identity.subprocess.run", return_value=response) as run:
        with pytest.raises(RuntimeError, match="cannot read an HGGC version banner"):
            compiler_identity(compiler)
    assert run.call_count == 2


@pytest.mark.parametrize("failure", [
    OSError("cannot execute compiler"),
    subprocess.TimeoutExpired(["hgcc", "-V"], 10),
])
def test_unreadable_query_is_not_silently_ignored(tmp_path, failure):
    compiler = tmp_path / "hgcc"
    compiler.write_bytes(b"host-fixture")
    with patch("tools.hgcc_identity.subprocess.run", side_effect=failure):
        with pytest.raises(RuntimeError, match="compiler_sha256"):
            compiler_identity(compiler)


def test_missing_compiler_never_falls_back_to_path(tmp_path):
    with patch("tools.hgcc_identity.subprocess.run") as run:
        with pytest.raises(FileNotFoundError):
            compiler_identity(tmp_path / "absent-hgcc")
    run.assert_not_called()
