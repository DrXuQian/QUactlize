"""SDK/source identity is identical on Python 3.9/3.10 and newer hosts."""
import hashlib

import pytest

from quactlize.runtime import compiler


@pytest.mark.parametrize("payload", [b"", b"source identity\n", bytes(range(256)) * 9000])
def test_older_python_streaming_hash_matches_sha256(tmp_path, monkeypatch, payload):
    path = tmp_path / "input.bin"
    path.write_bytes(payload)
    monkeypatch.delattr(compiler.hashlib, "file_digest", raising=False)
    assert compiler.sha(path) == hashlib.sha256(payload).hexdigest()


def test_newer_python_uses_file_digest_when_available(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    payload = b"same source, same cache key"
    path.write_bytes(payload)
    calls = []

    def file_digest(stream, algorithm):
        calls.append(algorithm)
        return hashlib.new(algorithm, stream.read())

    monkeypatch.setattr(compiler.hashlib, "file_digest", file_digest, raising=False)
    assert compiler.sha(path) == hashlib.sha256(payload).hexdigest()
    assert calls == ["sha256"]
