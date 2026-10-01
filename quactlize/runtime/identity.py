"""Stable identifiers shared by selection and compile-only cache manifests."""

import hashlib
import json

ROUTES = ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
