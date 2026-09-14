# Typed decode identity repair

Source: develop 92ed05f (including measured prefill policy 431a698).
SDK: PPU 2.1.1; exact compiler/runtime identities are in manifest.json.

27 F32/BF16 endpoint modules, five ordinary controls and two SIMT stage proofs.
Only the typed host device probe changes: it now reads the SM attribute, like
the existing ordinary entry and the typed resource query. No GEMM arithmetic,
thread mapping, split factor or decode policy changes.

The original 104-request gate remains required. This package is compile-only;
PPU numerical/graph admission is pending. Fetch with Git LFS on this artifact
branch and pass this directory as BUNDLE to the develop runner. Do not copy
an old manifest over these payloads or silently reuse the old typed modules.
