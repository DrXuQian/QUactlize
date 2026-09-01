---
name: ppu-main-productization
description: Curate proven PPU kernel, offline-format, routing, and packaging changes from the development branch into a minimal product main branch. Use for main admission audits, selective ports, or cleanup that determines what may ship.
---

# PPU Main Productization

Build main from the product boundary; never merge the development branch wholesale. Development contains experiments, negative controls, local oracles, profiling harnesses, and collaboration state that are useful there but are not product dependencies.

## Product boundary

- Start from the current remote main tip and port only the dependency closure of an admitted feature.
- Ship only PPU product code. Keep NVIDIA-only compatibility, CPU/fake backends, local probes, and profiler assets in development or CI.
- Ship only the canonical K-pack formats: Q4 K-pack4 and the per-plane K-pack layouts for Q2/Q3/Q5/Q6. Do not include Xplane producers, readers, restore paths, or automatic fallbacks.
- Do not port experimental layout identities, delivery plugins, or scheduler arms until a real PPU kernel has passed their admission gates.
- Remove unused flags. Express a selected implementation as a type or an ordinary product policy, not as a dormant diagnostic macro.
- Exclude `.coord/`, `.codex/`, `dev/`, local artifacts, assistant/collaboration provenance, historical narratives, and source-matching test scaffolds from main.
- Preserve required copyright and license notices. Platform cleanup is not permission to remove attribution.

## Admission workflow

1. Resolve the exact source and target commits and inspect both worktrees. Preserve unrelated changes.
2. Enumerate the feature's runtime path from offline producer and descriptor through host routing, device launcher, collective, and public API. Port only files and generated tables reachable from that path.
3. Classify every candidate file or macro as product, test-only, diagnostic-only, compatibility-only, or dead. Do not copy a whole directory to avoid making this decision.
4. Make the product path fail closed on unknown layouts, mappings, qtypes, shapes, and unavailable kernels. Never silently reinterpret old bytes.
5. Run local ABI, inverse, mapping, policy, import, and packaging tests before using a device.
6. Require fresh PPU evidence whenever a change affects lowering, memory ordering, fragments, numeric results, resources, scheduling, or performance. Bind evidence to the tested source SHA and exact binary/config identity.
7. Update the supported public API, installed packer entry point, concise diagnostics, and README in the same product change.

## Device admission

A new kernel path is not admitted by host simulation or CuTe type formation. Require:

- independent offline prepare/recover oracles and a planted wrong-map control;
- raw-bit correctness on the complete declared shape/operator denominator;
- code generation proving the intended producer and reader instructions;
- registers, spills, shared/workspace bytes, and repeated timing against the shipping baseline;
- dense/decode, prefill, and grouped coverage when the format is claimed to unify them.

Keep measured K-pack versus Xplane regressions visible as technical debt. A maintenance decision or bounded performance waiver may permit shipment, but it must not rewrite a technical `KEEP_XPLANE` result into a false performance win.

## Binary handoff to the PPU box

- Build every box target locally with the pinned PPU SDK whenever the target can be compiled off-device. The box should execute an already-built artifact, not spend its run rebuilding it.
- Publish the exact executable or loadable library together with the source commit, submodule commits, PPU architecture, SDK/compiler identity, target, compile definitions, SHA-256, and invocation. A compile-only object is code-generation evidence, not a runnable box binary.
- Keep binary bundles out of `develop` and `main`. Publish them on a dedicated artifact branch or artifact store so product source history remains reviewable; the box fetches the exact bundle named by its manifest. When Git carries the bundle, store payloads with Git LFS and verify `git lfs ls-files` before pushing; never force-add a compiled payload as an ordinary Git blob.
- A prebuilt runner must verify the manifest and binary digest before launch and must not silently rebuild when the requested artifact is absent or mismatched.
- Device output remains authoritative only for execution, raw-bit/numeric results, hardware ordering, counters, and performance. Bind returned evidence to the published binary digest.

## Main review checklist

- No Xplane or NVIDIA-only code is reachable or packaged.
- No diagnostic flag, deliberately wrong arm, probe print, or historical negative remains in a product header or translation unit.
- No collaboration identity or conversation-derived wording appears in product source or docs.
- Public routes consume versioned arrangement descriptors and cannot guess an offline layout.
- Config scanning and format-selected library loading have one documented entry point.
- Logs report only actionable identity, failure, and summary information.
- Local gates pass; required PPU gates pass on the exact candidate commit.
- The skill and all other development-only process files remain outside main.
