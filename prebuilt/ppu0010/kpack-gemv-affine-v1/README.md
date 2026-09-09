# PPU GEMV FP32-affine experiment

This is a small development comparison DSO, not a production replacement.
It was compiled locally with hgcc; no NVIDIA compatibility headers are used.
The pair dot body is the same FP32 group-affine/vector-load experiment tested
on RTX 5090. Kernel namespaces use `kpack_affine_q*` to distinguish it from
the old pair reader in ACU and avoid identical kernel registration names.

`manifest.json` binds source/generated-code hashes, compiler, SDK runtime,
and the Git LFS DSO. `device_validated=false` is intentional until box results
are reviewed. The old pair SIMT/prepass is reused from the execution DSO in
`../kpack-decode-sweep-v1`; the selected FQ/SF modules are reused from
`../kpack-native-v1`. Neither production selection nor existing DSOs change.

Entry: `bash tools/run_kpack_gemv_fq_sf_box.sh`. See
`docs/KPACK_GEMV_FQ_SF.md` for shapes, timing boundaries and report names.
