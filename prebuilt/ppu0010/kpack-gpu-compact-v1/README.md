# GPU compact and persistent grouped experiment

PPU-only modules built from `93ac48f`, with immutable ordinary baselines from
the previously validated decode experiment. PPU admission of the new modules
is pending; this is not a replacement production bundle.

- 16 new modules: five-format FQ controls, Q4/Q5 TM8 and TM16, resident-SF controls.
- Four old ordinary modules and the unchanged scalar/pair SIMT execution DSO.
- Device-only compact directory and persistent S1/S2/S4/S8, unchanged grouped ABI.
- Eleven isolated jobs / 204 cells; no box compilation, failed-job resume.

`manifest.json` binds exact source/kernel identities and payload hashes. The
host-compact baseline excludes CPU preparation. New GPU timings include both
metadata/directory kernels and the reducer; SF controls use resident metadata.

See [the experiment guide](../../../docs/KPACK_GPU_COMPACT.md) for the command,
correctness checks and timing scope. Existing native selection is unchanged.
