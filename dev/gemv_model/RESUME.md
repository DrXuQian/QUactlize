# Resume

State: seven exact M1 candidates integrated; package publication in progress.
Integration starts at bfa5b38; Q6 TC and all unmeasured fallbacks stay unchanged.
See ../../docs/MODEL_GEMV_INTEGRATION_20260917.md for scope and validation.
Branch: `dev/gemv-model-tuning`.
Baseline source: `4f181a071ebf3715f90b2898033497342f9af4ca`.
Baseline artifact: `6a9b89a322f1ccd5cf1b724325294f7d2ae129b1`.
The initial experiment below did not change production. The authorized
integration now changes exact readers/selection and caller trace parsing only.

Build: `/tmp/gemv-model-build-r3-20260917`, 84.8 seconds, eight parallel jobs.
Local: host contracts, package and emitted ISA/resource checks PASS.
The earlier 62-body inspection found FP32 FMA, fast code extraction, vector
loads and zero stack. The final 68-body inspection is in native-inspection.json.
Full N248320/K2048 Q6 fixture preparation took 29.9 seconds on the build host;
the 417177600 packed bytes are generated in chunks, not a huge CPU GEMM.
CPU host runtime loading is not admitted: its conda libstdc++ lacks the SDK's
GLIBCXX_3.4.32 requirement. Device results below come from the returned box run.

Artifact: `artifacts/model-gemv-reader-v1`, `9df3eb212289d05c8dbee6368d8b63621f1c8305`.
Pin: `tools/kpack_model_gemv_artifact.json`. 18 host tests PASS; 68 emitted
candidate bodies verified. The package has 38 payloads (53.6 MB including
unchanged incumbent images and full ISA). No model or caller binary is shipped.
Returned: `/root/model-gemv.mZfskR.results.tgz`, source `2d8e5030afbbe757ab6759c0d2a37440310f1a32`.
Extraction: `/tmp/model-gemv-review.5KHTre/results`.
Local ACU re-import/audit: `/tmp/model-gemv-audit-r2-20260917/review.json`.
All 16 actual reports / 20 kernels re-imported; counters equal returned CSV.
510 numerical control rows and 2160 finite confirmation samples PASS.
23 host tests PASS, including wrong-config/finalist/winner audit negatives.

Exact M1 winners: Q4 paired arm3, Q5 down arm1, Q8 shared paired arm10,
Q8 shared down arm6, Q8 SSM arm1 (S8), Q8 QKV arm6, Q8 attention gate arm8.
These win every one of six rounds. Q6 arm5 is 1.15% slower overall with drift;
keep the current TC. M2/M8 have numerical evidence only. No global SIMT/S1
promotion and no new model TPOT claim. See `docs/MODEL_GEMV_REVIEW_20260917.md`
and its structured receipt. Next: exact-scope integration, then model gate/Asys.

Evidence before this task: Q4 paired loses the existing H32 specialization;
Q8 paired does not hoist whereas the old shared reader does. These are source
differences. Returned counters now support Q4/Q5 instruction-count reductions
and shared-Q8 CTA expansion. Q5's fastest arm has more registers; shared Q8
has more internal traffic. Neither register count nor bytes alone select a winner.
