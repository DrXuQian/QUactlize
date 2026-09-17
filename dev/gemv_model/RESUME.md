# Resume

State: all 68 candidate bodies and three frozen TC controls compiled;
emitted ISA and package verification PASS. No device admission.
Branch: `dev/gemv-model-tuning`.
Baseline source: `4f181a071ebf3715f90b2898033497342f9af4ca`.
Baseline artifact: `6a9b89a322f1ccd5cf1b724325294f7d2ae129b1`.
No production kernel, selection or caller changed.

Build: `/tmp/gemv-model-build-r3-20260917`, 84.8 seconds, eight parallel jobs.
Local: host contracts, package and emitted ISA/resource checks PASS.
The earlier 62-body inspection found FP32 FMA, fast code extraction, vector
loads and zero stack. The final 68-body inspection is in native-inspection.json.
Full N248320/K2048 Q6 fixture preparation took 29.9 seconds on the build host;
the 417177600 packed bytes are generated in chunks, not a huge CPU GEMM.
CPU host runtime loading is not admitted: its conda libstdc++ lacks the SDK's
GLIBCXX_3.4.32 requirement. No device correctness or performance is claimed.

Next: publish the independent experiment through Git LFS; run the box command
in README. Keep candidate/production selection separate until returned gates.

Evidence before this task: Q4 paired loses the existing H32 specialization;
Q8 paired does not hoist whereas the old shared reader does. These are source
differences, not evidence of a measured improvement. Register pressure can
reverse either expected gain.
