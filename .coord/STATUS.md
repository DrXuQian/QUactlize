# codex status

    updated-at:     2026-08-14 05:07:28 UTC
    inbox-consumed: 169
    working-on:     fixed the m8 ACU box compile: the target-wide profile arm now looks up InstructionM only for a compile-time-proved standalone Marlin kernel, while generic GemmUniversal instantiations fail closed without that member lookup
    blocked-on:     box rebuild and clean BPC1/2/3 subject-only ACU reports
    local-gates:    standalone/composed contracts, L169 and L181 PASS; the if-constexpr-to-runtime-if plant is rejected; exact m8 multi-TU boxdry PASS (1/1, 96.6s)
    last-commit:    Guard m8 ACU identity to standalone kernels
    last-heartbeat: 305
