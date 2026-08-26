# Rejected Q4 A64/F1 virtual-F2 experiment

The opt-in experiment was measured on PPU at source SHA
`614f297ea55b6af3b33f4e282b29863d0d783a5b` with the exact tactic
`64x128x128_w64x64_s3_bc0`, persistent capacity grid 216, and problem
`4096x5120x8192`.

All three arms were raw-bit exact. Median full-output latency was:

| resident layout / reader | median us | range us |
|---|---:|---:|
| A32/native-F2 | 2663.840055 | 2644.079924--2677.479982 |
| A64/F1 | 5589.320183 | 5566.959858--5633.319855 |
| A64/virtual-F2 | 5934.239864 | 5903.719902--5959.640026 |

The virtual reader was 6.171% slower than ordinary F1 and 122.770% slower
than native F2. Its ranges were separated from ordinary F1. The experiment
proved that A64/F1 deliveries can populate a fold-2 logical MMA fragment, but
also proved that changing only `MmaPermK` cannot reproduce native F2's folded
global/shared delivery, conversion, and mainloop cadence. The production
experiment and its dedicated proof/runner scaffolding were therefore removed.

The follow-up direction is native A32/F2 decode: retain the existing folded B
layout and connect the packed Q4_K metadata provider to that collective.
