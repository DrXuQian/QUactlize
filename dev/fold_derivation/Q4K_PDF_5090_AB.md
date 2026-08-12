# RTX 5090：Q4_K PDF 重建版与 gemv_lowbit 同机 A/B

结论边界：这是方向性实验，不是 PPU 判决，也不是 PDF 原实现的逐字复现。PDF 缺失 launcher 尾、
packer、golden 和 timer；launcher 由文档第 14 页的 grid/block/smem 伪码恢复。文档主 listing 的
scalar metadata 转换与解释页的 pair 转换同时保留为两个 arm，避免把两份不同代码揉成一个“原版”。

- Raw CSV: `dev/acu/q4k_pdf_5090_ab_5fbb494.csv`
- Git: `5fbb49418de77f85372004623ce4dfe50b4078b7`
- Binary SHA-256: `2cbba26c54e4de99e15da6aa8fe4ce09ec6a1b4fefd74b5ad6a80f18afc2e02e`
- Device / PCI / driver: `NVIDIA GeForce RTX 5090` / `0000:63:00.0` / `595.71.05`

## 输入与协议

- 两臂来自同一份 logical Q4_K。PDF 臂直接读 144 B/256-weight block；ours 读 Native affine int4
  code plane + fp16 scale/zero(gs=32)。CPU golden 独立从 raw block 解码；任一输出不满足固定
  conditioned error `<=2^-7` 时整组拒绝计时。
- `weight_metadata_cold`：先触碰 `max(2×L2,128 MiB)` flush buffer，再在一个 event 中逐份读取完整且
  不重叠的 representation。cold budget=512 MiB，copies=`min(64,floor(budget/max_repr))`；
  两臂 cold batch 相同，ours 的 S/Z 也逐份复制，不只冷 low plane。
- `warm`：计时前每个 arm warmup 100 rounds；每个 event 固定 64 个 logical workloads。
  每个 shape/state/arm 保留 31 个原始样本；两状态前均有 50 ms host enqueue window
  交替提交各 arm，随后同步。该窗口不是 GPU 恰好运行同样时长的声明。
  AB/BA 交替；初始化、pack、H2D、flush 与 NVML 查询均在目标 event 外。event span 包含 GPU launch
  间隙，因此是 kernel-only 的上界而非 CUPTI kernel duration 同义词。
- 每个 stop event 入队后采一次 NVML SM clock，并记录 event 当时是否仍 pending。它是 adjacent snapshot，
  不是 time-integrated kernel clock。
- L=8 点同时报 `ours_native_grouped1`（1 kernel/workload）与 `ours_native_dense8`（8 kernels/workload）；
  PDF API 只有 dense，因此是 8 kernels/workload。不能把 1-vs-8 launch 差异藏起来。

## 原始汇总

| shape | state | arm | batch | kernels/work | repr MiB | median us | min..max us | GB/s* | SM MHz median[min,max] | pending | observed GCD grid/work | admissible quantum/work |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-EXT-O | weight_metadata_cold | ours_native_dense1 | 20 | 1 | 25.000 | 17.7152 | 17.6112..17.9200 | 1481.3 | 2407[2407,2407] | 31/31 | 0.001600 us | UNKNOWN |
| D-EXT-O | weight_metadata_cold | pdf_pair_dense1 | 20 | 1 | 22.500 | 16.7216 | 16.6864..16.8208 | 1412.5 | 2407[2407,2407] | 31/31 | 0.001600 us | UNKNOWN |
| D-EXT-O | weight_metadata_cold | pdf_scalar_dense1 | 20 | 1 | 22.500 | 16.7424 | 16.6880..16.8432 | 1410.8 | 2407[2407,2407] | 30/31 | 0.001600 us | UNKNOWN |
<!-- timer D-EXT-O/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| D-EXT-O | warm | ours_native_dense1 | 64 | 1 | 25.000 | 7.4030 | 7.3640..7.5230 | 3544.6 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-O | warm | pdf_pair_dense1 | 64 | 1 | 22.500 | 7.1705 | 7.1380..7.2680 | 3294.0 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-O | warm | pdf_scalar_dense1 | 64 | 1 | 22.500 | 6.8580 | 6.8245..6.9540 | 3444.1 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
<!-- timer D-EXT-O/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| D-EXT-K1024 | weight_metadata_cold | ours_native_dense1 | 64 | 1 | 3.125 | 3.9835 | 3.9535..4.0020 | 825.7 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-K1024 | weight_metadata_cold | pdf_pair_dense1 | 64 | 1 | 2.812 | 3.6790 | 3.6655..3.6825 | 804.9 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-K1024 | weight_metadata_cold | pdf_scalar_dense1 | 64 | 1 | 2.812 | 3.6495 | 3.6390..3.6745 | 811.5 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
<!-- timer D-EXT-K1024/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| D-EXT-K1024 | warm | ours_native_dense1 | 64 | 1 | 3.125 | 2.9155 | 2.8990..2.9240 | 1128.1 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-K1024 | warm | pdf_pair_dense1 | 64 | 1 | 2.812 | 2.4890 | 2.4650..2.4935 | 1189.8 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-K1024 | warm | pdf_scalar_dense1 | 64 | 1 | 2.812 | 2.4600 | 2.4495..2.4905 | 1203.8 | 2407[2407,2407] | 31/31 | 0.000500 us | UNKNOWN |
<!-- timer D-EXT-K1024/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| D-EXT-Q | weight_metadata_cold | ours_native_dense1 | 20 | 1 | 25.000 | 18.1760 | 18.1056..18.3616 | 1443.7 | 2917[2917,2917] | 31/31 | 0.001600 us | UNKNOWN |
| D-EXT-Q | weight_metadata_cold | pdf_pair_dense1 | 20 | 1 | 22.500 | 17.4896 | 17.3856..17.6144 | 1350.5 | 2917[2917,2917] | 31/31 | 0.001600 us | UNKNOWN |
| D-EXT-Q | weight_metadata_cold | pdf_scalar_dense1 | 20 | 1 | 22.500 | 18.2048 | 18.1008..18.3264 | 1297.4 | 2917[2917,2917] | 31/31 | 0.001600 us | UNKNOWN |
<!-- timer D-EXT-Q/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| D-EXT-Q | warm | ours_native_dense1 | 64 | 1 | 25.000 | 8.0040 | 7.9695..8.0120 | 3278.5 | 2940[2940,2940] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-Q | warm | pdf_pair_dense1 | 64 | 1 | 22.500 | 8.3240 | 8.2980..8.3485 | 2837.5 | 2940[2940,2940] | 31/31 | 0.000500 us | UNKNOWN |
| D-EXT-Q | warm | pdf_scalar_dense1 | 64 | 1 | 22.500 | 8.2815 | 8.2560..8.3050 | 2852.1 | 2940[2940,2940] | 31/31 | 0.000500 us | UNKNOWN |
<!-- timer D-EXT-Q/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| H-G8-2048 | weight_metadata_cold | ours_native_dense8 | 25 | 8 | 20.000 | 26.6650 | 26.5779..26.7878 | 788.9 | 2925[2925,2925] | 31/31 | 0.001280 us | UNKNOWN |
| H-G8-2048 | weight_metadata_cold | ours_native_grouped1 | 25 | 1 | 20.000 | 14.6035 | 14.5203..14.6637 | 1440.5 | 2925[2925,2925] | 31/31 | 0.001280 us | UNKNOWN |
| H-G8-2048 | weight_metadata_cold | pdf_pair_dense8 | 25 | 8 | 18.000 | 25.0522 | 24.9843..25.1584 | 756.0 | 2925[2925,2925] | 31/31 | 0.001280 us | UNKNOWN |
| H-G8-2048 | weight_metadata_cold | pdf_scalar_dense8 | 25 | 8 | 18.000 | 25.0112 | 24.9267..25.2096 | 757.3 | 2925[2925,2925] | 31/31 | 0.001280 us | UNKNOWN |
<!-- timer H-G8-2048/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->
| H-G8-2048 | warm | ours_native_dense8 | 64 | 8 | 20.000 | 20.4420 | 20.3930..20.4765 | 1029.1 | 2925[2925,2925] | 31/31 | 0.000500 us | UNKNOWN |
| H-G8-2048 | warm | ours_native_grouped1 | 64 | 1 | 20.000 | 7.8040 | 7.7465..7.8600 | 2695.7 | 2925[2925,2925] | 31/31 | 0.000500 us | UNKNOWN |
| H-G8-2048 | warm | pdf_pair_dense8 | 64 | 8 | 18.000 | 19.4185 | 19.3695..19.4655 | 975.4 | 2925[2925,2925] | 31/31 | 0.000500 us | UNKNOWN |
| H-G8-2048 | warm | pdf_scalar_dense8 | 64 | 8 | 18.000 | 19.0355 | 18.9755..19.0680 | 995.0 | 2925[2925,2925] | 31/31 | 0.000500 us | UNKNOWN |
<!-- timer H-G8-2048/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor -->

`GB/s*` 用每个 arm 自己的 distinct representation + A + D。warm 行是 cache-equivalent rate，不是 DRAM 利用率。

## 相对方向（事前判据）

方向只有在两臂 raw `[min,max]` 不重叠，且 median 差大于一个有效 event quantum 时才判定；否则为
`UNRESOLVED`。PDF 两个 metadata variant 先各自展示，比较时采用其中更快者并明确这是文档内部歧义，
不是事后把两份实现冒充成一个确定原版。
本协议事前规定：总 event 的 GCD 低于 0.5 us 时只作为 observed grid 展示，不准入为计时器分辨率；
因此本次 admissible quantum 全为 `UNKNOWN`，正式 verdict 全部 fail-close 为 `UNRESOLVED`。

| shape | state | comparison | ratio target/PDF | sampled bands | paired target/PDF/tie | resolution-qualified verdict |
|---|---|---|---:|---|---:|---|
| D-EXT-O | weight_metadata_cold | ours_native_dense1 / pdf_pair_dense1 | 1.0594 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| D-EXT-O | warm | ours_native_dense1 / pdf_scalar_dense1 | 1.0795 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| D-EXT-K1024 | weight_metadata_cold | ours_native_dense1 / pdf_scalar_dense1 | 1.0915 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| D-EXT-K1024 | warm | ours_native_dense1 / pdf_scalar_dense1 | 1.1852 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| D-EXT-Q | weight_metadata_cold | ours_native_dense1 / pdf_pair_dense1 | 1.0392 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| D-EXT-Q | warm | ours_native_dense1 / pdf_scalar_dense1 | 0.9665 | target faster | 31/0/0 | UNRESOLVED: quantum rejected by policy |
| H-G8-2048 | weight_metadata_cold | ours_native_dense8 / pdf_scalar_dense8 | 1.0661 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| H-G8-2048 | weight_metadata_cold | ours_native_grouped1 / pdf_scalar_dense8 (topology-inclusive 1-vs-8) | 0.5839 | target faster | 31/0/0 | UNRESOLVED: quantum rejected by policy |
| H-G8-2048 | warm | ours_native_dense8 / pdf_scalar_dense8 | 1.0739 | selected PDF variant faster | 0/31/0 | UNRESOLVED: quantum rejected by policy |
| H-G8-2048 | warm | ours_native_grouped1 / pdf_scalar_dense8 (topology-inclusive 1-vs-8) | 0.4100 | target faster | 31/0/0 | UNRESOLVED: quantum rejected by policy |

方向性证据：10 个比较中 raw bands 有 10 个不重叠，按同 pass 配对有
10 个呈 31/31 单向。它证明 sampled direction 稳定；它不把被事前政策拒绝的
32 ns observed grid 升格成可准入 timer quantum，因此不会把方向证据冒充 resolution-qualified 判决。

## 不可外推的部分

1. 这只消除了机器差异。5090 与 PPU 在 gs=32 上出现过 config 排名反转，因此只能提出方向与待验假设。
2. PDF 第 1 页的 15/15/4 us headline、instrumented profile、以及第 22 页 `warmup 3 + 20 iters`
   throughput 不是同一精确 protocol；本表不把它们拼接成一个基线。
3. 原生 Q4_K 是 0.5625 B/weight；ours 是 0.625 B/weight。绝对时间可比，GB/s 必须用各自行的分子。
4. PDF 未提供 L=8 grouped kernel；该行的 8 次 dense launch 是 API 事实，不是 grouped 等价实现。
5. scalar/pair 两版均来自 PDF，但无法从文档判定第 22 页时间对应哪版。本结果保留两行，不替作者选择。
