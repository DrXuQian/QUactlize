# RTX 5090 dense vLLM Marlin：K/N 轴（INBOX 144）

范围：这是本机 RTX 5090 / SM120 的 vLLM dense Marlin W4A16、M=1、gs=32 测量，
不是 PPU 结论。分母固定为该 5090 的 1792 GB/s；仓库已有 gs=32 在两台机器上
config 排名反转的记录，因此不能把这里的排序外推到 PPU。

- Raw CSV: `dev/acu/vllm_marlin_dense_axis_5090_144_9e9814e.csv`
- Repository SHA: `9e9814e2cff08406a97e2daa3c73521665eea1eb`
- Measurement source authority: `/root/ref5090/marlin/fullrun/marlin_fullrun.cu` / `686e72323967bbeea8739d28b0143f18b69c3e951375efa3e8b7fa6f96ea8cb8`
- vLLM commit: `11ba93f3646d4c5476c3b3fd56835589701f0fb1`
- Binary SHA-256: `ecb058147e9c47ad931e02e1025c124edb9abfa9dd2ca1e0db0914a48fb548f8`
- Device / PCI / driver: `NVIDIA GeForce RTX 5090` / `0000:63:00.0` / `595.71.05`

## 协议

- 直接 include pinned `marlin_fullrun.cu`（rename `main`），复用原 `Config`、
  `select_dense_config`、`DenseLaunch` 和 kernel 实例化；没有复制 selector/launch ABI。
- 计时前先做非零、可精确预测的 correctness gate：A=1，所有 int4 code=9，
  fp16 scale=1/256，故每个输出严格为 K/256；所有 fp16 bits 必须逐位相等。
  correctness fixture identity=`exact_q9_a1_scale2m8_expectedKover256_fp16bits_v1`。通过后才把 A/B/scale 全量重填为
  pinned fullrun 的三个原始 seed；timing fixture identity=`pinned_fullrun_seeds_b57a41d9b_s16334a2f_a91104f23_v1`。因此常量
  correctness 数据不会进入计时，也不会把压缩性带入 cold HBM 结果。
- cold：event 外先触碰 `max(2×L2,128 MiB)`，event 内依次消费互不重叠的完整
  B+scale replica。warm：同一 B/scale 上批量调用。两者均 31 个独立 event 样本；
  初始化、correctness、warmup、precondition、flush 与 NVML 查询均在 event 外。
- 单 arm 不存在 forward/reverse；raw 明确记录 `single_arm_no_counterbalance`，不冒充
  positional counterbalance。stop event 后紧邻采 NVML clock，并保留 binary32 event bits。
- 事前分辨率政策沿用 q4k：总 event 小于 0.5 us 的 observed GCD 不准入。表中同时
  给 `policy-min/work = 0.5 us / batch` 与由 event bits 推出的 admissible resolution；
  后者拿不到即标 `QUANTUM UNKNOWN`；拿到则标 `QUANTUM ESTABLISHED`。单臂绝对时间
  没有“赢家”判决；quantum 只约束后续差值。

## 新测四格

| K | N | state | batch | median us | min..max us | distinct MiB | GB/s* | % of 1792 HBM | policy-min/work us | observed GCD/work | admissible resolution/work | resolution status | clock MHz median[min,max] | pending |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 1024 | warm | 64 | 7.2420 | 7.2265..7.3290 | 0.5664 | 82.01 | 4.576% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K1024/N1024/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 1024 | 1024 | weight_metadata_cold | 64 | 7.7825 | 7.7375..8.0260 | 0.5664 | 76.31 | 4.259% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 30/31 |
<!-- K1024/N1024/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 1024 | 5120 | warm | 64 | 3.5940 | 3.5850..3.6190 | 2.8242 | 823.99 | 45.981% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K1024/N5120/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 1024 | 5120 | weight_metadata_cold | 64 | 4.8895 | 4.8575..4.9060 | 2.8242 | 605.67 | 33.798% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K1024/N5120/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 5120 | 1024 | warm | 64 | 9.2895 | 9.2480..9.3560 | 2.8242 | 318.79 | 17.790% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K5120/N1024/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 5120 | 1024 | weight_metadata_cold | 64 | 10.6810 | 10.6345..10.6980 | 2.8242 | 277.26 | 15.472% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K5120/N1024/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 5120 | 5120 | warm | 64 | 6.7300 | 6.6975..6.7875 | 14.0820 | 2194.07 | 122.437% | 0.007812 | 0.000500 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K5120/N5120/warm: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->
| 5120 | 5120 | weight_metadata_cold | 36 | 12.3316 | 12.2738..12.3902 | 14.0820 | 1197.42 | 66.820% | 0.013889 | 0.000889 | UNKNOWN | QUANTUM UNKNOWN | 1297[1297,1297] | 31/31 |
<!-- K5120/N5120/weight_metadata_cold: adjacent-difference GCD 0.032 us rejected by the predeclared 0.5-us floor; admissible=None -->

`GB/s*` 使用 Marlin 自己的 distinct bytes：A + biased-int4 B + fp16 scale(gs32) + D。
warm 超过 100% 也只表示 cache-equivalent rate，不是 DRAM counter。cold event 每次读取互不
重叠的 B+scale replica；A 与 D 各自复用同一份 buffer，所以每个 logical workload 的 modeled
分子各计一次 A/D，并不冒充四个 operand 都逐 batch 复制。

## 旧协议交叉检查：K=5120, N=1024

Pinned fullrun case `S003`: 11.488000 us, 14.385197% of 1792 GB/s, raw band 9.888000..12.672000 us, 31 samples。

两次使用相同的 pinned random fill seeds，故 fixture matched；但该旧协议是
`warm same buffers; no explicit L2 flush`、每个 event 只发一次，且当时没有
事前注册 0.5-us event-floor。因此它不替代本次四格中的任何一格，只用来检查口径漂移；
旧值自身的分辨率状态为 **QUANTUM UNKNOWN (legacy protocol had no registered floor)**。

本次 warm: 9.289500 us vs archived 11.488000 us; delta=2.198500 us, current admissible resolution/work=UNKNOWN; **DRIFT UNRESOLVED: current admissible quantum unavailable**。

本次 248 个 adjacent clock snapshot 为 `1297 MHz`；旧 S003 只记录运行前/后的 `2520/2445 MHz`。两种 scope 的 clock evidence 不可比较：它既不能证明新旧时钟条件相同，也不能排除时钟差异，更不能用于频率归一化。
因此 `9.2895 vs 11.488 us` 是一个**事前判据未覆盖的观察**，不能归因 kernel drift。

## 旧 warm anchors（引用，不重测）

| K | N | archived median us | archived MBU | scope |
|---:|---:|---:|---:|---|
| 8192 | 5120 | 9.920000 | 132.868664% | cache-equivalent; old warm-only single-launch protocol |
| 5120 | 8192 | 9.888000 | 133.298659% | cache-equivalent; old warm-only single-launch protocol |

## 解释边界

1. K=1024 的低利用率事前视为小工作量/setup 负控；不能仅凭它判实现有墙。
2. K=5120 若仍低，才是需要解释的主信号；解释前必须按每权重/每输出元素归一，
   不能用总指令或总字节直接比较。
3. 本报告只含 M=1，因此只报 %HBM；没有把 M>1 MFU 混进同一列。
