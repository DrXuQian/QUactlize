# ncp_flash_lib / llama.cpp：独立 Quactlize 库交接

本文件取代原“在 ncp 中实现 dense provider”的指令。不再根据
`/root/quactlize_dense.txt` 实现第二份 selector、动态执行入口或工作区管理。

## 职责边界

| 仓库 | 负责内容 |
| --- | --- |
| Quactlize | pack/dense C ABI、准入、实测策略、fallback、JIT、模块、prepass、GEMM、reducer、测试与打包 |
| ncp_flash_lib | `third_party/quactlize` 固定依赖，以及 `build.sh` 中调用 Quactlize 构建并安装产物 |
| llama.cpp | buffer/cache/TP 集成、传入局部 shape 与动态 A/C/stream、按返回大小提供 scratch、调用公开 C ABI |

ncp 不编写、不编译自己的 Quactlize provider，不复制内部 C++/Python
selector，不生成策略表，不提供转发 dense.so。现有 FA/MoE 功能不动。
Quactlize 精度测试归 Quactlize；ncp 如需在 CI 执行，仅调用上游测试入口，
不维护另一份算子测试实现。

本轮是普通 dense `MUL_MAT` 的接入：Q8_0、Q2_K～Q6_K，包含 decode/prefill。
没有把新的公开 dense 接口冒充 grouped/MoE 接口；已有 grouped 代码不变。

## ncp 要做的修改

1. 将 `third_party/quactlize` gitlink 固定到本次 Quactlize 交付提交。
   actlize 由 Quactlize gitlink 管理，保留 SSH checkout 修复；不要恢复
   HTTPS 嵌套拉取，也不要在运行时 fetch 或 checkout。
2. 保留统一入口 `bash build.sh fa moe quactlize`。quactlize target 内仅调用：

   ```bash
   bash third_party/quactlize/build.sh --library --output /absolute/fresh/package --jobs 16
   ```

   `/absolute/fresh/package` 是 ncp 构建工作区中的全新组件输出目录，
   不是已经放有 FA/MoE 的总 `build/`。`--jobs` 使用现有构建并发设置。
   SDK 默认读 `PPU_SDK`，再取 `/usr/local/PPU_SDK`；无需从 nvcc 路径剥离。
3. 从该包安装到 ncp 的 `build/`：

   ```text
   build/libquactlize_ppu.so
   build/quactlize-runtime/             # 必须完整保留、与公开 so 同级
   build/include/quactlize/dense/api.h
   build/include/quactlize/packing/api.h
   build/include/quactlize_ppu_config.h
   ```

   保存该包的 `manifest.json` 用作构建凭证。不要把 `.build` 中间目录安装进去。
   对外只有 **libquactlize_ppu.so**；私有辅助 DSOs 和按需 JIT 所需文件
   在 `quactlize-runtime/`，不是另外两个公开 pack/dense 库。
4. 删除 ncp 自写的 Quactlize dense wrapper/selector，以及对应 CMake 源文件
   和独立 pack/dense 链接目标；不要删除 FA/MoE 的源文件、依赖或构建规则。
   不要重新实现本文描述的库内流程。
5. 不设置 `QUACTLIZE_ROOT`，生产编译只使用导出的 `build/include/`。
   运行不依赖 ncp/Quactlize 原 checkout 的路径。

调用方不需要 ncp 就能使用这套安装。保留 ncp 是为了统一项目构建，不是运行时
调用必须穿过 ncp 的包装层。

## 公开 ABI 与选路契约

权威头文件为 `quactlize/dense/api.h`、`quactlize/packing/api.h`。
旧 params/info 布局维持 112/224 字节。核心入口：

```text
quactlize_ppu_dense_supported_v1
quactlize_ppu_dense_create_v1
quactlize_ppu_dense_create_with_options_v1
quactlize_ppu_dense_prepare_v1
quactlize_ppu_dense_run_v1
quactlize_ppu_dense_destroy_v1
quactlize_ppu_dense_error_v1
```

- `supported(M=0)` 是完整整数域内的权重准入，不是采样 M 阶梯；读取 header
  中的精确对齐/stride/整数边界，不能由 ncp 改写为只支持 Q4 或只支持表内 shape。
- `create_v1` 保留 PoC **dense F16 compute + F32 storage**。
  可显式调用 options 入口请求 BF16。F32 storage 不代表 F32 算术，
  F16 路径不保证所有有限 F32 值可表示；不加入 clipping。
- prepare 选择并持有 immutable plan；权重由调用方持有，run 每次传
  A/C/workspace/stream，不固化第一次的激活地址。
- 库内保持 matched small-M → 原 decode 策略 → prefill 成本策略 →
  同族 TC → 旧 canonical FQ fallback。只有旧路径未覆盖的新 shape
  才使用显式未测的通用兜底。模块错误不触发偷偷换 kernel。
- SF 每次现展开现用，full-dequant 路径保持 BF16/cuBLAS。
  reducer、转换与所有 workspace 偏移均由 Quactlize 处理。
- 默认精度、parent、Split-K、SF dequant config 和旧 adapter 都保持 PoC 语义。
  不把“通用 fallback”当作实测最优，也不强制 M1 全走 SIMT。
- 热路径不 JIT、不调优、不分配设备内存、不 D2H 或等待 GPU。
  两条 stream 使用不同 output/scratch；prepare/JIT 和 cuBLAS eager warmup
  都在 capture 和正式计时之前完成。

## llama.cpp 后续配套

`pr/kpack-dense` 的调用端已有上述六个基本函数的动态 ABI；本次没有在 llama
中再写一份 selector。要跟随新交付名称把加载/安装目标改成
`libquactlize_ppu.so`，从同一个 handle 取 pack 和 dense 符号。
旧调用端仍使用 `_pack.so` 的位置不能原样宣称已接通新产物。

保留 dense 默认开启及支持的二维权重自动选择逻辑，不新增 `-ot` 要求；
`GGML_QUACTLIZE_DENSE=0`、load-only forward 拒绝、cache、TP 和 grouped
隔离边界保持不变。只更新现有 `build.sh` 的产物安装，不新建 llama 构建脚本。

## 本轮验证与交付限制

静态对照基线：llama.cpp `abaccee355cc69169535bc142e29c339a6ae682e`。
76 个权重族 × 19 个 M 共 1,444 点全部选路一致，包含原先的 18 个旧
fallback 点；另覆盖 1,053 个 matched 行和 560 个 prefill 行（含 BF16）。
比较 route、parent、recipe、精度、prepass、endpoint/adapters，不重新 sweep。

```bash
python3 -m pytest -q tests/test_dense_provider.py tests/test_ppu_library_package.py tests/test_kpack_dynamic.py
python3 tools/build_ppu_library.py --verify --output /absolute/package
```

主机测试使用明确隔离的 device stubs，不是数值或性能证据。
真实 PPU 编译检查也不等于实际库已链接或设备验收通过。
移植的静态等价不应写成实测 latency 相同。

本地 SDK 缺少真实 wrapper 链接库，因此目前不能交付已链接的公开 so。
box/ncp 的完整 SDK 可通过上述统一入口构建，不需要旧六库 bundle。
