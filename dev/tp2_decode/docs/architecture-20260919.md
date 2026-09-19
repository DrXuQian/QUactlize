# Decode 选路架构审查与渐进整理计划

审查基线：`2d72ee58e845e07c3128a30a78823d47a9a19aab`，2026-09-19。
范围：Quactlize 的 small-M 策略、实现分派、包库存与融合接口；不修改 device
kernel、生产选择或当前运行时 pin。来自独立只读 review 与本地代码交叉核查。
另一次外部 CLI 审查因认证过期未开始，不能标为已获外部模型确认。

## 结论

需要渐进整理，不需要推倒重写。主要问题不是精确表大、kernel 家族多，而是：

1. 同一次最终选择分散在 matched 表、Q8 replacement、kernel 特派与 fusion 中。
2. config 只表达部分几何，同一个 tuple 不一定表示同一个实际实现。
3. 运行时选路、构建/JIT 预热规划、包库存检查并非完全使用同一事实源。

优先做 host 侧的等价重构，建立可验证的最终选择记录。后续性能推广单独提交，
不把“改组织方式”和“换最优 kernel”混在一次变更里。

## 具体发现

| 优先级 | 证据 | 影响与边界 | 建议 |
|---|---|---|---|
| P1 | `dispatch/binding.cpp::quactlize_kpack_dispatch_query_smallm_v3` 先执行 `matched::select`，再调用 `q8_vector::select/select_tc/select_bucket` | 新优化通过运行时补丁顺序生效；bucket 必须重放 donor 的 replacement。当前继承修正有用，但不适合无限叠加 | 生成已折叠 replacement 的最终策略表，保留 donor 与证据链，第一版不改变 donor/tie-breaking |
| P1 | `execution/simt.h` 的 recipe 只有 variant/columns/warps/values/split；`simt_q8_vector.cuh::launch_v2` 和 `simt_kernel.cuh::launch_v2` 另选 hoist、fixed N/K、Changes 与 reducer | tuple 相同不代表测量时与执行时是同一个实现。已有整体 source/binary hash，但逐选择的实现关联不完整 | 显式记录 implementation ID/revision、实际 producer/reducer、固定参数；不必改公开 C ABI |
| P1 | `tools/kpack_native_policy.cpp` 仅调用 `select/select_decode_tc` 与 compute proposal；`build_kpack_dispatch.py::plan` 用它生成 parent 集合 | 不包含运行时 v3 的完整 matched/Q8 逻辑；不能仅凭此 plan 证明实际所需模块已打包。JIT/prewarm 可缓解，不能等同于共享事实源 | runtime、build/prewarm、explain 共用纯 host selector |
| P2 | v3 中的 `simt::query_v2` 检查结构合法性；`execution/simt.cpp` 还检查生成的 `qkg_simt_supported_*` | 合法 recipe 与某个已编译包支持的 recipe 是两件事。不是已证实现包缺失，而是裁剪/旧包组合的部署风险 | 从实际编译 inventory 生成 capability manifest；打包时核对所有最终 recipe，运行时保留防御校验 |
| P2 | `fusion/validation.hpp::select` 的 key 少于实际 call 的 storage、channels、round_projection、row mapping 等语义；`fusion/simt.cu` 又有 exact 特派 | “已测”配置的语义边界依赖 caller 隐式保证；普通 projection 的收益不能充当 paired GateUp+SwiGLU 的收益 | 将融合链作为独立 operator scope，显式区分 logical N 与 physical 2N、舍入与 endpoint |
| P2 | `binding.cpp` 同时处理策略、JIT/DSO、缓存、ABI、MoE lifecycle；cache 的 decode 参数还有 `2+channels`、`20+channels` 等历史编码 | 当前未发现碰撞，但维护者需同时理解多套 API 历史才能改动 | 等价测试齐全后分离纯选择与资源装载；内部用明确 selection profile，外部 ABI 保持兼容 |
| P3 | 全表再生成测试已有覆盖，但部分 fallback 测试依赖字符串切分和调用文本 | 适合临时防回退，不足以证明最终实现身份，也阻碍无行为变化的整理 | 增加 executable final-decision 快照，再替代脆弱的结构断言；ISA 检查继续保留 |

复查未发现“305 条旧 exact 已被 bucket 意外覆盖”的具体反例，不将其写成现存 bug。
独立审查核对了这些旧记录在 matched exact/open 中的对应关系。

## 最小目标结构

`既有 ABI → 规范化请求 → 纯选路 → 装载/prepare → run`

只保留三项职责，不建立新的多层插件框架：

| 职责 | 输入/输出 | 不应该做的事 |
|---|---|---|
| Contract / capability | 请求、layout、精度、结构约束 → 合法性及库存能力 | 不根据性能数据偷偷改精度或 layout |
| Selection / catalog | 合法请求、策略表 → 最终 recipe + provenance | 不加载 GPU 库、不编译、不做在线搜索 |
| Resolver / execution | 最终 recipe → 已验证模块/handle/资源 → launch | 不拥有第二套 heuristic，不在装载失败时偷偷换 tactic |

内部请求至少区分 operator、qtype、canonical/paired mapping、local shard N/K、
tokens/experts/topk/channels、external storage、compute/accumulator、projection
rounding 与必要的 stride/alignment 类别。已有请求字段可复用，不必为每个概念新增类。
不要把 GPU router histogram 读回 CPU 作为选路条件；cache regime 是证据属性，
不是运行时能凭空知道的 shape 属性。

最终 recipe 应同时说明：

- 算法及几何：TC/SIMT、tile、split、grid。
- 实现：reader/hoist/fold/fixed specialization、reducer、prepare/finish 版本。
- 语义与能力：格式、精度、operator、布局与对齐范围。
- 选择依据：exact/bucket/structural fallback、donor、测量 scope、证据 revision。
- 编译身份：源码/生成器、SDK/compiler/flags、架构/ABI、binary digest。

策略 revision 与 kernel build key 分开：仅调整测量排名，不应重新编译未变化的实现。

## 精确表和 fallback 的关系

1. 同语义、已准入的 exact 最终 winner 优先。
2. 对明确未解决/禁止推广的请求，保留其规定的旧合法路线或拒绝；不能借 bucket
   把失败或缺测伪装成已准入。
3. 表外使用有范围限制、同语义族的预测配置，标签必须仍是 predicted。
4. 通用优化进入实现族的默认代码，使 exact 和 fallback 都受益；固定尺寸实测
   winner 作为少量显式例外保留。配置外推与通用实现升级是两个维度。
5. 最终无合法/可用实现时返回明确原因，由既有调用契约处理；不静默降精度或换格式。

不要统一 all-SIMT：上一轮 Q6 TP2 head 的 TC 仍胜过 SIMT。不要为了快 prepare
强制换掉更快的 matmul。paired 与 canonical 不是可互换的字节解释。

## 分阶段 TODO

| ID | 工作 | 验收 | PPU 需求 |
|---|---|---|---|
| A0 | 完成当前 fallback gate，冻结候选和旧最小值；不混入架构改动 | 表外/旧 shape、M1..8 数值、冷态完整调用、prepare、ACU；性能失败保留旧 winner | 需要，独立于本计划 |
| A1 | final-decision 快照与内部 request/recipe 类型，先做等价 ABI adapter | 1842 exact、634 bucket、30 exclusions、16 missing、305 legacy、Q8 overlays、Q4 分支与边界请求；包括实现 ID、donor、状态与 workspace | 纯 host；可本地完成 |
| A2 | 生成 effective policy，折叠 overlay；runtime/build/prewarm 共用 selector | 最终实现、配置、donor、状态不变；不重拟合 bucket；包库存覆盖选择闭包 | host 等价检查本地；发布前集成 smoke |
| A3 | 显式化 hoist/fold/fixed/reducer/fusion 身份，统一 capability 描述 | 逐选择可追溯到 source/binary/measurement；保留已有例外与 generic 改善 | 若改变 device lowering、资源或归约顺序，须 PPU |
| A4 | 按职责拆 binding.cpp，删已证明无依赖的重复胶水/flag | ABI、capture/lifetime、alias、冷启动、拒绝行为与缓存隔离不回退 | host 为主；GPU 生命周期需 smoke |
| A5 | 单独推广新的 TP2 exact winner 与融合优化，发布对应 runtime/caller | 数值→完整组件性能→暖态模型 ABBA/Asys；不拿编译通过代替性能 | 需要 |

顺序：A0 保持当前实验不变；A1/A2 是首批架构工作；A3/A4 分小步执行；A5
按独立性能证据推广。先不重命名全部 device 模板，不动离线格式，不改 public ABI。

等价重构的验收不是“单元测试都绿”：要比较旧/新 final-decision 快照及完整语义。
换 TC/SIMT 或 reduction 几何则不要求无根据的 bitwise 相等，而是保留独立 oracle、
BF16 大有限值、舍入边界、错误 partial 布局负控、弱对齐、router/map、alias 与 replay。
性能始终比较冻结的完整调用最小值，不能只挑更慢的旧 generic 作为 baseline。

## 本轮边界

这是审查与计划，不是架构重构已完成。当前 box 候选仍按现有真实 production
launcher/selector 构建。下一步最小且有价值的改动是 A1/A2，而不是再增加一套
运行时 heuristic 或把所有精确结果压成一条未经验证的规则。
