---
tags:
  - vllm
  - vllm-ascend
  - vllm-omni
  - CUDAGraph
  - ACLGraph
  - torch.compile
  - 图模式
---

# 图模式：eager / PIECEWISE / FULL 是什么（GPU · NPU · omni 三层串讲）

> 一个问题：**vLLM 里说的「图模式」——eager、PIECEWISE、FULL——到底是什么?在 GPU 和 NPU 上怎么对应?omni 又涉及哪一块?**
>
> 本文从「eager→编译→图捕获」的坐标系讲起，拆 `CUDAGraphMode` 的几种取值，再映射到昇腾 ACL Graph，最后看 omni 多阶段流水线怎么用。枚举与实现基于 `vllm` / `vllm-ascend` / `vllm-omni` 源码核对。

## 一、坐标系：eager → 编译 → 图捕获

一次前向，按优化程度从低到高分三档，**「图模式」是最后一档**：

| 档位 | 在干什么 | 开销 |
|---|---|---|
| **eager**（`enforce_eager=True`） | 纯 PyTorch，算子逐个由 Python 派发、逐个 launch | 最灵活、最慢 |
| **torch.compile** | Dynamo 抓 FX 图、融合算子、生成更优 kernel | 编译一次跑得快，但每步仍逐个 launch |
| **图捕获（CUDA/ACL Graph）** | 把一串 kernel 启动序列**录制成一张图**，之后**一次 replay** | 几乎零 CPU launch 开销 |

**为什么要图**：decode 每步只算 1 个 token，kernel 又小又多，瓶颈不在算力而在 **CPU 逐个 launch kernel**（launch-bound）。图捕获把「几百次 launch」压成「一次 replay」，decode 吞吐显著提升。代价：图是**静态**的——形状/地址固定，**动态形状不能进图**。

## 二、`CUDAGraphMode` 的取值（vLLM 实测枚举）

```python
NONE = 0          # 不捕获，eager 跑 forward
PIECEWISE = 1     # 分段捕获：attention 留图外，其余段进图
FULL = 2          # 整个 forward（含 attention）进一张图
FULL_DECODE_ONLY  = (FULL, NONE)        # decode→FULL，prefill/mixed→不捕获
FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # decode→FULL，prefill/mixed→PIECEWISE
```

后两个是**组合**：元组 `(decode_mode, mixed_mode)`——对**纯 decode 批**与**prefill/混合批**分别用不同模式。因为 decode 批形状统一（每请求 1 token，适合 FULL），prefill/混合批形状多变（只能 PIECEWISE 或 eager）。

### PIECEWISE：为什么要分段

vLLM 默认在 **attention 边界切图**（`compilation_config.splitting_ops`，默认即 attention 类算子）：

```
[embed → dense/matmul/norm] →┊ attention(图外) ┊→ [MLP/dense …] →┊ attention ┊→ …
   └──── 进图的段 ────┘          动态形状          └── 进图的段 ──┘
```

原因：attention 是**动态形状**重灾区（变长序列、prefix cache、chunked prefill、varlen）。塞进静态图会爆，于是把稳定的稠密段捕获、attention 留图外 eager 跑。**PIECEWISE 因此能配合 chunked prefill / 混合批**，是 V1 通用默认。它依赖 piecewise 的 torch.compile 按 `splitting_ops` 切 FX 图（`requires_piecewise_compilation`）。

### FULL：整图，要求 uniform

把 attention 也录进同一张图，launch 开销最低，但要求 **attention backend 支持被捕获 + 形状统一**——最适合 **decode-only / 均匀负载**。混合 prefill 时通常退化，这正是 `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE` 的意义：只在 decode 用 FULL，其余用 NONE / PIECEWISE。

## 三、GPU ↔ NPU：同一套抽象，两套实现

| | GPU | NPU（vllm-ascend） |
|---|---|---|
| 图原语 | CUDA Graph | **ACL Graph** = `torch.npu.NPUGraph` |
| 包装器 | vLLM 内置 cudagraph wrapper | `ACLGraphWrapper`（`vllm_ascend/compilation/acl_graph.py`） |
| 模式枚举 | `CUDAGraphMode` | **直接复用** vLLM 的 `CUDAGraphMode` |
| forward context 注入 | `cudagraph_runtime_mode=...` | `set_ascend_forward_context(..., aclgraph_runtime_mode=...)` |

关键点：**vllm-ascend 没另起炉灶，而是套用 vLLM 的 `CUDAGraphMode` 同一套语义**，只把底层 CUDA Graph 换成 NPUGraph。其注释也写明「FULL or FULL_DECODE_ONLY for mostly uniform decode workloads」，与 GPU 选型逻辑一致。`eager` 两边同义：`enforce_eager` 关掉编译+图，纯算子派发，用于调试或图不兼容兜底。

## 四、omni 涉及的部分

omni 通过平台 hook 决定「用哪种图包装器」，再在多阶段流水线上**自己加捕获逻辑**：

1. **平台 hook 选图包装器**（`NPUOmniPlatform`，见 [platforms/npu 架构导读](../vllm-omni/npu-platform-architecture.md)）：
   ```python
   get_graph_wrapper_cls() -> ACLGraphWrapper
   set_forward_context(..., cudagraph_runtime_mode=...) -> set_ascend_forward_context(..., aclgraph_runtime_mode=...)
   ```
   这是图模式在 omni 里「平台无关骨架 + 平台相关注入」的接缝（见 [平台无关/相关解耦](../vllm-omni/platform-decoupling.md)）。

2. **多阶段各自捕获**（`NPUARModelRunner.capture_model`）：在父类捕获之外，额外为 Qwen3 的 **talker_mtp**（多 token 预测）单独捕获：
   ```python
   def capture_model(self):
       mem = super().capture_model()       # vLLM/ascend 常规捕获
       self._capture_talker_mtp_graphs()   # omni 多阶段特有
   ```
   `_capture_talker_mtp_graphs` 先用 `aclgraph_runtime_mode=NONE` warmup，再用 **`CUDAGraphMode.FULL`** 对一组 `cudagraph_capture_sizes` 正式捕获——用 FULL 是因为 **MTP decode 形状统一**，正好满足 FULL 的 uniform 要求。

> 一句话：**omni 的图模式抽象沿用 vLLM/ascend，但因为它是多阶段（Thinker/Talker/Code2Wav…），每个 stage 可独立决定捕不捕、用哪种模式**——talker_mtp 这种均匀 decode 子模块走 FULL，带变长 attention 的主体走 PIECEWISE。

## 五、怎么选

| 场景 | 建议模式 |
|---|---|
| 调试 / 新模型 bring-up / 图老报错 | **eager**（`enforce_eager`） |
| 通用在线服务（含 chunked prefill / 混合批） | **PIECEWISE**（V1 默认，稳） |
| 纯 decode、均匀负载、要极致吞吐 | **FULL** / **FULL_DECODE_ONLY** |
| 兼顾 decode 快 + prefill 稳 | **FULL_AND_PIECEWISE** |

配置入口：`--enforce-eager` 或 `compilation_config.cudagraph_mode`；NPU 同样经 ascend forward context 落到 `aclgraph_runtime_mode`。

## 小结

| 模式 | attention 在哪 | 适用 | 形状要求 |
|---|---|---|---|
| eager / NONE | 图外（无图） | 调试、兜底 | 任意 |
| PIECEWISE | 图外，其余进图 | 通用、混合批 | 段内稳定即可 |
| FULL | 图内 | decode-only、均匀 | 必须统一 |
| FULL_DECODE_ONLY | decode 图内 / prefill 无图 | decode 重 | decode 统一 |
| FULL_AND_PIECEWISE | decode 图内 / prefill 分段 | 兼顾 | 分别满足 |

!!! info "说明"
    `CUDAGraphMode` 枚举、`splitting_ops`、`ACLGraphWrapper`、omni `capture_model` 均基于源码核对，行号随版本漂移，以实际仓库为准。
