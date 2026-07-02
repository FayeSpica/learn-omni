---
tags:
  - NPU
  - Ascend
  - 漂移
  - 对齐
---

# 对齐漂移总索引

> "跟上社区节奏"对 NPU 适配来说 = **盯住若干条会漂移的轴**:上游 vLLM 或 vLLM-Ascend 一改,omni 侧就欠债。本页是所有漂移轴的**总表**,每条指向深挖笔记与巡检命令;runner 那条最细的账在 [runner-compare / drift-log](runner-compare/drift-log.md)。
> 基线:`vllm 6c427dd40 · vllm-ascend 12c8da7a · vllm-omni 724f5d13`。

## 漂移轴总表

| 轴 | 上游/ascend 源 | omni 承接方式 | 当前风险 | 深挖 |
|---|---|---|---|---|
| **model_runner** | `GPUModelRunner` / ascend `NPUModelRunner` | 菱形多继承 `OmniNPUModelRunner(OmniGPU, NPU)` | ⚠️ `_dummy_run` 手抄漏 3 参数 + SP all-gather 待核实 | [runner-compare](runner-compare/index.md) |
| **平台层** | `vllm/platforms/` | `vllm_omni/platforms/npu/` | 平台无关/相关边界仍在演进 | [platform-decoupling](../vllm-omni/platform-decoupling.md) |
| **图捕获** | cudagraph → aclgraph | 四方各自重写 `_dummy_run` | ⚠️ is_tracing 失灵、嵌套捕获、PIECEWISE cap(#4674) | [npu-gpu-graph](../vllm-omni/npu-gpu-graph-in-runner.md) |
| **EPLB** | vllm / vllm-ascend | omni 不自实现,仅继承透传判断 | 判断点分散,易漏 | [eplb-inheritance](../vllm-omni/snippets/eplb-inheritance.md) |
| **量化** | vllm-ascend 量化栈 | omni factory 透传 | 昇腾低精度格式随代次变化 | [ascend-quantization](../vllm-ascend/snippets/ascend-quantization.md) |
| **HF 依赖** | transformers | omni 依赖其 tracing 行为 | NPU 上 is_tracing 语义不同 | [transformers-is-tracing-npu](../vllm-omni/transformers-is-tracing-npu.md) |

## 定期巡检(每周 / 每次拉基线)

```bash
cd ~/git/vllm_omni

# 1) runner 结构矩阵重生成(结构漂移一目了然)
OMNI_SRC=$PWD python3 ~/git/FayeSpica/learn-omni/tools/runner_matrix.py \
  > ~/git/FayeSpica/learn-omni/docs/npu-adaptation/runner-compare/_matrix.generated.md

# 2) ascend NPUModelRunner 自基线以来的提交(要跟的 delta 源)
git -C vllm-ascend log --oneline --since="6 weeks ago" -- vllm_ascend/worker/model_runner_v1.py

# 3) 上游 GPU runner 变化(含 v1→v2)
git -C vllm log --oneline --since="6 weeks ago" -- vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu/

# 4) 针对性:某方法的逐行演进(以 _dummy_run 为例)
git -C vllm-ascend log --oneline -10 -L :_dummy_run:vllm_ascend/worker/model_runner_v1.py
```

## 待办化流程

巡检发现的新漂移 → 记进 [runner-compare / drift-log](runner-compare/drift-log.md)(runner 类)或本页总表(其它轴)→ 转成 [适配待办](task-list.md) 一行 → 回填 [#4610](https://github.com/vllm-project/vllm-omni/issues/4610) / 下一版 checklist。

## 长期漂移轴:Runner V1 → V2

最大的单条债:上游与 ascend 均已有 v2 runner,omni NPU 仍 v1。详见 [发布看板 §三](release-board.md#v024)。
