---
tags:
  - NPU
  - Ascend
  - release
  - checklist
---

# 发布看板:NPU release v0.23.0

> **真相源是 [issue #4610](https://github.com/vllm-project/vllm-omni/issues/4610)**;本页是本地可勾选镜像,便于把 checklist 状态与 [漂移索引](alignment-drift.md)、[适配待办](task-list.md) 串起来。基线:`vllm 6c427dd40 · vllm-ascend 12c8da7a · vllm-omni 724f5d13`。
> 关联 roadmap:[#2223 NPU 2026 Q2](https://github.com/vllm-project/vllm-omni/issues/2223)。

## 一、代码对齐项

| 状态 | 项 | PR | 备注 |
|:---:|---|---|---|
| ✅ | 对齐上游最新 `GPUModelRunner` | [#4454](https://github.com/vllm-project/vllm-omni/pull/4454) | 已合并 |
| ⬜ | **升级 `NPUModelRunner` 对齐 vLLM-Ascend** | — | **本版长杆,唯一无 PR 的开放项** |
| ✅ | 更新 v0.23.0 文档 & Dockerfile | [#4602](https://github.com/vllm-project/vllm-omni/pull/4602) | 已合并 |

!!! danger "开放项即 release 阻塞项"
    "升级 `NPUModelRunner` 对齐 vLLM-Ascend" 没有 PR。具体差异已在 [runner-compare](../vllm-omni/runner-compare/index.md) 逐方法拆过,可直接转成子任务:

    - `_prepare_inputs`:ascend 返回 **4 元组**(+PCP total、+压缩 KV list),保持 OmniGPU 不 override → [详情](../vllm-omni/runner-compare/prepare-inputs.md)
    - `_dummy_run`:omni NPU 手抄版**缺 3 个 ascend context 参数** + 直调 `self.model()` 跳过 SP all-gather,**待 SP>1 核实** → [详情](../vllm-omni/runner-compare/graph-capture.md)

## 二、重点模型验证(基于 [vllm-ascend#10278](https://github.com/vllm-project/vllm-ascend/pull/10278))

| 状态 | 模型 | PR |
|:---:|---|---|
| ✅ | Qwen3-Omni | [#4519](https://github.com/vllm-project/vllm-omni/pull/4519) |
| ✅ | Qwen3-TTS | — |
| ✅ | Hunyuan-Image 3.0 · AR | [#4386](https://github.com/vllm-project/vllm-omni/pull/4386) |
| ✅ | Hunyuan-Image 3.0 · DiT | [#4712](https://github.com/vllm-project/vllm-omni/pull/4712) |
| ✅ | Hunyuan-Image 3.0 · AR + DiT | [#4712](https://github.com/vllm-project/vllm-omni/pull/4712) |
| ✅ | Wan2.2 | — |
| ✅ | Qwen-Image | — |

> 模型验证已全绿;release 就绪度取决于上面 §一 那个开放的 runner 对齐项。

## 三、下一版前瞻(v0.24 / 后续) { #v024 }

| 优先级 | 项 | 依据 |
|:---:|---|---|
| 高 | **Runner V1 → V2 迁移** | 上游已有 `vllm/v1/worker/gpu/model_runner.py`,ascend 已有 `worker/v2/model_runner.py`,omni NPU 仍 v1 血统。RFC [#1770](https://github.com/vllm-project/vllm-omni/issues/1770) |
| 中 | NPU CI 门禁补齐 | [#3565](https://github.com/vllm-project/vllm-omni/issues/3565) |

> 建议在 #4610 或新开的 v0.24 checklist 里为 Runner V2 单列 tracking 子项,别等上游删 v1 才被动迁移。
