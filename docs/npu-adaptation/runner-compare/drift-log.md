---
tags:
  - vllm-omni
  - vllm-ascend
  - vllm
  - model_runner
  - 漂移
  - release
---

# 漂移日志:runner 对齐的版本 delta

> Release owner 视角。每次更新对齐基线(重跑 [`runner_matrix.py`](index.md#regen))后,把矩阵**新出现的分叉**转成带 owner 的行动项,并回填 [#4610](https://github.com/vllm-project/vllm-omni/issues/4610)。
> 这张表是 spine 矩阵的时间维:矩阵回答"现在差在哪",本表回答"这一版新欠了什么、谁来还"。

## 当前对齐基线

| repo | SHA | 版本 | 备注 |
|---|---|---|---|
| vllm | `6c427dd40` | dev | 上游真相源 |
| vllm-ascend | `12c8da7a` | dev | 对齐目标 |
| vllm-omni | `724f5d13` | v0.23 线 | 本仓 |

> 更新基线时同步改这里,并在下方追加一行 delta 记录。

## Delta 记录(最新在上)

| 日期 | 基线区间 | 上游/ascend 变化 | omni 状态 | 动作 | owner | #4610 |
|---|---|---|---|---|---|---|
| 2026-07-01 | 建档 | —(首次快照,110 个分叉方法) | 见 spine 矩阵 | 逐个 L2 核对 MRO 分叉 | @FayeSpica | 关联 |
| 2026-07-01 | 建档 | ascend `_dummy_run` 的 `set_ascend_forward_context` 带 `has_sinks`/`input_ids`/`eplb_heat_collection_status` | omni NPU 手抄版**缺这 3 个参数**,且直调 `self.model()` 跳过 `_model_forward`(SP all-gather) | 核实 SP>1 下 aclgraph 是否漏捕 all-gather;评估补齐参数 | @FayeSpica | ➕ 待新增 |
| 2026-07-01 | 建档 | ascend `_prepare_inputs` 返回 **4 元组**(+PCP total、+压缩 KV list) | omni NPU 经 MRO 正确继承 | 保持 OmniGPU 不 override 此方法 | @FayeSpica | 关联 |
| _待填_ | v0.23→v0.24 | Runner V2 迁移(#1770)上游推进 | omni NPU 仍 v1 血统 | 立 tracking 子项 | @gcanlin | ➕ 新增 |

## 定期巡检命令(每周/每次拉基线)

```bash
cd ~/git/vllm_omni

# 1) 重生成结构矩阵
OMNI_SRC=$PWD python3 ~/git/FayeSpica/learn-omni/tools/runner_matrix.py \
  > ~/git/FayeSpica/learn-omni/docs/npu-adaptation/runner-compare/_matrix.generated.md

# 2) ascend NPU runner 自上次基线以来的提交(要跟的 delta 源)
git -C vllm-ascend log --oneline --since="6 weeks ago" -- vllm_ascend/worker/model_runner_v1.py

# 3) 上游 GPU runner 变化(omni 迟早要跟,含 v2 迁移)
git -C vllm log --oneline --since="6 weeks ago" -- vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu/
```

## 长期漂移轴

- **Runner V1 → V2**:上游已有 `vllm/v1/worker/gpu/model_runner.py`,ascend 已有 `worker/v2/model_runner.py`,omni NPU 仍 v1。这是最大的对齐债,建议在 #4610 或后续 release checklist 单列 tracking。
