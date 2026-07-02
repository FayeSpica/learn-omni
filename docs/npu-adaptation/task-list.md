---
tags:
  - NPU
  - Ascend
  - task
  - 社区
---

# 适配待办

> 接社区 open issue 的 NPU 适配清单,按"契合度 × 上手难度"排。已被认领的单列避坑区。快照日期见各条;认领状态会变,动手前先在 GitHub 复核。
> 刷新命令见页尾。

## 优先:release 阻塞项(源自 [#4610](https://github.com/vllm-project/vllm-omni/issues/4610))

- ⬜ **升级 `NPUModelRunner` 对齐 vLLM-Ascend** —— 本版长杆,无 PR。子任务已在 [runner-compare](runner-compare/index.md) 拆好:
    - `_dummy_run` 补齐 3 个 ascend context 参数 + 核实 SP>1 下 all-gather 捕获([详情](runner-compare/graph-capture.md))
    - `_prepare_inputs` 4 元组契约守护([详情](runner-compare/prepare-inputs.md))

## Tier 0 · 小而正中方向(未认领,可抢)

| # | 标题 | 类型 | 为什么适合 |
|---|---|---|---|
| [#4800](https://github.com/vllm-project/vllm-omni/issues/4800) | Qwen3-TTS CustomVoice uploaded_voices 在 NPU 失败 | Bug | TTS+NPU,复现型,合并率高 |
| [#4258](https://github.com/vllm-project/vllm-omni/issues/4258) | vllm-ascend 版本兼容性文档 | Doc | 破冰;顺手建版本对齐表 |

## Tier 1 · 中等,建立"NPU 专家"标签(未认领)

| # | 标题 | 类型 | 备注 |
|---|---|---|---|
| [#4042](https://github.com/vllm-project/vllm-omni/issues/4042) | NPU RoPE 统一走 `torch_npu.npu_rotary_mul` | RFC | 局部 kernel 清理,先对齐方案 |
| [#4814](https://github.com/vllm-project/vllm-omni/issues/4814) | Wan2.2 ulysses 并行 + cache-dit 冲突 [ASCEND] | Bug | 补 diffusion/distributed 认知 |
| [#3188](https://github.com/vllm-project/vllm-omni/issues/3188) | Qwen-Image 双卡 TP2/DP2/PP2 不提速 | Perf | 带 good-first-issue 背书 |

## Tier 2 · 中长期(独占领地,工作量大)

| # | 标题 | 备注 |
|---|---|---|
| [#3565](https://github.com/vllm-project/vllm-omni/issues/3565) | Enable NPU CI Tests | 多阶段 pipeline,建议先认领一个子项 |
| [#1770](https://github.com/vllm-project/vllm-omni/issues/1770) | Migrate to Model Runner V2 | 最大对齐债,见 [发布看板](release-board.md#v024) |

## ⛔️ 已认领(别重复,可跟进 / review)

| # | 标题 | 认领者 |
|---|---|---|
| [#2759](https://github.com/vllm-project/vllm-omni/issues/2759) | NPU ring attention | gcanlin(你有 ring_pytorch_attn 笔记,可 review) |
| [#3218](https://github.com/vllm-project/vllm-omni/issues/3218) | NPU Qwen3-TTS eager 音频损坏 | ChefWu551 / gcanlin |
| [#3842](https://github.com/vllm-project/vllm-omni/issues/3842) | CI pytest 版本约束 | fallintoplace |

## 跟节奏路标(读,不做)

- [#4610](https://github.com/vllm-project/vllm-omni/issues/4610) NPU v0.23.0 checklist · [#2223](https://github.com/vllm-project/vllm-omni/issues/2223) NPU Q2 Roadmap

## 刷新本清单

```bash
R=vllm-project/vllm-omni
gh issue list -R $R --state open --search "NPU OR Ascend in:title,body" --limit 60 \
  --json number,title,labels,assignees,updatedAt \
  --jq '.[] | select([.assignees[]]|length==0) | "#\(.number) \(.updatedAt[0:10]) \(.title[0:70])"'
```

> 只列**未认领**的;认领状态是动态的,以 GitHub 为准。
