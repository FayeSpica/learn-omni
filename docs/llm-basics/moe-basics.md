---
tags:
  - LLM
  - 基础
  - MoE
  - 专家并行
---

# MoE 基础:稀疏专家与路由(骨架)

> 🏗 学习骨架。所属 [LLM 基础](index.md) § 2 · 现代变体。

## 学习目标

学完能:说清 MoE 为什么能「参数量大但激活量小」,理解 Top-K 路由、专家负载不均问题,并接上工程侧的 [EPLB](../vllm-omni/snippets/eplb-inheritance.md)。

## 带着问题读

- Dense FFN 换成 MoE 后,**总参数** vs **每 token 激活参数** 各是多少?为什么这是 MoE 的核心卖点?
- 门控(gating)如何为每个 token 选 Top-K 专家?为什么是「token 选专家」而非反过来?
- **负载不均**从哪来(热专家/冷专家)?为什么在专家并行(EP)下它会拖慢整个 step?
- 训练期的辅助均衡损失 vs 推理期的动态重排(EPLB),分别解决什么?

## 要点提纲(待填)

- MoE 层结构:router + N experts + Top-K
- 容量因子 capacity factor、token drop
- 专家并行 EP 的通信(all-to-all dispatch/combine)
- 负载均衡:aux loss(训练)/ EPLB(推理)

## 关联

- 工程/NPU:[EPLB 工作原理与 omni 的继承透传](../vllm-omni/snippets/eplb-inheritance.md)
- 并行维度:[EP/DP/TP/SP(从 FusedMoE 讲起)](../vllm/ep-dp-tp-sp-fused-moe.md)
