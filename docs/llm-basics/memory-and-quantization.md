---
tags:
  - LLM
  - 基础
  - 显存
  - 量化
---

# 显存账本与量化(骨架)

> 🏗 学习骨架。所属 [LLM 基础](index.md) § 5 · 效率与部署。

## 学习目标

学完能:把一张卡上「装得下多大模型、多长上下文、多少并发」算清 —— 权重/KV/激活三块分别怎么估;再理解量化如何用「精度换显存/带宽」。

## 带着问题读

- 显存三大块:**权重、KV Cache、激活**,各自的估算公式?哪块随并发/上下文线性增长?
- 推理时激活显存为什么远小于训练(无需存反向图)?
- 量化对象有三类:**权重量化 / KV 量化 / 激活量化**,各自省的是什么(显存?带宽?),代价是什么(精度?算子支持?)?
- 结合 [roofline](prefill-decode-roofline.md):KV 量化为什么对 decode 阶段特别值?

## 要点提纲(待填)

- 权重显存 = 参数量 × dtype 字节;KV 显存见 [专题](kv-cache-per-token.md)
- 激活显存量级与峰值点
- 量化格式:INT8 / FP8 / AWQ / GPTQ 概览
- 精度-显存-吞吐三角权衡

## 关联

- 定量前置:[KV Cache 显存](kv-cache-per-token.md)、[roofline](prefill-decode-roofline.md)
- 昇腾侧:[昇腾量化特性速查](../vllm-ascend/snippets/ascend-quantization.md)、[代次与低精度格式](../vllm-ascend/snippets/ascend-generations-low-precision.md)
