# LLM 基础

以**问题驱动**的方式打底：从一道道「给定条件、动手算/推导」的题出发，把 LLM 推理的显存、带宽、算力、并发等基础吃透。每题都尽量做到「公式可推导、数字可复算、坑能识别」。

## 题目序列

> 随学习推进逐步补充。

1. [给定 config.json 与 H100，算「每生成一个 Token」的 KV Cache 显存](kv-cache-per-token.md) — KV Cache 公式推导、GQA/MLA 差异、H100 容量换算

待补充方向：

- 一次请求的总 KV（prefill + decode）与并发数估算
- 为什么解码是访存受限（HBM 带宽），prefill 是算力受限
- 模型权重显存、激活显存怎么估
- 吞吐 vs 延迟：batch size 的取舍
- 量化（权重 / KV / 激活）对显存与精度的影响

另见 [碎片知识](snippets/index.md)：速查、结论快照等零散条目。

## 如何新增一题

1. 在 `docs/llm-basics/` 下新建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` → `LLM 基础` 下登记一行
