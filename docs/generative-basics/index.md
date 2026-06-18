# 生成模型基础

打底生成式与多模态模型（扩散、视觉编码、VAE、采样器等）的核心概念与流程，和 [LLM 基础](../llm-basics/index.md) 并列——后者偏文本自回归推理，这里偏视觉/多模态生成与理解。

## 目录

> 随学习推进逐步补充。

- [DiT 是什么，核心流程](dit.md) — Diffusion Transformer 的定义、三大构件、推理/训练流程、与 U-Net/LLM 的差异
- [ViT 是什么，核心流程](vit.md) — Vision Transformer：patchify 流程、在 VLM 里当视觉编码器、与 DiT 的异同

待补充方向：

- VAE：latent 压缩与重建，为什么扩散要在 latent 空间做
- 扩散采样器：DDPM / DDIM / DPM-Solver / Euler 的取舍
- Flow Matching / Rectified Flow：与传统扩散的关系
- Classifier-Free Guidance（CFG）原理与算力代价
- 多阶段生成管线：Thinker / Talker / Vocoder / Code2Wav 的分工

另见 [碎片知识](snippets/index.md)：速查、结论快照等零散条目。

## 如何新增一篇

1. 在 `docs/generative-basics/` 下新建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` → `生成模型基础` 下登记一行
