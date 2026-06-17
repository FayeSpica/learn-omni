---
hide:
  - navigation
  - toc
---

# Learn Omni

欢迎来到我的学习与知识库。先打 **LLM 基础**，再按项目分三块：**vLLM 内核 → vllm-ascend 昇腾适配 → vllm-omni 任意模态框架**，从下到上分层。每块下都设「碎片知识」收录速查、排错与源码片段。

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } __LLM 基础__

    ---

    问题驱动打底：KV Cache 显存、带宽/算力、并发估算等推理基本功。

    [:octicons-arrow-right-24: 进入笔记](llm-basics/index.md) · [碎片知识](llm-basics/snippets/index.md)

-   :material-image-multiple:{ .lg .middle } __生成模型基础__

    ---

    扩散与多模态生成打底：DiT、VAE、采样器、flow matching、生成管线。

    [:octicons-arrow-right-24: 进入笔记](generative-basics/index.md) · [碎片知识](generative-basics/snippets/index.md)

-   :material-engine:{ .lg .middle } __vLLM__

    ---

    硬件无关的推理引擎内核：引擎骨架、算子分发、编译与中间表示（IR）。

    [:octicons-arrow-right-24: 进入笔记](vllm/index.md) · [碎片知识](vllm/snippets/index.md)

-   :material-chip:{ .lg .middle } __vllm-ascend__

    ---

    vLLM 在昇腾 NPU 上的适配层：Platform/Worker 适配、量化、算子与设备管理。

    [:octicons-arrow-right-24: 进入笔记](vllm-ascend/index.md) · [碎片知识](vllm-ascend/snippets/index.md)

-   :material-layers-triple:{ .lg .middle } __vllm-omni__

    ---

    任意模态服务框架：多引擎/多阶段编排、平台解耦，及 Qwen3-Omni 等模型落地。

    [:octicons-arrow-right-24: 进入笔记](vllm-omni/index.md) · [碎片知识](vllm-omni/snippets/index.md)

</div>

---

## 关于本站

- 使用 [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/) 构建
- 源码托管在 [GitHub](https://github.com/FayeSpica/learn-omni)，推送到 `main` 自动部署
- 支持全文搜索、深色模式、代码高亮、Mermaid 图与数学公式
