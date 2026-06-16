# vllm-omni · 碎片知识

vllm-omni 相关的零散知识：速查、排错、源码片段、结论快照。

- [npu_model_runner 的上游适配困境：为什么每次都要跟 GPU 联动，怎么解耦](npu-runner-decoupling.md) — 从 PR #4454 拆解三套 runner 的继承断链与解耦方案

## 如何新增

1. 在 `docs/vllm-omni/snippets/` 下新建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` → `vllm-omni` → `碎片知识` 下登记一行
