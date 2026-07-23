# vLLM · 碎片知识

vLLM 内核相关的零散知识：速查、排错、源码片段、结论快照。

- [cuda.Event / npu.Event：异步时间线上的路标](cuda-npu-event.md) — 计时/跨 stream 依赖/非阻塞查询三大用途，与 async output 的关系

## 如何新增

1. 在 `docs/vllm/snippets/` 下新建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` → `vLLM` → `碎片知识` 下登记一行
