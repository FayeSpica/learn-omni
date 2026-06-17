# vllm-ascend · 碎片知识

昇腾适配相关的零散知识：速查、排错、源码片段、结论快照。

- [昇腾(vllm-ascend)量化特性支持速查](ascend-quantization.md) — W8A8 / W4A8 / W4A4 / MXFP8 / KV C8 支持矩阵
- [昇腾代次与原生低精度格式(A2/A3/A5·950)](ascend-generations-low-precision.md) — 各硬件代次原生支持的 FP8 / MXFP8 / HiF8 / MXFP4 / INT8 对比与 950 算力表

## 如何新增

1. 在 `docs/vllm-ascend/snippets/` 下新建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` → `vllm-ascend` → `碎片知识` 下登记一行
