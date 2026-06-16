# vllm-ascend（昇腾）

记录 vLLM 在昇腾 NPU 上的适配层 `vllm-ascend`：Platform / Worker 适配、量化方案、算子与设备管理等平台相关的内容。

## 目录

> 随着学习推进逐步补充。建议每个主题单独建一篇 `.md`，并在下方与 `mkdocs.yml` 的 `nav` 中登记。

- [昇腾(vllm-ascend)量化特性支持速查](ascend-quantization.md) — W8A8 / W4A8 / W4A4 / MXFP8 / KV C8 支持矩阵

## 如何新增一篇笔记

1. 在 `docs/vllm-ascend/` 下新建 Markdown 文件，例如 `docs/vllm-ascend/platform.md`
2. 在 `mkdocs.yml` 的 `nav` → `vllm-ascend` 下添加一行：

   ```yaml
   - Platform 适配: vllm-ascend/platform.md
   ```

3. 本地预览：`mkdocs serve`，推送到 `main` 后自动部署
