# Omni 学习笔记

这里系统性地记录学习 omni 的过程。

## 目录

> 随着学习推进逐步补充。建议每个主题单独建一篇 `.md`，并在下方与 `mkdocs.yml` 的 `nav` 中登记。

- [vLLM / vllm-ascend / vllm-omni 模块导图与 Omni NPU 适配研究方向](vllm-omni-npu.md)
- [Qwen3-Omni 在 NPU 上是怎么跑起来的](qwen3-omni-npu.md)
- [Omni 平台无关/相关解耦：现状与演进](platform-decoupling.md)

## 如何新增一篇笔记

1. 在 `docs/omni/` 下新建 Markdown 文件，例如 `docs/omni/getting-started.md`
2. 在 `mkdocs.yml` 的 `nav` → `Omni 学习笔记` 下添加一行：

   ```yaml
   - 快速开始: omni/getting-started.md
   ```

3. 本地预览：`mkdocs serve`，推送到 `main` 后自动部署
