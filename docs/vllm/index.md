# vLLM 内核

记录 vLLM 推理引擎本身的原理与机制：引擎骨架、算子分发、编译与中间表示等平台无关的核心设计。

## 目录

> 随着学习推进逐步补充。建议每个主题单独建一篇 `.md`，并在下方与 `mkdocs.yml` 的 `nav` 中登记。

- [vLLM IR 是什么：从 CustomOp 的困境说起](vllm-ir-and-customop.md)

## 如何新增一篇笔记

1. 在 `docs/vllm/` 下新建 Markdown 文件，例如 `docs/vllm/scheduler.md`
2. 在 `mkdocs.yml` 的 `nav` → `vLLM` 下添加一行：

   ```yaml
   - 调度器: vllm/scheduler.md
   ```

3. 本地预览：`mkdocs serve`，推送到 `main` 后自动部署
