# vllm-omni

记录在 vLLM 之上构建的任意模态服务框架 `vllm-omni`：多引擎/多阶段编排、平台解耦、核心组件与请求流转，以及 Qwen3-Omni 等具体模型在 NPU 上的落地。

## 目录

> 随着学习推进逐步补充。建议每个主题单独建一篇 `.md`，并在下方与 `mkdocs.yml` 的 `nav` 中登记。

- [vLLM / vllm-ascend / vllm-omni 模块导图与 Omni NPU 适配研究方向](vllm-omni-npu.md)
- [Qwen3-Omni 在 NPU 上是怎么跑起来的](qwen3-omni-npu.md)
- [Omni 平台无关/相关解耦：现状与演进](platform-decoupling.md)
- [以 Qwen3-Omni 拆解 vllm-omni 核心组件与请求流转](components-request-flow.md)

另见 [碎片知识](snippets/index.md)：
- [npu_model_runner 的上游适配困境与解耦](snippets/npu-runner-decoupling.md) — 从 PR #4454 拆解三套 runner 的继承断链

## 如何新增一篇笔记

1. 在 `docs/vllm-omni/` 下新建 Markdown 文件，例如 `docs/vllm-omni/getting-started.md`
2. 在 `mkdocs.yml` 的 `nav` → `vllm-omni` 下添加一行：

   ```yaml
   - 快速开始: vllm-omni/getting-started.md
   ```

3. 本地预览：`mkdocs serve`，推送到 `main` 后自动部署
