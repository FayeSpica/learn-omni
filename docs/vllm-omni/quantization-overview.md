---
tags:
  - vllm-omni
  - 量化
  - FP8
  - INT8
  - MXFP4
  - MXFP8
  - W4A16
  - 骨架待填
---

# 量化全景：从 factory 到各 config（骨架）

> 三问连答（待填）：① omni 的量化是怎么按名字装配的（factory 模式）？② FP8 / INT8 / MXFP4 / MXFP8 / GGUF / INC 各走哪个 config？③ diffusion 侧的量化和 LLM 侧是不是两套？
>
> 与昇腾原生低精度对照见 [昇腾量化特性支持速查](../vllm-ascend/snippets/ascend-quantization.md)、[昇腾代次与原生低精度格式](../vllm-ascend/snippets/ascend-generations-low-precision.md)。源码 `~/git/vllm_omni/vllm-omni/vllm_omni/quantization/`，行号随版本漂移。

## 一句话定位
待填：量化层负责把权重/激活按目标格式打包并挂上对应 kernel，component_config 决定「哪一层用哪种量化」。

## 入口断点
| 行为 | 入口 `file:line`（待补行号） |
|---|---|
| 量化方案装配入口 | `quantization/factory.py` |
| 每层量化配置 | `quantization/component_config.py` |
| INT8 / MXFP4 / MXFP8 | `quantization/{int8_config,mxfp4_config,mxfp8_config}.py` |
| FP8 kernel | `quantization/quack_fp8.py` |
| GGUF / Intel INC | `quantization/{gguf_config,inc_config}.py` |
| diffusion 侧量化 | `diffusion/quantization/`（独立一套，待确认差异） |

## 数据流一张图（待画）
待填：`load_model → factory 选 config → 按 component_config 逐层替换 Linear/kernel → forward 走量化路径`。

## 与上游 vLLM 的 diff
待填：哪些 config 直接透传 vllm，哪些是 omni 为 diffusion / NPU 新增。

## 一个可跑的最小例子
- [ ] 用一个 FP8 或 W4A16 recipe 起服务，在 factory 装配处断点，打印被替换的层。

## Open questions
- [ ] ModelOpt 混合 FP8/NVFP4 的「混合」粒度在哪配置？
- [ ] NPU MXFP4 online/offline 两条路径分叉点？
