---
tags:
  - vllm-omni
  - diffusion
  - DiT
  - CacheDiT
  - VAE
  - CFG parallel
  - 骨架待填
---

# Diffusion pipeline 内部：从 diffusion_engine 到 VAE（骨架）

> 三问连答（待填）：① diffusion 请求进来后 `diffusion_engine` 怎么组织去噪循环？② CacheDiT / MagCache 这类加速缓存挂在哪一步？③ CFG parallel、VAE parallel、PP/USP 这些并行怎么切？
>
> 你已有 [Diffusion 注意力后端全貌](diffusion-attention-backend.md) 讲的是 attention **后端**；本篇补 pipeline **主体**。生成基础见 [DiT 是什么](../generative-basics/dit.md)。源码 `~/git/vllm_omni/vllm-omni/vllm_omni/diffusion/`，行号随版本漂移。

## 一句话定位
待填：diffusion_engine 是 diffusion stage 的「引擎」,等价于 LLM 侧的 EngineCore,驱动多步去噪 + VAE 解码。

## 入口断点
| 行为 | 入口 `file:line`（待补行号） |
|---|---|
| diffusion 引擎主体 | `diffusion/diffusion_engine.py` |
| stage 进程/客户端 | `diffusion/stage_diffusion_proc.py` / `stage_diffusion_client.py` |
| 去噪步/缓存加速 | `diffusion/cache/`（CacheDiT / MagCache） |
| 并行(PP/SP/CFG) | `diffusion/distributed/group_coordinator.py:70` `GroupCoordinator` 及子类 |
| 编译 | `diffusion/compile.py` |
| 输出格式化 | `diffusion/output_formatter.py` / `postprocess/` |
| LoRA(step-wise) | `diffusion/lora/` |

## 数据流一张图（待画）
待填：`prompt → text encoder → 初始噪声 → [DiT 去噪 × N 步(可 CacheDiT 跳步)] → VAE decode → 图/视频`。

## 与上游 vLLM 的 diff
待填：diffusion 这套基本是 omni 独立实现（vLLM 主干无 diffusion），重点看它怎么复用 vLLM 的调度/内存/CUDA Graph 抽象。

## 一个可跑的最小例子
- [ ] 跑一个 Qwen-Image / HunyuanImage3 recipe，在去噪循环断点，数 N 步、看 CacheDiT 命中。

## Open questions
- [ ] CFG parallel 的 companion 请求和 `Orchestrator` 的 `CfgCompanionTracker` 怎么配合？
- [ ] VAE parallel 的切分维度（patch/序列）在哪定义？
