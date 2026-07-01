---
tags:
  - vllm-omni
  - TTS
  - 语音
  - Code2Wav
  - 高并发
  - entrypoints
  - 骨架待填
---

# 语音/TTS 服务链路：从 OpenAI 接口到 Code2Wav（骨架）

> 三问连答（待填）：① 一个语音生成请求从 `/v1/audio` 类接口进来后走哪条 stage 链？② Code2Wav 的 cross-request batching 怎么在高并发下复用？③ ref-context cache / 自定义音色是怎么缓存的？
>
> 编排层见 [Orchestrator](engine-orchestrator.md)，模态路径对比见 [全模态与纯文本用例路径区别](multimodal-vs-text-path.md)。源码 `~/git/vllm_omni/vllm-omni/vllm_omni/entrypoints/` 与 `model_executor/`，行号随版本漂移。

## 一句话定位
待填：TTS 链路通常是 thinker(文本/语义) → talker(声学 token) → Code2Wav(波形) 的多 stage 流水，末端产音频流。

## 入口断点
| 行为 | 入口 `file:line`（待补行号） |
|---|---|
| OpenAI 兼容 serving | `entrypoints/openai/` |
| async omni 入口 | `entrypoints/async_omni.py` / `omni.py` |
| 客户端请求状态 | `entrypoints/client_request_state.py` |
| Code2Wav / 声码器 | 待定位（grep `Code2Wav`） |
| 音色 / ref-context cache | 待定位（grep `custom voice` / `ref_context`） |

## 数据流一张图（待画）
待填：`文本 → thinker stage → talker stage(声学 token) → Code2Wav stage(batching) → 音频流(streaming)`。

## 与上游 vLLM 的 diff
待填：thinker/talker 各自是原生 vLLM 引擎（透传），omni 新增的是声学 token → 波形这段 + 音频流式输出协议。

## 一个可跑的最小例子
- [ ] 跑 Qwen3-TTS / Qwen3-Omni recipe，并发多请求，在 Code2Wav batching 处断点看是否跨请求合批。

## Open questions
- [ ] 音频 SLO metrics（`metrics/`）采集在链路哪一环？
- [ ] streaming 音频的 finish reason 与文本流式如何统一？
