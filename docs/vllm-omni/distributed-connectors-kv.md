---
tags:
  - vllm-omni
  - distributed
  - 连接器
  - KV 传输
  - PD 分离
  - 骨架待填
---

# 跨 stage 数据面：连接器与 KV 传输（骨架）

> 三问连答（待填）：① stage 之间的中间结果 / KV 到底怎么从一个子进程传到另一个？② `omni_connectors` 提供了哪几类连接器（NPU Yuanrong TransferEngine 等）？③ PD 分离时 prefill 产的 KV 怎么喂给 decode stage？
>
> 配套编排层见 [Orchestrator：多 stage 编排核心](engine-orchestrator.md)。源码 `~/git/vllm_omni/vllm-omni/vllm_omni/distributed/`，行号随版本漂移。

## 一句话定位
待填：编排层(`Orchestrator`)决定「搬到哪」,这一层负责「怎么搬」——把 tensor/KV/chunk 在 stage 子进程之间实际传输。

## 入口断点
| 行为 | 入口 `file:line`（待补行号） |
|---|---|
| 连接器实现集合 | `distributed/omni_connectors/` |
| 分布式成员/心跳 | `distributed/omni_coordinator/omni_coordinator.py:19` `OmniCoordinator` |
| Ray 工具 | `distributed/ray_utils/` |
| NPU Yuanrong TransferEngine | 待定位（grep `Yuanrong` / `TransferEngine`） |

## 数据流一张图（待画 Mermaid）
待填：`stage_i 输出 → 连接器 send → 网络/共享内存 → 连接器 recv → stage_{i+1} 输入`。

## 与上游 vLLM 的 diff
待填：vLLM 的 KV connector（如 LMCache/NIXL）在 omni 里如何被复用/扩展到「跨 stage」而不仅是「跨 PD 实例」。

## 一个可跑的最小例子
- [ ] 找一个开了 PD 分离或多 replica 的 recipe，在连接器 send/recv 处下断点。

## Open questions
- [ ] chunk transfer / memory pool 的生命周期由谁管？
- [ ] 多 replica 的 GPU/NPU device mapping 在哪决定？
