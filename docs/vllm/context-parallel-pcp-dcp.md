---
tags:
  - vllm
  - vllm-ascend
  - vllm-omni
  - 并行
  - Context Parallel
  - PCP
  - DCP
  - 长序列
  - KV Cache
---

# 上下文并行 CP：PCP 与 DCP（长序列 prefill / 长 KV decode）

> 三问连答：① CP（Context Parallel）和 EP/DP/TP/SP 有什么本质不同？② PCP 和 DCP 各切什么、为什么要分成两个？③ `use_cp` / `pcp_size` / `dcp_size` 在代码里怎么定义、怎么约束？
>
> 这是并行家族的「第 5 根轴」，与 [EP/DP/TP/SP 区别：从 FusedMoE 讲起](ep-dp-tp-sp-fused-moe.md) 互补——那篇切**权重**，本篇切**序列/KV**。两阶段背景见 [两阶段与 Roofline](../llm-basics/prefill-decode-roofline.md)；CP 的通信 kernel（环形注意力）见 [Diffusion 注意力后端全貌](../vllm-omni/diffusion-attention-backend.md)。源码 `~/git/vllm_omni/{vllm,vllm-ascend,vllm-omni}`，行号随版本漂移。

## 一句话定位

CP 沿 **sequence / context 维度**把一条序列切到多张卡上。EP/TP 切的是**权重矩阵**(模型放不下 / 算不动),CP 切的是**一条样本的 token 轴**(**单条序列太长** → 激活、KV、attention 放不下 / 算不动)。所以 CP 是**长上下文**专用的并行轴。

```python
# vllm-ascend/vllm_ascend/worker/model_runner_v1.py:589
@property
def use_cp(self) -> bool:
    return self.pcp_size * self.dcp_size > 1
```

只要 PCP 或 DCP 任一 > 1,就进入 CP 路径。

## PCP vs DCP:为什么是两个,而不是一个

长序列的痛点在 **prefill** 和 **decode** 两阶段**表现不同**(见 [prefill/decode roofline](../llm-basics/prefill-decode-roofline.md)),所以拆成两根子轴:

| | PCP (Prefill Context Parallel) | DCP (Decode Context Parallel) |
|---|---|---|
| 切什么 | prefill 时的 **输入序列(context)** | decode 时的 **KV cache / context** |
| 解决 | 长 prompt 的 prefill 算力/激活瓶颈(compute-bound) | 长历史的 KV 显存/带宽瓶颈(memory-bound) |
| **是否增大 world size** | **是**,独立并行组,额外占卡 | **否**,复用 TP group 的卡,把一个 TP 组切成 `tp//dcp` 个 DCP 组 |
| 关键约束 | 与其它并行维乘进 world size | **`tp_size % dcp_size == 0`** |
| 通信形态 | 环形/allgather 跨 pcp(ring attention) | 跨 dcp rank 收集 KV(a2a / allgather) |

> DCP 的巧思:**它不额外要卡**。原来 TP=8 每卡存一份完整 KV;开 DCP=2 后,把这 8 卡看成 4 个 DCP 组,每组 2 卡**各存一半 KV** → 单卡 KV 显存减半、attention 时跨 2 卡收集。因此 `tp_size` 必须能被 `dcp_size` 整除。

## 代码入口（按 breakpoint 范式）

**配置层(vLLM 定义源头)**
| 项 | `file:line` |
|---|---|
| `prefill_context_parallel_size: int = 1` | `vllm/vllm/config/parallel.py:124` |
| `decode_context_parallel_size: int = 1` | `vllm/vllm/config/parallel.py:339` |
| DCP 约束 `tp % dcp == 0` | `vllm/vllm/config/parallel.py:503` |
| world size 乘上 pcp | `vllm/vllm/config/parallel.py:796` |
| CP 工具函数 | `vllm/vllm/v1/worker/cp_utils.py` |

**执行层(读取并生效的地方)**
| 项 | `file:line` |
|---|---|
| scheduler 读 pcp/dcp world size | `vllm/vllm/v1/core/sched/scheduler.py:168` |
| KV cache 尺寸按 dcp/pcp 分 | `vllm/vllm/v1/kv_cache_interface.py:238` |
| attention 后端判 dcp>1 | `vllm/vllm/v1/attention/backends/flash_attn.py:692`、`flashinfer.py:1421` |
| `use_cp` 属性(ascend) | `vllm-ascend/.../model_runner_v1.py:589` |

**omni / NPU 侧**
| 项 | `file:line` |
|---|---|
| pcp_manager / 跨 pcp token 数 | `vllm-omni/.../platforms/npu/worker/npu_generation_model_runner.py:99,169,343` |
| AR runner CP 分支 | `vllm-omni/.../platforms/npu/worker/npu_ar_model_runner.py:389,480` |

## 与上游 vLLM 的 diff（继承/透传/改写）

- **PCP/DCP 概念、config 字段、约束** 全部来自 **vLLM 主干**(`config/parallel.py`),ascend/omni **继承**。
- **ascend** 加了 `use_cp` 便捷属性和 NPU 通信后端(`dcp_comm_backend`)对接。
- **omni** 在 NPU runner 里引入 `pcp_manager`、`max_num_tokens_across_pcp` 等**跨 pcp 对齐逻辑**,把 CP 接到多模态/多 stage 的 runner 上(改写点在 runner 层,概念透传)。

## 一个可跑的最小例子(待填)

- [ ] 起服务时加 `--decode-context-parallel-size 2`(需 `tp_size` 可被 2 整除),观察单卡 KV 显存是否下降。
- [ ] 在 `scheduler.py:168` 和 `kv_cache_interface.py:238` 断点,打印 `dcp_world_size` 生效路径。
- [ ] 长 prompt 场景加 PCP,在 `npu_generation_model_runner.py:343` 看 `max_num_tokens_across_pcp` 的 allgather。

## Open questions

- [ ] `dcp_comm_backend='a2a'` vs allgather 的选择依据?(`parallel.py:509`)
- [ ] `dcp_kv_cache_interleave_size` 控制 KV 在 dcp rank 间的交错粒度,对 prefix cache 命中有何影响?
- [ ] PCP 与 SP(Sequence Parallel)都切序列,边界在哪?(SP 切的是 layernorm/激活的序列维,CP 切的是 attention 的 KV 维——待展开对照)
