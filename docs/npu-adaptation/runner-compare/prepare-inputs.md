---
tags:
  - vllm-omni
  - vllm-ascend
  - vllm
  - model_runner
  - _prepare_inputs
  - MRO
---

# L2 下钻:`_prepare_inputs` —— 菱形 MRO 的对齐风险点

> 覆盖矩阵:`🔧 vllm GPU · ⬆️ omni GPU · 🔧 ascend NPU · ⬆️ omni NPU`。
> 上游和 vllm-ascend **都**重写了它,而 `OmniGPUModelRunner` / `OmniNPUModelRunner` 自己没写 → 走 MRO,实际用 **vllm-ascend 的 NPU 版本**。
> 源码位置(基线见 [index](index.md#regen) 头部 SHA):
> - vllm `vllm/v1/worker/gpu_model_runner.py:1889`
> - vllm-ascend `vllm_ascend/worker/model_runner_v1.py:773`

## 关键问题:返回签名都不一样

第一眼就有硬差异——**返回值的元数个数不同**:

=== "vllm GPU(2 元组)"

    ```python
    # vllm/v1/worker/gpu_model_runner.py:1889
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
    ```

=== "vllm-ascend NPU(4 元组)"

    ```python
    # vllm_ascend/worker/model_runner_v1.py:773
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None,
               int,                        # + total_num_scheduled_tokens(PCP 本地切分后可能 ≠ 全局)
               list[np.ndarray]]:          # + num_scheduled_tokens_compressed_list(压缩 KV 索引)
    ```

`OmniNPUModelRunner` 经 MRO(`OmniNPU → OmniGPU → NPUModelRunner(ascend) → GPUModelRunner`)拿到 **ascend 的 4 元组版本**。✅ 这是对的:omni 的调用方必须按 4 元组解包。

!!! danger "对齐时必须核对这两点"
    1. **返回 arity**:一旦 `OmniGPUModelRunner` 将来**也** override `_prepare_inputs`(矩阵格子从 ⬆️ 变 🔧),MRO 会让 omni GPU 版**抢在 ascend 之前**——它大概率返回上游的 **2 元组**,omni NPU 侧按 4 元组解包会直接 `ValueError: not enough values to unpack`,或更糟静默丢掉后两个 NPU 专属字段。**每次生成矩阵后,盯死这一行的 omni GPU 列是否从 ⬆️ 变 🔧。**
    2. **PCP / 压缩 KV 字段**:后两个返回值(`total_num_scheduled_tokens`、`compressed_list`)是 NPU 专属,GPU 基准根本没有。跟 ascend 对齐时若漏掉,PCP 多卡切分或 DSV4 压缩 KV 会错位。

## vllm-ascend 相对上游改了什么(NPU delta)

以下是 omni NPU 实际继承、且对齐时要跟的关键增量。均为基线 SHA 下的真实代码。

=== "① FIA 注意力状态"

    ```python
    # :796 —— 区分 prefill-only / decode-only / 混合,FIA 聚合需要
    attn_state = self._build_attn_state(
        num_reqs, num_scheduled_tokens, num_valid_tokens)
    with_prefill = attn_state not in [
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    ]
    self.with_prefill = with_prefill
    ```

    `num_valid_tokens` 要**剔除 spec-decode draft**再算,GPU 基准无此枚举。

=== "② PCP 本地切分前的 slot mapping"

    ```python
    # :829 —— PCP 把请求切成多个 micro-batch,slot mapping 必须在切分「前」算好
    if self.pcp_size > 1:
        pre_pcp_positions = torch.from_numpy(
            positions_np[:total_num_scheduled_tokens]).to(self.device)
        pre_pcp_qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        pre_pcp_qsl[1:num_reqs + 1] = torch.from_numpy(cu_num_tokens).to(
            dtype=torch.int32, device=self.device)
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs, pre_pcp_qsl, pre_pcp_positions)
    ```

=== "③ torch_npu.Event 异步同步"

    ```python
    # :1229 —— async spec-decode:GPU 上修正的 seq_lens 必须在 _build_attention_metadata 读它「之前」回写 CPU
    if (self._needs_seq_lens_cpu_sync and self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None and prev_req_id_to_index):
        self.optimistic_seq_lens_cpu[:num_reqs].copy_(
            self.seq_lens[:num_reqs], non_blocking=True)
        if self._seq_lens_cpu_event is None:
            self._seq_lens_cpu_event = torch.npu.Event()   # NPU 的 GPU↔CPU 同步原语
        self._seq_lens_cpu_event.record()
        self._seq_lens_cpu_event_pending = True
    ```

=== "④ GDN / CP 的 padding 约定"

    ```python
    # :962 —— GDN 算子要一份「未 padding」的 query_start_loc
    if self._has_gdn:
        self.gdn_query_start_loc.np[0] = 0
        self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])

    # :986 —— CP 的 reshape_and_cache 要求 padding 填 -1(不是任意大值)
    self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)
    ```

## 一句话对齐结论

NPU 版的 `_prepare_inputs` 比上游多出 **PCP 本地切分、FIA 注意力状态、`torch_npu.Event` 异步同步、压缩 KV 元数据**四类逻辑,并因此**多返回 2 个字段**。omni NPU 目前靠 MRO 正确落到这个版本——**保持 `OmniGPUModelRunner` 不 override 此方法,是这条对齐链不断的前提**。

## 相对基线的 delta(定期巡检)

```bash
git -C ~/git/vllm_omni/vllm-ascend log --oneline -10 \
  -L :_prepare_inputs:vllm_ascend/worker/model_runner_v1.py
```

> 输出摘要回填 [drift-log.md](drift-log.md),并在 [#4610](https://github.com/vllm-project/vllm-omni/issues/4610) 勾选/新增子项。
