---
tags:
  - vllm-omni
  - vllm-ascend
  - vllm
  - model_runner
  - _dummy_run
  - 图捕获
  - cudagraph
  - aclgraph
---

# L2 下钻:`_dummy_run` / 图捕获 —— 四方全 override 的雷区

> 覆盖矩阵:`🔧 vllm GPU · 🔧 omni GPU · 🔧 ascend NPU · 🔧 omni NPU`(唯一四方全部直接 override 的核心方法,且**无一调用 `super()`**,四份都是完整重写)。
> 相关既有笔记:[runner 图捕获实现差异(NPU vs GPU)](../npu-gpu-graph-in-runner.md) · [嵌套图捕获 #4519](../nested-graph-capture.md) · [is_tracing 在 NPU 失灵](../transformers-is-tracing-npu.md) · [talker_mtp 图安全](../talker-mtp-graph-safety.md)
>
> 源码位置(基线见 [index](index.md#regen) 头部 SHA):
> vllm `gpu_model_runner.py:5670` · omni GPU `gpu_model_runner.py:854` · vllm-ascend `model_runner_v1.py:3321` · omni NPU `npu_model_runner.py:51`

## 一、四方为什么各自重写,且都不 super()

`_dummy_run` 是图捕获暖机入口。四层各有非改不可的理由,且**都选择完整 fork 而非 `super()` 组合**——这本身就是维护负担:上游一改,四份都要手动跟。

| 层 | 图后端 | 重写动机 |
|---|---|---|
| vllm GPU | cudagraph | 基准路径 |
| omni GPU | cudagraph | 插 4 个多模态钩子(attn 扩展 / inputs_embeds 路径 / talker_mtp / extract_multimodal_outputs) |
| vllm-ascend | **aclgraph** | 换 NPU 图后端,`set_ascend_forward_context`,`_model_forward` 封装 |
| omni NPU | **aclgraph** | 菱形合流:既要 ascend 的 aclgraph,又要 omni 的多阶段——**但没走 `super()`,是手抄+改** |

## 二、四方实现并列(均为基线 SHA 下真实代码)

=== "vllm GPU(基准)"

    ```python
    # gpu_model_runner.py:5670  返回 tuple[Tensor, Tensor]
    with self.synchronize_input_prep():
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)
            # ...
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                # ...
            )
    # 纯 cudagraph,无任何 omni / NPU 扩展
    ```

=== "omni GPU(+多模态钩子)"

    ```python
    # gpu_model_runner.py:854  完整重写,不调 super()。在 4 处插钩:
    self._maybe_attach_attention_metadata_extensions(   # 钩子① routed-expert 等元数据
        attn_metadata=attn_metadata, ..., for_cudagraph_capture=is_graph_capturing)
    elif getattr(getattr(self, "model", None), "has_preprocess", False):  # 钩子② inputs_embeds 捕获路径
        input_ids = self.input_ids.gpu[:num_tokens_padded]
        inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
    if getattr(self.model, "talker", None) is not None and self.has_talker_mtp:  # 钩子③ talker_mtp
        outputs = self.talker_mtp(...)
    hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)  # 钩子④
    ```

=== "vllm-ascend NPU(aclgraph)"

    ```python
    # model_runner_v1.py:3321  返回 tuple[Tensor, Tensor]
    if self.use_compress:                 # NPU compress:捕获期位置填 127 标记压缩推理
        self.positions.fill_(127)
        self._dsa_positions_cpu_buf.fill_(127)
    # ...
    with set_ascend_forward_context(
        attn_metadata, self.vllm_config,
        num_tokens=num_tokens_padded, aclgraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_desc, model_instance=self.model,
        has_sinks=self._has_sinks,                        # ← NPU 专属
        input_ids=input_ids,                              # ← NPU 专属
        eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,  # ← NPU 专属
    ):
        outputs = self._model_forward(                    # ← 走 NPU 图封装(full-graph params + SP all-gather)
            num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds)
    ```

=== "omni NPU(菱形合流)⚠️"

    ```python
    # npu_model_runner.py:51  完整重写,不调 super() 也不显式调父类
    with set_ascend_forward_context(
        attn_metadata, self.vllm_config,
        num_tokens=num_tokens_padded, aclgraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_desc, model_instance=self.model,
        # ⚠️ 缺 has_sinks / input_ids / eplb_heat_collection_status(ascend 有)
    ):
        if getattr(self.model, "talker", None) is not None and self.has_talker_mtp:  # omni 多阶段
            outputs = self.talker_mtp(...)
        # Call self.model() directly (like GPU) to avoid make_omni_output during dummy_run
        outputs = self.model(                             # ⚠️ 直调 model(),绕过 self._model_forward()
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)
    ```

## 三、对齐核对:capture 路径 ≠ execution 路径

omni NPU 与 ascend 有两处**已验证**的真实差异,均集中在 `_dummy_run`:

1. **`set_ascend_forward_context` 少三个 NPU 参数**(`has_sinks` / `input_ids` / `eplb_heat_collection_status`)。ascend 传、omni 不传。
2. **直调 `self.model()` 而非 `self._model_forward()`**。这是**有意为之**——omni 自己的注释写明「避免 dummy_run 期间触发 `make_omni_output`」。

!!! warning "值得核实(不是已确认的 bug)"
    omni NPU **有**自己的 `_model_forward`(`npu_model_runner.py:337`),里面做两件事:`make_omni_output` 包装 + `_all_gather_hidden_states_and_aux`(**SP all-gather**)。
    `_dummy_run` 为绕开前者而直调 `self.model()`,**副作用是同时跳过了后者(SP all-gather)**。

    → 由此产生一个真实的对齐问题:**捕获出的图走的 forward 路径,与真实执行(经 `_model_forward`)是否一致?** 若 SP all-gather 会改变张量形状 / 图拓扑,那么「捕获时没有它、回放时有它」就可能对不上。这值得针对性验证,但目前证据只到「路径不同」,**不足以断定必然出错**——ascend 的 `_model_forward` 里那些逻辑在纯 dummy 场景下可能本就是 no-op。

    **建议动作**:在 SP size > 1 的 NPU 配置下,确认 aclgraph 捕获是否覆盖了 all-gather;若否,评估是否应改为「直调 model() 但补一次 all-gather」或「让 dummy 也走 _model_forward 并在其内部跳过 make_omni_output」。回填 [drift-log](drift-log.md) 与 [#4610](https://github.com/vllm-project/vllm-omni/issues/4610)。

!!! note "MRO 视角:为什么不能简单 super()"
    有人会问「omni NPU 直接 `NPUModelRunner._dummy_run(self, ...)` 复用 ascend 版不就行了?」——不行,因为 ascend 版会调 `self._model_forward()`,而 `self` 是 `OmniNPUModelRunner`,`_model_forward` 已被 omni override 成带 `make_omni_output` 的版本,dummy_run 里正是要避开它。这就是为什么 omni 选择手抄 `_dummy_run`。**代价**是 ascend 侧 `set_ascend_forward_context` 参数一旦新增(如上面三个),omni 手抄版不会自动跟上——这正是矩阵要盯的漂移点。

## 四、定期巡检

```bash
# ascend 的 _dummy_run 自基线以来改了什么(尤其 set_ascend_forward_context 新增参数)
git -C ~/git/vllm_omni/vllm-ascend log --oneline -10 \
  -L :_dummy_run:vllm_ascend/worker/model_runner_v1.py
```

- [ ] ascend `set_ascend_forward_context` 是否新增了 omni 手抄版没有的参数?
- [ ] `_check_and_update_cudagraph_mode` 是否仍 cap PIECEWISE(#4674 未回归)?
- [ ] SP>1 下,dummy_run 直调 `self.model()` 是否漏捕获 all-gather?
