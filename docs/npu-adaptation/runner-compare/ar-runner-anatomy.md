---
tags:
  - vllm-omni
  - vllm-ascend
  - model_runner
  - AR
  - async_scheduling
  - prefix_cache
  - 多阶段流水线
---

# L2 下钻：`npu_ar_model_runner` 全函数解剖（含理论底座）

> 对象：`vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`（对齐 PR 后约 1800 行），GPU 基准是 `vllm_omni/worker/gpu_ar_model_runner.py`。
> 类：`NPUARModelRunner(OmniNPUModelRunner, OmniConnectorModelRunnerMixin)` —— omni 多阶段流水线里 **AR（自回归）stage 的 NPU 执行器**，Thinker / Talker 都跑在它上面。
> 读法建议：先看 §0 的簇地图建立骨架，再按簇下钻；每簇后面的"理论卡"是这簇代码背后的通用推理系统知识，面试可直接复用。

## §0 一张地图：35 个函数分 8 簇

这个文件看着吓人，其实只回答一个问题：**"一个 AR stage 每步 forward 之后，除了 token，还要把 hidden states / 多模态 payload 正确地交给谁"**。所有函数都围绕这条主线：

| 簇 | 函数 | 一句话职责 |
|---|---|---|
| A 构造与缓冲 | `__init__` · `_make_buffer` | 建 buffer、KV 传输管理器、connector |
| B 自定义采样器 | `_build_model_sampler_output_token_ids` · `_sampling_metadata_for_model_sampler` · `_sample` | 让 CosyVoice3 这类自带 sampler 的模型拿到解码历史 |
| C 下游路由 | `_request_final_stage_id` · `_request_needs_downstream_stage_payload` · `_resolve_pooler_payload_req_ids` · `_sparse_mm_req_ids` · `_resolve_sparse_mm_routing` · `_is_sparse_audio_marker` | 决定哪些请求需要给下游 stage 发 payload |
| D 图捕获 | `capture_model` · `_capture_talker_mtp_graphs` | 在基类图捕获之外，额外捕获 talker MTP 子图 |
| E 前缀缓存 | `_model_needs_full_prefix_hidden_states` · `_maybe_update_prefix_cache` · `_maybe_get_combined_prefix_cache_tensors` | omni 张量前缀缓存的写入与命中合并 |
| F payload 构建 | `_build_combined_prefix_cache_mm_payload` · `_build_omni_mm_payload` · `_build_omni_pooler_payload` · `_resolve_req_hidden_states` · `_build_multimodal_outputs` | 把 hidden/mm 张量切成 per-request 的下游 payload |
| G 两阶段主循环 | `execute_model` · `sample_tokens` · `_run_post_sample_side_effects` | forward（攒状态）→ 采样（出结果） |
| H 异步输出 | `_should_use_async_omni_output` · 3 个 `_snapshot_*` · `_build_omni_async_snapshot_payload` · `_maybe_run_eager_omni_postprocess_before_async_output` · `_get_or_create_omni_payload_copy_stream` · `_build_omni_model_runner_output_from_snapshot` | 把昂贵的输出构建挪到后台线程 |
| 杂项 | `_should_return_omni_routed_experts` · `_model_omni_flag` · `_runner_model_omni_flag` · `_model_omni_pooler_payload_include_hidden` · `_resolve_global_request_id` | 小开关与 ID 解析 |

模块级还有 4 个定义：`_OmniOutputTensorSnapshot`（:68）、`OmniAsyncNPUModelRunnerOutput`（:74）、`_ensure_tensor_values`（:80）、`ExecuteModelState`（:114），归属 G/H 簇，随簇讲。

!!! info "为什么先分簇再读"
    runner 类的阅读陷阱是按行读——execute_model 500 行里 omni 逻辑和 ascend 基类逻辑交错（`Omni-new` 注释块标的就是前者）。按簇读等于按**数据流**读：B/C/E/F 全是 G 的"配件"，H 是 G 的"性能外挂"。

---

## §A 构造与缓冲

### `__init__`（:135）

在基类（ascend NPU runner → omni 公共层）初始化之后补 5 件事：

1. `input_ids` / `inputs_embeds` 两块持久 buffer（`max_num_tokens` 大小，图模式要求地址稳定）；
2. `hidden_size` 从 `hf_text_config` 取——**每个 stage 的 hidden size 不同**（Thinker 2048 vs Talker 1024 之类），不能用全局配置；
3. `OmniKVTransferManager`：跨 stage KV 迁移的管理器；
4. 按 `model_arch` 白名单初始化 **omni connector**（`init_omni_connectors`，来自 mixin）——只有多阶段模型（Qwen3-Omni、CosyVoice3、TTS 系列等）才建，白名单和 GPU 逐字一致；
5. `_downstream_payload_cache`：C 簇的 per-request 结果缓存。

### `_make_buffer`（:169）

包一层基类的 `_make_buffer`：算出总字节数，超过阈值时用 `maybe_disable_pin_memory_for_ray` 临时关掉 pin memory——**Ray 场景下大块 pinned 内存会被 Ray 的对象存储机制长期钉住**，导致宿主机内存被吃光。

!!! note "理论卡 · pinned memory 是什么，为什么又爱又怕"
    - **pinned（page-locked）内存**：不会被 OS 换页的主机内存。设备↔主机的 DMA 拷贝要求物理地址稳定，所以只有 pinned 内存才能走真正的异步拷贝（`copy_(non_blocking=True)`）；pageable 内存会退化成**同步**拷贝（驱动先拷到内部 pinned 暂存区）。
    - **代价**：pinned 内存减少 OS 可调度的物理页。分配过大或泄漏会拖垮整机。
    - 本文件有两处围绕它做文章：这里的 Ray 规避，以及 H 簇快照用 `is_pin_memory_available()` 决定快照目的地是否 pin（GPU 侧曾因为拿错这个开关，异步 D2H 变同步，单步慢 240ms——见 GPU 文件注释）。

---

## §B 自定义采样器支持

### `_build_model_sampler_output_token_ids`（:184）

给 `prefer_model_sampler` 模型重建**每个请求的已解码 token 历史**。vLLM 只在有 penalty/logits processor 时才填 `sampling_metadata.output_token_ids`，而 CosyVoice3 的 RAS（Repetition-Aware Sampling）采样器需要这份历史。难点在 async scheduling：历史尾部是 `-1` 占位符（上一步采样结果还在 D2H 路上），本函数用 `prev_req_id_to_index` 找到上一步 `sampled_token_ids_cpu` 里的行，惰性 `async_copy_ready_event.synchronize()`（整个 batch 最多等一次）后回填占位符；填不上的截断到第一个 `-1`。

### `_sampling_metadata_for_model_sampler`（:229）

外层开关：模型声明 `skips_model_sampler_output_token_history` 就跳过；重建结果与现有相同就不 `replace`（省一次 dataclass 拷贝）。

### `_sample`（:1058，与 GPU 逐字一致）

覆盖基类采样入口：无 spec decode 且模型 `prefer_model_sampler` 时，先用 runner 的 `logit_bias_state` 把 min_tokens / allowed_token_ids 的偏置打到 logits 上（标准 sampler 内部会做，绕过它就得手动补），再调 `model.sample(logits, metadata)`；模型返回 None 或不满足条件则回落标准 sampler。spec decode 路径直接走基类。

!!! note "理论卡 · async scheduling 的『最终一致』语义"
    v1 引擎开 async scheduling 后，调度器**不等**上一步采样落地就构建下一步 batch，CPU 侧状态（token 历史）从"同步可见"变成"最终一致 + 占位符"。所有依赖历史的组件都要各自处理回填：penalty、RAS、grammar、bookkeeping。**这类"异步化漏出来的语义窟窿"是 v1 bug 密度最高的地方**——本文件曾经的 bug 级缺口（`_sample` 调用了 NPU 侧不存在的 `_sampling_metadata_for_model_sampler`）正是在这条链上。

---

## §C 下游 payload 路由

多阶段流水线里，不是每个请求都需要把 hidden/mm 发给下游（有的请求在本 stage 就是终点）。这簇 6 个函数回答"发给谁"：

| 函数 | 行 | 作用 |
|---|---|---|
| `_request_final_stage_id` | :237 | 从 `model_intermediate_buffer` 或 req_state 的 `additional_information_cpu` 里读 `omni_final_stage_id`（模型在 prefill 时写入的"我到第几 stage 结束"标记） |
| `_request_needs_downstream_stage_payload` | :250 | `final_stage_id is None or > 0` → 需要发。**缺标记时保守地发**（宁可多传不可饿死下游）。带 `_downstream_payload_cache` 缓存 |
| `_resolve_pooler_payload_req_ids` | :260 | 过滤出需要 payload 的请求；特例：`engine_output_type == "audio"` 且没人需要时**全发**——单 stage TTS 模型（VoxCPM2）本 stage 就是终点但仍要 mm payload 做音频后处理 |
| `_sparse_mm_req_ids` | :270 | 从 mm 输出的 `meta.req_id` + `meta.sparse_audio` 标记里解析"这一步真正产出了音频的请求列表" |
| `_resolve_sparse_mm_routing` | :289 | 上面的封装：audio 输出 + 稀疏标记存在时，把下游列表收窄成"有产出的请求"，并返回 `{rid: 稀疏索引}` 映射 |
| `_is_sparse_audio_marker` | :306 | 宽容地解析 truthy 标记（"1"/"true"/list/bool 都认） |

!!! note "理论卡 · 稀疏输出（sparse audio）为什么存在"
    Talker 这类模型不是每个 decode step 都产出音频 codec 帧——只有攒够一个 chunk 才吐一次。所以 mm 输出的 batch 维度和请求 batch **不对齐**：本步 8 个请求可能只有 2 个有音频。`meta.req_id` 就是这个稀疏 batch 的"行名"，路由函数负责把稀疏行对回请求。漏掉这层映射的典型症状：请求 A 拿到请求 B 的音频。

---

## §D 图捕获

### `capture_model`（:314）

先让基类捕获主模型的 ACL graph，再补捕 talker MTP 子图。返回图占用的字节数（NPU 侧命名 `npugraph_memory_bytes`）。

### `_capture_talker_mtp_graphs`（:319）

Talker 的 MTP（multi-token prediction）头是**主 forward 之外的第二个可捕获子图**。流程：按 `cudagraph_capture_sizes` 从大到小，每个 size 先 warmup N 次（`aclgraph_runtime_mode=NONE`）再正式捕获（`FULL`），捕获间 `torch.npu.synchronize()`。与 GPU 版的差异就是三处平台替换：`ACLGraphWrapper`↔`CUDAGraphWrapper`、`set_ascend_forward_context`↔`set_forward_context`、`torch.npu`↔`torch.accelerator`。

!!! note "理论卡 · 为什么图要按 size 捕获、为什么先 warmup"
    设备图（CUDA graph / ACL graph）录制的是**固定 shape 的算子序列**，回放时省掉每步的 kernel launch 开销（对 decode 这种小 kernel 密集场景收益最大）。所以：
    1. **按 batch size 分档捕获**，运行时把实际 batch pad 到最近的档；
    2. **先 warmup 再捕获**：首次执行会触发算子编译、内存池分配等一次性副作用，录进图里会出错或浪费；
    3. 捕获期间**不能有动态控制流/动态 shape**——这正是本仓一系列图相关 bug 的根源（嵌套捕获 #4519、`is_tracing` 失灵等，见相关笔记）。

---

## §E omni 张量前缀缓存

先分清两个"前缀缓存"：vLLM 的 **KV prefix cache** 缓存注意力 KV（省 prefill 算力）；omni 的 **OmniTensorPrefixCache** 缓存 **hidden states + mm 输出**——因为下游 stage（如 Talker）需要上游完整前缀的 hidden，KV 缓存命中后这些张量不会再被算出来，必须单独缓存。

| 函数 | 行 | 作用 |
|---|---|---|
| `_model_needs_full_prefix_hidden_states` | :372 | 模型 opt-out 钩子：postprocess 只消费尾部的模型（如 qwen3-tts-talker）设 `requires_full_prefix_cached_hidden_states=False`，跳过 hidden 的缓存写入与命中重建，只缓 mm |
| `_maybe_update_prefix_cache` | :377 | forward 后把本步 hidden（若需要）+ 拍扁的 mm 输出按 `slot_mapping` 写进缓存。仅 last PP rank 执行 |
| `_maybe_get_combined_prefix_cache_tensors` | :400 | 输出构建时的反向操作：把"缓存的前缀 + 本步新算的尾部"合并成 per-request 完整张量（`get_merged_hidden_states` / `get_merged_multimodal_states`） |

!!! danger "对齐尾巴：这簇是 NPU 与 GPU 的最大剩余差距"
    GPU 侧已进化成**异步写流水线**（`schedule_async_write` 入队非阻塞 D2H + 记 event，下一步 `drain_ready_async_writes` 收割）+ deferred mm + staged-CPU hidden；共享模块 `vllm_omni/core/prefix_cache.py` 里机制齐全但被 `torch.cuda.is_available()` 门控，NPU 走的还是同步 `update_omni_tensor_prefix_cache`。另有一个**不能盲搬**的点：GPU 用 `slot_mapping.gpu` 再同步回 CPU，修的是 Triton kernel 只写 `.gpu` 侧的 bug；ascend 侧 `.cpu` 是否 stale 需要真机验证。详见对齐分析（B 类清单）。

!!! note "理论卡 · slot mapping"
    `slot_mapping[i]` = 第 i 个 token 在分页缓存里的物理槽位（block_id × block_size + offset）。KV cache 和 omni 张量缓存共用这套寻址——**张量缓存本质上是把 paged attention 的页表复用到了 hidden states 上**。这也是"懂推理底层"的一个好面试切入点：同一套 block table 服务两种缓存。

---

## §F per-request payload 构建

这簇是本次对齐 PR 从 200 行内联代码提取出来的（现与 GPU 逐字一致）：

| 函数 | 行 | 作用 |
|---|---|---|
| `_resolve_req_hidden_states` | :1092 | 选 hidden 来源：前缀缓存命中 → merged 张量（缺 rid 直接 raise）；否则整批 CPU hidden 切片；两者皆无返回 None（调用方省略 "hidden" 键） |
| `_build_combined_prefix_cache_mm_payload` | :424 | 前缀缓存命中路径的 mm payload：递归 `_unwrap_lists` 把 list 包裹的叶子解成 per-request 张量 |
| `_build_omni_mm_payload` | :442 | mm payload 总入口：优先 combined（缓存命中）；否则从 `mm_cpu` 逐 key 取——稀疏音频走 `sparse_mm_index` 取对应行并 `.clone()`（切断与整批张量的存储共享），常规 key 走 `to_payload_element` 按 token span 切片 |
| `_build_omni_pooler_payload` | :496 | 单请求完整 payload = hidden（非稀疏时）+ mm payload |
| `_build_multimodal_outputs` | :1110 | 把 per-request payload 列表转成**线上格式**：`_ensure_tensor_values`（:80）强制 tensor-only（标量/list 包成 tensor，包不了的丢弃并告警）——msgspec 序列化的硬性要求 |

!!! note "理论卡 · 为什么线上格式必须 tensor-only"
    跨进程（engine core → API server / 下游 stage）传输走 msgspec + 自定义 tensor 编码。允许任意 Python 对象意味着 pickle——慢且有安全问题。所以边界上强制 `dict[str, Tensor]`，宁可丢字段也不传杂物。`payload.clone()` 那类细节同理：切片不 clone 会把**整块**底层存储序列化出去。

---

## §G 两阶段主循环

### `ExecuteModelState`（:114）

`execute_model` 和 `sample_tokens` 之间的**接力棒**：logits、hidden、spec decode 元数据、attn_metadata、mm 输出等 13 个字段。NamedTuple 只读，用完置 None——两个方法必须严格交替调用（开头有状态断言）。

### `execute_model`（:544，~500 行）

只做 forward，**不采样**。骨架（`Omni-new` 注释块之外的是 ascend 基类同款逻辑）：

1. **前置杂务**：routed experts 收尾、profiling 计时起点、warmup 状态清理；
2. **跨 stage KV 迁移**（Omni）：对 `finished_requests_needing_kv_transfer` 里的请求，向模型要 `get_kv_transfer_metadata`（`num_computed` 用调度器填的 `seq_len`，本次 PR 对齐），交给 `kv_transfer_manager` 搬 KV；
3. **connector 收发**（Omni，本次 PR 接入）：`register_chunk_recv` 注册流式接收 / `recv_full_payload_inputs` 收上游全量输入 / 对已完结请求 `flush_full_payload_outputs`；
4. **scheduler_output 保护性拷贝**：PCP+MM 场景 deepcopy（突变面不明，保守）；async scheduling + spec decode 场景 `dataclasses.replace` 浅拷两个 dict（本次 PR 从 deepcopy 优化而来，突变面就这两个字段——ascend 基类 ngram 路径同款写法）；
5. **输入准备**：`_update_states` → encoder 提前返回 / 空批提前返回（都挂 `attach_omni_connector_output`）→ `_prepare_inputs` → 图模式与 padding 决策 → attention metadata；
6. **forward**：`set_ascend_forward_context` 下调 `_model_forward`；
7. **后处理**：`extract_multimodal_outputs` 从模型输出里拆出 mm、PCP 场景还原 hidden、写前缀缓存（§E）、算 logits（PP 最末 rank）；
8. **存接力棒** `ExecuteModelState`，返回 None。

### `sample_tokens`（:1487，~260 行）

1. 取接力棒；PP 非末 rank 的空路径直接 `with_kv_conn_output_only` 返回（本次 PR 用共享 helper 替换手写三行）；
2. **grammar bitmask**：logits 搬到 CPU 转 float 应用再搬回——上游用 torch.compile 优化了这步，**ascend 不支持**，这是 C 类（做不了）差异的代表；
3. `_sample`（§B）；
4. `_bookkeeping_sync`：ascend 基类的 6 元组版本（比 GPU 少 `num_nans_in_logits`——基类 API 差异，C 类）；
5. **spec decode**：eagle/draft-model 用 device 上的 sampled ids 直接提草稿（不等 bookkeeping）；ngram 等 CPU 方法在 bookkeeping 后提；之后 `finalize_kv_connector`；
6. **输出构建**：快照输入（§H）→ `output_builder`（即 `_build_omni_model_runner_output_from_snapshot`）同步执行或交给异步包装；
7. `_run_post_sample_side_effects` + async scheduling 包装返回。

### `_run_post_sample_side_effects`（:1748，NPU 独有）

GPU 把这些副作用内联在 sample_tokens 里，NPU 提成方法（因为同步/异步两条路都要调）：profiling 时间戳、EPLB `forward_end`、debugger step、`need_accepted_tokens` 时在 `global_stream` 上等 `sampling_done_event` 再 `_update_states_after_model_execute`、async+PP 时从末 rank 广播 sampled ids。

!!! note "理论卡 · 为什么拆成 execute/sample 两阶段"
    v0.12 起的两阶段设计是 async scheduling 的地基：engine 可以在 GPU 还在跑第 N 步 forward 时就调度第 N+1 步（`execute_model` 返回 None 不阻塞），采样和 CPU bookkeeping（`sample_tokens`）延后到真正需要结果时。收益 = 调度与计算流水线化，隐藏 CPU 开销；代价 = 状态从"函数局部"变成"跨调用接力"（`ExecuteModelState`），以及 §B 说的占位符语义。

---

## §H 异步 omni 输出

背景：omni 输出构建很贵（hidden D2H + mm 切片 + 序列化准备），同步做会卡住下一步 decode。这簇把它挪到后台线程，代价是必须先把所有输入**快照**成不随下一步变化的副本。

| 函数 | 行 | 作用 |
|---|---|---|
| `_should_use_async_omni_output` | :1186 | 六重门控：async scheduling 开 + 无前缀缓存 + 无 spec decode + `async_chunk` 开 + 不返 routed experts + 模型声明 `use_async_omni_output`（有 postprocess 的还必须声明可提前 eager 执行）。**全真才异步**——每个条件背后都是一个"后台线程会读到脏状态"的坑 |
| `_build_omni_async_snapshot_payload` | :1212 | 决定快照里装什么：talker 类模型 `include_hidden=False`（只给下游 codec codes），hidden 完全不进快照，**省掉整个 D2H** |
| `_snapshot_omni_output_tensors_for_async_output` | :1228 | 同步路径原样透传；异步路径调共享 helper `snapshot_tensor_payload_to_cpu_async` 在**专用拷贝流**上发起非阻塞 D2H，返回带 event 的快照（hidden 缺席时用 `hidden_states[:0]` 空占位） |
| `_get_or_create_omni_payload_copy_stream` | :1297 | 惰性建 `torch.npu.Stream()`（GPU 版唯一差异：`torch.cuda.Stream()`） |
| `_snapshot_query_start_loc_cpu` | :1138 | 防御性深拷 `query_start_loc`（tensor/ndarray/list 三种形态都处理）——它是 runner 的**持久 buffer**，下一步会被覆写 |
| `_snapshot_scheduler_output_for_async_omni_output` | :1151 | 浅拷 scheduler_output 里会被改的两个 dict |
| `_maybe_run_eager_omni_postprocess_before_async_output` | :1262 | 有 postprocess 的模型必须在快照**前**、在**live device 张量**上跑完 postprocess（后台线程里跑会读到已推进的 runner 状态），返回 True 让 builder 跳过重复执行 |
| `_build_omni_model_runner_output_from_snapshot` | :1304 | 输出构建总装线（本次 PR 重写成 GPU 结构）：路由（§C）→ hidden D2H（整批或 per-request 切片）→ 前缀缓存合并（§E）→ postprocess → per-request payload 循环（§F）→ `async_chunk` 时 `partition_payload_list` 拆 inter/client 两路 → connector 全量累积 → 组 `OmniModelRunnerOutput`（挂 routed_experts、`omni_connector_output`） |
| `_OmniOutputTensorSnapshot` | :68 | 快照三元组（GPU 版多一个 `staged_hidden_states_cpu` 字段——绑定未移植的 staged-CPU 流水线，所以**故意不共享**） |
| `OmniAsyncNPUModelRunnerOutput` | :74 | 共享 `OmniAsyncModelRunnerOutput` 的空壳平台子类：持有 `output_builder` 闭包，engine 真正要结果时才在后台构建 |

!!! note "理论卡 · 快照（snapshot）语义与流/事件"
    异步化的铁律：**后台任务不得读会被主线程覆写的状态**。三类输入三种处理——device 张量（发非阻塞 D2H + event，读之前 `wait()`）、持久 buffer（深拷）、纯 Python 结构（浅拷够用就浅拷）。专用拷贝流的意义：D2H 与主流的 compute **重叠**，不抢主流的执行顺序。事件（event）是流之间的同步原语：record 在拷贝流，wait/synchronize 在消费方。这套"流水线 + 快照"是所有推理引擎异步优化的通用范式（async scheduling、异步前缀缓存写、异步输出，全是同构的）。

---

## §杂项

| 函数 | 行 | 作用 |
|---|---|---|
| `_model_omni_flag` / `_runner_model_omni_flag` | :1177/:1180 | 安全读模型上的布尔声明（模型可能没有该属性甚至还没加载） |
| `_model_omni_pooler_payload_include_hidden` | :1183 | 读 `omni_pooler_payload_include_hidden`（默认 True）；talker 类设 False 省 hidden D2H |
| `_should_return_omni_routed_experts` | :1168 | `enable_return_routed_experts` 且 capturer 已初始化（本次 PR 补的 helper，MoE 路由观测用） |
| `_resolve_global_request_id` | :1792 | 把 stage 内部 req_id 映射回全局 ID（跨 stage KV 迁移要用全局键） |

---

## §Z 对齐现状速查（2026-07 对齐 PR 后）

- **与 GPU 逐字一致（18 个）**：`__init__`、B 簇全部、C 簇大部分、F 簇全部、`_should_*` 开关族、快照族大部分。
- **有意保留的平台差异**：图捕获（ACL/CUDA）、拷贝流（npu/cuda）、grammar bitmask CPU 绕行、`_bookkeeping_sync` 6 元组、execute_model 主干（跟各自基类走）。
- **GPU 独有待移植（B 类，动共享模块）**：`shutdown`、runner-assisted full-attn 三件套（VoxCPM2）、deferred/staged 前缀缓存三件套 + 异步写流水线。
- **NPU 独有**：`_run_post_sample_side_effects`（合理的结构差异，保留）。
- **两个待真机验证点**（已在代码注释标注）：per-request hidden D2H 移到 postprocess 之前（对齐 GPU 顺序）；hidden D2H 条件不带 `omni_prefix_cache is None` 门控（NPU 无 staged 通道的补偿）。

## 面试速答卡

1. **"介绍一下你们 NPU AR runner 的结构"** → 两阶段 execute/sample 主循环 + 四个配件系统（下游路由、张量前缀缓存、payload 构建、异步输出），外加 connector mixin 做跨 stage 数据面。
2. **"异步输出怎么保证正确性"** → 快照语义三分法（device 张量走专用流 D2H + event、持久 buffer 深拷、Python 结构浅拷）+ 六重门控把有风险的组合直接退回同步。
3. **"omni 前缀缓存和 KV 前缀缓存什么关系"** → 复用同一套 block table/slot mapping 寻址，缓的是 hidden+mm 而非 KV；因为 KV 命中后下游 stage 需要的 hidden 不会再被算出来。
4. **"GPU/NPU 对齐最难的是什么"** → 不是逐行 diff，是分清四类：纯搬（设备无关）、换 API（stream/graph）、动共享模块（cuda 门控泛化）、等上游（基类签名分叉）。用逐方法 diff 工具把 33 个方法收敛到 18 个逐字一致，剩余差异每条都能说出归类理由。
