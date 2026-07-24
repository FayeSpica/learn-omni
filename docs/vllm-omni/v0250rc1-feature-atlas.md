---
tags:
  - vllm-omni
  - 特性图鉴
  - v0.25.0rc1
  - 源码分析
  - 全特性
  - 索引
---

# vLLM-Omni 全特性图鉴（v0.25.0rc1 · 按特性分章节）

> 本页对标 [vllm-notes/MMProcessor.md](https://github.com/JaredforReal/vllm-notes/blob/main/MMProcessor.md) 的写法：**每个特性都标注它是什么、为什么存在、落在哪些代码路径（file:line）、有哪些配置旋钮、瓶颈/风险在哪**。分析基线是 tag `v0.25.0rc1`（commit `d3c47ef`，Rebase to vllm v0.25.0 #5042），在独立 worktree 中只读分析，不动主工作树。
>
> **读法**：先看 §0 的三层总览图与特性总表定位；每一章可独立阅读；每章末尾链到本板块已有的深潜笔记（本页是 umbrella 索引，不重复深潜内容）。file:line 为 v0.25.0rc1 快照，跨版本会漂，引用前请复核。

!!! abstract "一句话定位"
    **vLLM-Omni = 架在 vLLM 之上的「多阶段异构任意模态服务框架」**。vanilla vLLM 服务单模型（token 进 token 出）；vLLM-Omni 补的是它缺的那层——**多模型流水编排（Orchestrator）**、**跨阶段隐状态/KV 传递（OmniConnector）**、**非 AR 的扩散生成 stage（DiT）**、**每 stage 可组合并行（Composable Parallel）**、以及**异构输出（文/音/图/视频）的 OpenAI 兼容服务**。详见 [架构地图](architecture-map.md)。

---

## 0. 目录

| 章 | 特性域 | 关键词 |
| --- | --- | --- |
| §1 | 全局架构与三类 omni 模型 | AR / DiT / AR+DiT、E/P/D/G 解耦 |
| §2 | 异步 Omni 架构与多阶段引擎 | AsyncOmniEngine、Orchestrator、StageEngineCore |
| §3 | AR 模块、调度器与前缀缓存 | Thinker/Talker、omni scheduler、prefix cache |
| §4 | DiT 扩散运行时与执行模型 | 去噪步执行、连续/请求级批处理、MoT |
| §5 | 扩散加速与模型覆盖 | Cache-DiT、TeaCache、async-chunk、LoRA、offload、模型族 |
| §6 | 并行策略与可组合并行 | TP/PP/SP/EP/CFG-P/VAE-P/HSDP、per-stage 组合 |
| §7 | 解耦推理、OmniConnector 与 Ray 执行 | 分离部署、连接器传输、协调器、多机 |
| §8 | 入口与 OpenAI 兼容异构输出服务 | chat/speech/image/video API、流式 |
| §9 | 配置系统、Pipeline 注册表与自定义流水线 | OmniConfig/StageConfig、pipeline_registry、patch |
| §10 | 平台后端、量化、注意力与工具链 | NPU/XPU/ROCm/MUSA、FP8/AWQ、bench/profiler、sleep、ComfyUI |
| §11 | 瓶颈/风险总表 + 组件-文件索引 | 优先级、file:line 速查 |

---

## §1. 全局架构：三类 omni 模型 + E/P/D/G 解耦

vLLM-Omni 的立项前提（`docs/design/architecture_overview.md`）：当下**没有单模型能「全模态进、全模态出」**。主流开源 omni 模型几乎都是 **AR + DiT 的组合**，按主结构分三类：

| 类型 | 主结构 | 代表 | 说明 |
| --- | --- | --- | --- |
| DiT 为主、AR 作文本编码器 | DiT | Qwen-Image、Flux | 图像生成基座，AR 只出 conditioning |
| AR 为主、DiT 作多模态生成器 | AR | BAGEL | 统一理解+生成，先出 CoT 文本再出图 |
| AR+DiT 原生端到端 | AR→DiT | Qwen3-Omni | 文/图/音/视频进，文/音出（Thinker→Talker→Code2wav）|

因为没有单模型能全出，框架用 **stage 编排**把 omni-in 的理解结果分发给各自专精一种输出的下游生成器。这就是「omni」当下准确的说法是 *many-in, text(+speech)-out* 的原因（见 [架构地图](architecture-map.md) 的核心论断）。

### 三层心智模型（对标 MMProcessor 的 Phase 图）

```
┌──────────────────────────────────────────────────────────────────────┐
│  L0  Entrypoints / API 层  (CPU, FastAPI)                               │
│      OpenAI 兼容路由 → 解析多模态输入 → 构造 OmniRequest                │
│      Omni / AsyncOmniEngine.generate()                     §8          │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  OmniRequest + sampling_params_list
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  L1  Orchestration 层  (Orchestrator + StageEngineCoreClient)          │
│      按 pipeline 定义把请求路由过多个 stage；管理 stage 生命周期        │
│      CFG companion 请求配对；跨 stage 输出装配             §2 §9        │
└───────────────┬───────────────────────────────┬──────────────────────┘
                │ stage handoff (OmniConnector)  │
                ▼                                ▼
┌──────────────────────────┐      ┌──────────────────────────────────────┐
│  L2a  AR stage(s)         │      │  L2b  DiT stage(s)                     │
│  Thinker / Talker         │ ───► │  去噪步循环 / VAE decode               │
│  KV cache、prefix cache   │ KV/  │  连续批处理、Cache-DiT/TeaCache        │
│  §3                       │ hs   │  §4 §5                                 │
└──────────────────────────┘      └──────────────────────────────────────┘
        每个 stage 可独立配置 TP/PP/SP/EP/CFG-P/VAE-P（Composable Parallel §6）
        stage 可同进程、跨进程、跨节点（Ray）分离部署（Disaggregation §7）
```

### Key Components（源自 architecture_overview.md 的官方划分）

| 组件 | 角色 | 章 |
| --- | --- | --- |
| **OmniRouter** | omni 请求的智能分派 | §8 §9 |
| **EntryPoints** | 离线 `Omni` / 在线 `AsyncOmniEngine` + APIServer | §8 |
| **AR** | 复用 vLLM 的 KV cache 管理，适配 omni | §3 |
| **Diffusion** | 原生实现 + 加速组件 | §4 §5 |
| **OmniConnector** | 基于 E/P/D/G（Encoding/Processing/Decoding/Generation）的全解耦跨 stage 传输 | §7 |

> **E/P/D/G 解耦**是 vLLM-Omni 相对 vanilla vLLM「PD 分离」的推广：不只 prefill/decode，而是把 Encoding（多模态编码）/Processing（AR）/Decoding（vocoder/VAE）/Generation（DiT）四类算力独立成 stage，各自弹性配比资源。以 Qwen3-Omni 为例，Thinker/Talker/Code2wav 被声明为**独立配置的 stage**，运行期由 `Orchestrator` 经 `StageEngineCoreClient` / `StageDiffusionClient` 路由。


---

## §2. Async Omni 架构、分阶段流水线引擎与编排（Orchestration）

> 版本：vLLM-Omni v0.25.0rc1
> 依据文档：`docs/design/module/async_omni_architecture.md`、`docs/design/architecture_overview.md`
> 依据代码：`vllm_omni/engine/*.py`（`async_omni_engine.py`、`orchestrator.py`、`stage_pool.py`、`stage_engine_core_client.py`、`stage_engine_core_proc.py`、`stage_engine_core_proc_manager.py`、`stage_runtime.py`、`output_processor.py`、`membership_controller.py`、`cfg_companion_tracker.py`、`messages.py`、`mm_outputs.py` 等）

### 1. 是什么、为什么存在

vLLM-Omni 要支持的模型不再是单一的 "AR-only" 结构，而是 **AR + DiT 混合流水线**（如 Qwen3-Omni：Thinker(文本 AR) → Talker(语音 token AR) → Code2Wav(vocoder/DiT)）。这类模型的请求生命周期天然是"多段(stage)串行/半并行"的：每个 stage 可能是不同的模型、不同的并行策略（TP/DP/EP）、不同的执行范式（自回归 decode-loop vs. 扩散去噪 step-loop），甚至跑在不同机器上。

`docs/design/module/async_omni_architecture.md` 第 4 节给出了架构演进对比：**旧拓扑**是"每个 stage 一个独立 Worker 进程，Worker 内部再各自起一个 EngineCore 子进程，Worker 与主进程之间用 `mp.Queue` 通信"——这样每加一个 stage 就多一层进程 + 多一份事件循环，且每个 Worker 都要自己实现"轮询 EngineCore 输出 → 处理 → 转发到下一 stage"的逻辑，代码重复、状态分散。

**新架构（当前实现）** 把"跨 stage 路由 / 转发 / 生命周期管理"这件事收敛成**单一的 `Orchestrator`**，运行在主进程的一个**后台线程 + 独立 asyncio 事件循环**里；stage 本身的计算仍然在独立的**子进程**（`StageEngineCoreProc`，对 diffusion stage 是 `DiffusionEngine`）中执行，Orchestrator 通过 ZMQ + msgpack 与之通信。这样：

- 事件循环（协调/路由/编排）与 stage 计算（EngineCore busy-loop / diffusion denoise-loop）被**解耦**：Orchestrator 只做非阻塞的 poll/route/forward，真正的 GPU 计算全部在子进程里，不会互相阻塞主线程或 API 层。
- 请求跨 stage 流转的"胶水逻辑"（下一 stage 输入怎么构造、CFG 伴随请求怎么处理、PD 分离怎么路由、异步分片预热）集中在 `Orchestrator._route_output` / `_forward_to_next_stage` 一处，而不是分散在每个 Worker 里。
- `AsyncOmniEngine` 退化为一个"瘦代理"（thin proxy）：只负责启动 Orchestrator 线程、把请求塞进 `janus.Queue`、把输出从 `janus.Queue` 取出来还给 `AsyncOmni`/`Omni` 入口层。

### 2. 关键组件表

| Component | File:line | Role |
|---|---|---|
| `AsyncOmniEngine` | `vllm_omni/engine/async_omni_engine.py:191` | 面向 `AsyncOmni`/`Omni` 入口的瘦代理；持有 `request_queue`/`output_queue`/`rpc_output_queue` 三个 `janus.Queue`；`__init__` 在后台线程启动 Orchestrator |
| `AsyncOmniEngine._bootstrap_orchestrator` | `async_omni_engine.py:418` | 在独立线程内 `asyncio.new_event_loop()`，调用 `_initialize_stages()` 构建 `StagePool` 列表，再构造并 `run()` 一个 `Orchestrator` |
| `AsyncOmniEngine.add_request` | `async_omni_engine.py:1287` | 本地做 stage-0 输入处理（`InputProcessor.process_inputs`），构造消息后 `request_queue.sync_q.put`；若配置了 CFG，触发 `_enqueue_cfg_companions` |
| `AsyncOmniEngine._enqueue_cfg_companions` | `async_omni_engine.py:786` | CFG 伴随请求的入口：调用 `prompt_expand_func` 生成负向 prompt，逐个构造 `AddCompanionRequestMessage` 入队 |
| `Orchestrator` | `vllm_omni/engine/orchestrator.py:203` | 常驻后台线程事件循环的核心编排器；持有 `stage_pools`、`request_states`、`CfgCompanionTracker`、可选 `MembershipController` |
| `Orchestrator.run` | `orchestrator.py:330` | 启动 `_request_handler` / `_orchestration_output_handler` / (可选) membership watcher 三个协程任务并 `gather` |
| `Orchestrator._request_handler` | `orchestrator.py:396` | 消费 `request_async_queue`：`add_request`/`streaming_update`/`add_companion_request`/`abort`/`collective_rpc`/`shutdown` |
| `Orchestrator._orchestration_loop` | `orchestrator.py:659` | 编排主循环：逐 stage、逐 replica 轮询输出（LLM: `poll_llm_raw_output`；diffusion: `poll_diffusion_output`），处理后路由 |
| `Orchestrator._route_output` | `orchestrator.py:876` | 决定一次 stage 输出是"回给前端(final_output)"还是"转发到下一 stage"还是"作为 CFG 伴随结果缓存" |
| `Orchestrator._forward_to_next_stage` | `orchestrator.py:1178` | 真正的跨 stage handoff：构造下一 stage 的输入（`process_engine_inputs`/`custom_process_input_func`），处理 diffusion CFG 拼接、PD prefill→decode 路由 |
| `Orchestrator._prewarm_async_chunk_stages` | `orchestrator.py:1464` | `async_chunk` 模式下提前给下游 stage 预热连接/资源 |
| `StagePool` | `vllm_omni/engine/stage_pool.py:48` | 一个逻辑 stage 的**多副本(replica)集合** + 负载均衡/亲和路由；封装 `submit_initial`/`submit_update`/`poll_llm_raw_output`/`poll_diffusion_output` |
| `StageClient` (Protocol) | `vllm_omni/engine/stage_client.py:18` | 所有 stage client 的公共元数据/生命周期接口（`stage_type`、`final_output`、`prompt_expand_func`…） |
| `StagePoolLLMClient` / `StagePoolDiffusionClient` (Protocol) | `stage_client.py:68` / `stage_client.py:92` | 分别定义 AR-LLM stage 与 diffusion stage 的 client 调用面（`add_request_async`、`get_output_async` vs `get_diffusion_output_nowait`） |
| `StageEngineCoreClientBase` | `vllm_omni/engine/stage_engine_core_client.py:62` | 头端(head)侧与一个 EngineCore 子进程通信的基类，复用 vLLM `AsyncMPClient` 的 ZMQ/msgpack 传输 |
| `StageEngineCoreClient` / `DPLBStageEngineCoreClient` | `stage_engine_core_client.py:436` / `440` | 分别对应非 DP-LB 与 DP 负载均衡场景的具体 client（继承 vLLM `AsyncMPClient`/`DPLBAsyncMPClient`） |
| `StageEngineCoreProc` | `vllm_omni/engine/stage_engine_core_proc.py:46` | 继承 vLLM `EngineCoreProc`；`run_stage_core` 是子进程入口，跑 EngineCore busy-loop，并可选挂 `OmniCoordClientForStage` 心跳 |
| `StageEngineCoreProcManager` | `vllm_omni/engine/stage_engine_core_proc_manager.py:48` | 替代 vLLM `CoreEngineProcManager`：按 replica 用 `multiprocessing.Process(target=StageEngineCoreProc.run_stage_core)` 拉起子进程 |
| `StageRuntime` / `DistStageRuntime` | `vllm_omni/engine/stage_runtime.py:112` / `723` | 单机模式 / 分布式模式下的 stage 初始化编排：解析 stage 配置 → 拉起进程/连接远端 replica → 组装成 `StagePool` 列表 |
| `create_stage_runtime` | `stage_runtime.py:1069` | 工厂函数，根据是否分布式选择 `StageRuntime` 或 `DistStageRuntime` |
| `MembershipController` | `vllm_omni/engine/membership_controller.py:33` | 分布式 replica 的动态加入/退出监督：轮询 `OmniCoordClientForHub` 的 replica 列表，驱动 `StagePool.add_client/remove_client` |
| `OrchestratorMonitor` / `create_orch_monitor` | `vllm_omni/engine/orchestrator_monitor.py:86` / `76` | 可选诊断监控：按 1s 窗口记录 loop idle/active 计数与各 replica 的 `outputs_queue_size`/`inflight`，落盘 JSON |
| `CfgCompanionTracker` | `vllm_omni/engine/cfg_companion_tracker.py:16` | CFG 伴随请求的 parent↔companion 映射、完成度追踪、延迟转发(`defer_parent`)与清理 |
| `messages.py` | `vllm_omni/engine/messages.py:14` 起 | `EngineQueueMessage` 家族：`StageSubmissionMessage`(18)、`AddCompanionRequestMessage`(32)、`AbortRequestMessage`(42)、`OutputMessage`(83)、`ErrorMessage`(73) 等，均为 `msgspec.Struct` |
| `mm_outputs.py` | `vllm_omni/engine/mm_outputs.py:18` / `109` | `MultimodalPayload`（Mapping 化的张量+元数据容器）与 `MultimodalCompletionOutput`（`CompletionOutput` 的多模态子类） |
| `serialization.py` | `vllm_omni/engine/serialization.py:15` | `serialize_additional_information`/`deserialize_additional_information`：跨进程传递 `additional_information`（如 CFG KV、bridge 状态）的编解码 helper |
| `output_modality.py` | `vllm_omni/engine/output_modality.py:50` | `OutputModality` 位标志枚举（TEXT/IMAGE/AUDIO/LATENT）+ `TensorAccumulationStrategy`（决定增量输出如何 concat/replace） |
| `OmniRequestState` / `MultimodalOutputProcessor` | `vllm_omni/engine/output_processor.py:118` / `465` | 单 stage 内部的输出累积状态机（继承 vLLM `RequestState`/`OutputProcessor`），负责把 EngineCoreOutput 攒成 `OmniRequestOutput` |
| `OmniMasterServer` | `vllm_omni/engine/stage_engine_startup.py:156` | 分布式模式下头端的 TCP 注册服务器，接受远端 replica 的 handshake 注册（对应 `omni_master_address/port`） |
| `StageMetadata` / `extract_stage_metadata` | `vllm_omni/engine/stage_init_utils.py:327` / `349` | 从 `OmniEngineArgs`/stage config 提取出 `prompt_expand_func`、`cfg_kv_collect_func`、`custom_process_next_stage_input_func` 等运行期 hook |

### 3. 请求生命周期 / 跨 stage 数据流

#### 3.1 总体拓扑（对应 `async_omni_architecture.md` §1）

```text
┌───────────────── API Layer ─────────────────┐
│ AsyncOmni.generate() / Omni.generate()       │
└───────────────────┬───────────▲──────────────┘
     add_request()   │           │ try_get_output_async()
┌────────────────────▼───────────┴──────────────┐
│ AsyncOmniEngine (thin proxy)                   │
│  request_queue (janus)   output_queue (janus)  │
└──────────┬──────────────────────▲──────────────┘
           │                      │
┌──────────▼──────────────────────┴──────────────┐
│ Orchestrator（后台线程 + 独立 asyncio loop）      │
│  _request_handler()      _orchestration_loop()  │
│  route_output() / forward_to_next_stage()       │
└───┬──────────────┬───────────────┬──────────────┘
    │ ZMQ/msgpack   │ ZMQ/msgpack    │ ZMQ/msgpack
┌───▼──────┐   ┌───▼──────┐    ┌───▼───────────┐
│StageEngine│   │StageEngine│   │StageDiffusion │
│CoreClient │   │CoreClient │   │Client         │
│(stage0)   │   │(stage1)   │   │(stage2)       │
└───┬──────┘   └───┬──────┘    └───┬───────────┘
    │子进程         │子进程          │子进程
┌───▼──────┐   ┌───▼──────┐    ┌───▼───────────┐
│StageEngine│   │StageEngine│   │DiffusionEngine│
│CoreProc   │   │CoreProc   │   │(denoise loop) │
│(Thinker)  │   │(Talker)   │   │(Code2Wav/DiT) │
└──────────┘   └──────────┘    └───────────────┘
```

#### 3.2 一次 generate 请求的时序（对应 §2 arrow steps，行号取自实现）

```text
[1] AsyncOmni.generate(prompt, request_id)
[2] AsyncOmniEngine.add_request(...)                         async_omni_engine.py:1287
      -> InputProcessor.process_inputs() (stage-0 本地处理)
      -> request_queue.sync_q.put(StageSubmissionMessage)     messages.py:18
      -> (若启用 CFG) _enqueue_cfg_companions(...)            async_omni_engine.py:786
[3] Orchestrator._request_handler() 取出消息                   orchestrator.py:396
      -> _handle_add_request(msg)                             orchestrator.py:435
      -> stage_pools[0].submit_initial(request_id, ...)        stage_pool.py:932
[4] Orchestrator._orchestration_loop() 轮询所有 stage/replica   orchestrator.py:659
      -> pool.poll_llm_raw_output() / poll_diffusion_output()
      -> pool.process_llm_raw_outputs() -> OmniRequestOutput
      -> _handle_processed_outputs() -> _route_output()        orchestrator.py:765 / 876
[5] _route_output():
      - 若该 stage 是 final_output stage -> output_async_queue.put(OutputMessage)
      - 若未完成流水线且非 async_chunk -> _forward_to_next_stage()   orchestrator.py:1178
        -> next_client.process_engine_inputs(source_outputs, prompt)
        -> next_pool.submit_initial/submit_update(...)
[6] AsyncOmniEngine.try_get_output_async() 从 output_queue 取出   async_omni_engine.py:1425
[7] AsyncOmni._process_orchestrator_results -> yield OmniRequestOutput
[8] 收到 finished=True 时 generate() 结束；Orchestrator 清理
      request_states via _cleanup_request_ids()                 orchestrator.py:818
```

#### 3.3 消息 / IPC 类型

**主线程 ↔ Orchestrator 线程**（跨线程但同进程，走 `janus.Queue`，因此可以直接传 Python 对象，不需要序列化）：

- `messages.py` 定义了统一基类 `EngineQueueMessage(msgspec.Struct)` (`messages.py:14`)，子类均 `kw_only=True`：
  - `StageSubmissionMessage` (`messages.py:18`)：`type: "add_request"|"streaming_update"`，携带 `prompt`(已处理为 `EngineCoreRequest` 或原始 `PromptType`)、`sampling_params_list`、`final_stage_id`、`final_output_stage_ids`
  - `AddCompanionRequestMessage` (`messages.py:32`)：CFG 伴随请求专用，携带 `companion_id`/`parent_id`/`role`
  - `AbortRequestMessage`(42)、`CollectiveRPCRequestMessage`(47)、`ShutdownRequestMessage`(57)
  - `RegisterRemoteReplicaMessage`(61) / `UnregisterRemoteReplicaMessage`(67)：分布式 replica 上下线通知，交给 `MembershipController`
  - `OutputMessage`(83)：`engine_outputs: OmniRequestOutput` + `metrics: StageRequestMetrics`，是回传给前端的最终载体
  - `StageMetricsMessage`(94)：非 final stage 完成时只回传指标，不回传内容
  - `ErrorMessage`(73)：`fatal` 标志区分"单请求出错"与"整个 stage 挂了"

**Orchestrator ↔ Stage 子进程**（跨进程，必须序列化）：走 ZMQ ROUTER/PULL + msgpack，复用 vLLM 原生 `AsyncMPClient`/`DPLBAsyncMPClient` 的编解码器（`StageEngineCoreClient` 于 `stage_engine_core_client.py:436`），因此传输的仍是 vLLM 的 `EngineCoreRequest`/`EngineCoreOutputs`（Omni 扩展版为 `OmniEngineCoreRequest`/`OmniEngineCoreOutput(s)`，见 `stage_engine_core_proc.py:121` 对解码器类型的 monkey-patch）。

**多模态输出载体**：`mm_outputs.py` 定义了 `MultimodalPayload`（`mm_outputs.py:18`，实现 `collections.abc.Mapping`，把 `tensors`(torch.Tensor) 与 `metadata` 分离存放，支持 dict-like 访问）和 `MultimodalCompletionOutput`（`mm_outputs.py:109`，`CompletionOutput` 子类，附加 `multimodal_output` 字段）。`output_processor.py:118` 的 `OmniRequestState` 负责把多次增量输出按 `TensorAccumulationStrategy`（`output_modality.py:101`：`CONCAT_DIM0`/`CONCAT_LAST`/`APPEND_LIST`/`REPLACE`）累积合并成最终 `RequestOutput`。

**跨进程 additional_information**：`serialization.py:15` 的 `serialize_additional_information`/`deserialize_additional_information` 负责把 `AdditionalInformationPayload`（承载 CFG KV cache 等非标准字段）编解码，供 `OmniEngineCoreRequest` 携带跨进程传输（`orchestrator.py:144` 在构造下一 stage 请求时调用）。

### 4. Orchestrator vs StageEngineCore vs StagePool 职责划分

| 层 | 类 | 职责边界 |
|---|---|---|
| 编排层 | `Orchestrator` (`orchestrator.py:203`) | **只做逻辑路由，不做计算**：维护 `request_states: dict[str, OrchestratorRequestState]`（`orchestrator.py:166`）、决定输出去向（回前端/转发/CFG 缓存）、处理 PD 分离的 KV 参数传递（`_build_pd_decode_params` `orchestrator.py:1098`）、驱动 collective_rpc 广播（`_handle_collective_rpc` `orchestrator.py:599`） |
| 池化/路由层 | `StagePool` (`stage_pool.py:48`) | 代表**一个逻辑 stage 的所有 replica**：负载均衡选择 replica（`pick`/`select_replica_id`，`stage_pool.py:257`/`478`）、request→replica 亲和绑定（`bind`/`release`，`stage_pool.py:387`/`391`）、动态增删 replica（`add_client`/`remove_client`，`stage_pool.py:209`/`232`，供 `MembershipController` 调用）、封装轮询/提交 API（`submit_initial`(932)、`submit_update`(1001)、`poll_llm_raw_output`(1093)、`poll_diffusion_output`(1121)） |
| 传输层 | `StageEngineCoreClient`/`DPLBStageEngineCoreClient` (`stage_engine_core_client.py:436/440`) | 头端持有的 ZMQ client，直接复用 vLLM `AsyncMPClient`/`DPLBAsyncMPClient` 的 ROUTER/PULL socket 与 msgpack 编解码；额外挂了 `process_engine_inputs`(385)、`get_kv_sender_info`(351) 等 Omni 专属的 stage-bridge 逻辑 |
| 执行层（进程） | `StageEngineCoreProc` (`stage_engine_core_proc.py:46`) | 子进程内跑 vLLM 原生 `EngineCoreProc.run_busy_loop()`；`run_stage_core`(55) 是 `multiprocessing.Process` 的 `target`，做 monkey-patch（把解码器类型换成 `OmniEngineCoreRequest`）、注册死亡信号(`set_death_signal`)、可选启动 `OmniCoordClientForStage` 心跳客户端 |
| 进程生命周期 | `StageEngineCoreProcManager` (`stage_engine_core_proc_manager.py:48`) | 替代 vLLM `CoreEngineProcManager`，按 `local_engine_count` 循环 `context.Process(target=StageEngineCoreProc.run_stage_core, kwargs=...)`，每个子进程对应**一个 Omni replica**（`omni_replica_id = omni_replica_base_id + index`），继承父类的存活监控/关闭（`finished_procs`/`shutdown`） |

**Stage 是如何被拉起为进程的**（单机模式，`StageRuntime`，`stage_runtime.py:112`）：`_initialize_stages`（`async_omni_engine.py:360`）→ `create_stage_runtime`（`stage_runtime.py:1069`）→ `StageRuntime.initialize()`（`stage_runtime.py:230`）→ `_build_logical_stage_init_plans`（333）为每个逻辑 stage 生成 `LogicalStageInitPlan`（`stage_init_utils.py:58`，内含每个 replica 的 `ReplicaInitPlan` `stage_init_utils.py:42`）→ `_initialize_stage_replicas`（429）按 replica 分别调用 `_initialize_local_llm_replica`（535，内部走 `launch_stage_replica`/`stage_engine_startup.py:968` → `StageEngineCoreProcManager`）或 `_initialize_local_diffusion_replica`（630，拉起 `DiffusionEngine` 独立进程）→ `_assemble_stage_pools`（683）把每个逻辑 stage 的所有 replica client 组装成一个 `StagePool`。

**分布式模式**（`DistStageRuntime`，`stage_runtime.py:723`）额外引入：`OmniMasterServer`（`stage_engine_startup.py:156`，头端 TCP 注册服务）接受 headless 子进程/独立机器的 `register_stage_with_omni_master`（`stage_engine_startup.py:662`）注册；`create_membership_controller`（`stage_runtime.py:772`）构造 `MembershipController` 注入给 `Orchestrator`。

**membership_controller 与 orchestrator_monitor 的监督分工**：
- `MembershipController`（`membership_controller.py:33`）解决"**副本会消失/加入，Orchestrator 要跟着调整路由**"的问题：`_watch_replica_list`（128）每 `WATCH_INTERVAL_S=0.5`（40）轮询 `OmniCoordClientForHub.get_replica_list()`，diff 出消失的 `(stage_id, addr)`，触发 `handle_unregister`（75）→ `StagePool.invalidate_addr`(88) 释放绑定 + 通过注入的 `cleanup_callback` 回调 `Orchestrator._cleanup_request_ids`；反向地，`Orchestrator._request_handler` 收到 `RegisterRemoteReplicaMessage` 时调用 `handle_register`（68）动态 `pool.add_client`。这是**功能性**（容错/弹性伸缩）的监督。
- `OrchestratorMonitor`（`orchestrator_monitor.py:86`）是**纯诊断性**的：`--enable-orch-monitor` 开启后，`Orchestrator._orchestration_loop` 每轮调用 `note_loop(idle=...)`（`orchestrator.py:759`），按 1 秒窗口滚动统计 loop 忙闲比例和各 replica 的 `outputs_queue_size`/`inflight`，`flush()`（110）时落盘 JSON（路径由 `VLLM_OMNI_ORCH_MONITOR_PATH` 环境变量覆盖，`orchestrator_monitor.py:47`），不参与任何路由决策。

### 5. CFG 伴随请求（Classifier-Free Guidance Companion）流程

CFG 场景（如 Bagel、Qwen-Image 这类"AR 输出 KV cache 提供给 DiT 做 cross-attention 条件"的模型）需要**同时**对正向 prompt 和负向（无条件）prompt 跑一遍 AR stage，拿到两份 KV cache 一起喂给 DiT，避免 DiT 阶段自己重新算一遍负向 context。实现分三步（`docs/design/module/entrypoint_module.md` 引用的架构总览 + 代码对应）：

1. **Prompt 展开**：`AsyncOmniEngine.add_request`（`async_omni_engine.py:1287`）在正常提交请求后，若 `self.prompt_expand_func is not None and final_stage_id > 0`（1332），调用 `_enqueue_cfg_companions`（786）。该函数调用 stage 配置里注入的 `prompt_expand_func`（在 `stage_init_utils.py:342/384` 从 `StageMetadata` 提取，最终来自 `StageEngineCoreClient.prompt_expand_func = metadata.prompt_expand_func`，`stage_engine_core_client.py:146`）把原始 prompt 展开成若干个"伴随 prompt"（例如打上 `cfg_text` role 的默认负向 prompt），每个都构造一个新的 `request_id = f"{parent_id}{ep.request_id_suffix}"`，包装成 `AddCompanionRequestMessage`（`messages.py:32`）塞进 `request_queue`。
2. **KV 并发计算与追踪**：`Orchestrator._handle_add_companion`（`orchestrator.py:532`）收到消息后，调用 `CfgCompanionTracker.register_companion(parent_id, role, companion_id)`（`cfg_companion_tracker.py:50`）登记 parent↔companion 映射，然后像正常请求一样 `stage_pools[0].submit_initial(companion_id, ..., affinity_request_id=parent_id)`——`affinity_request_id` 保证伴随请求被路由到与父请求**相同的 replica**，从而共享同一批次/同一 KV 内存布局。AR stage 因此对正负 prompt 并发跑 batch。
3. **收集与延迟转发**：伴随请求在 stage-0 完成后，`Orchestrator._route_output`（876）检测 `self._cfg_tracker.is_companion(req_id)`（891）为真时，把输出暂存进 `_companion_outputs`（`set_companion_output`，`cfg_companion_tracker.py:56`）并调用 `_handle_cfg_companion_ready`（`orchestrator.py:1042`）；而**父请求**若完成但伴随请求尚未全部完成（`has_companions` 且非 `all_companions_done`，`orchestrator.py:939`），会被 `_cfg_tracker.defer_parent(req_id, output, stage_id)`（942，对应 `cfg_companion_tracker.py:85`）挂起，直到 `on_companion_completed`（`cfg_companion_tracker.py:70`）判定全部伴随请求做完才放行。真正转发到 DiT 时，`_forward_to_next_stage`（`orchestrator.py:1178`）在 `next_pool.stage_type == "diffusion"` 分支里调用 `self._cfg_tracker.pop_companion_outputs(req_id)`（1201）把父输出与所有伴随输出打包成 `diffusion_source_outputs = [output, *companion_outputs]` 一起送进 `next_client.custom_process_input_func`（对应设计文档里的 `cfg_kv_collect_func` 拦截 `cfg_text_past_key_values`），并通过 `_maybe_clone_diffusion_params_for_cfg`（`orchestrator.py:859`）把 `companion_request_ids` 写入 `OmniDiffusionSamplingParams.cfg_kv_request_ids`（对应 `vllm_omni/inputs/data.py:273`）供 DiT worker 侧读取。

生命周期结束时，`abort_parents`（`cfg_companion_tracker.py:109`）和 `cleanup_parent`（99）保证父请求被 abort/清理时，其所有伴随请求 ID 也一并释放/终止。

### 6. 配置项 / 环境变量

| 配置项 | 位置 | 作用 |
|---|---|---|
| `--async-chunk` / `async_chunk: bool` | `arg_utils.py:161`；使用点 `orchestrator.py:231,486,529,934` | 开启后 stage 间不再"等 finished 才转发"，而是走 `_prewarm_async_chunk_stages`（`orchestrator.py:1464`）提前预热下游连接，实现分片/流式转发 |
| `single_stage_mode` / `stage_id` kwarg | `async_omni_engine.py:252-258` | 单进程只跑某一个逻辑 stage（用于分布式 headless 部署），检测到 `kwargs["stage_id"]` 是 int 时自动置位 |
| `--omni-master-address` / `--omni-master-port` | `arg_utils.py:192-193`；服务端 `stage_engine_startup.py:156` (`OmniMasterServer`) | 分布式模式下 headless 子进程/远端 replica 向头进程注册的 TCP 地址，单 stage 模式下必填 |
| `--omni-dp-size-local` | `arg_utils.py:195`；校验 `async_omni_engine.py:265-267` | 单次调用在本进程为该 stage 启动的副本数（进程内 DP），必须 `>=1` |
| `--omni-lb-policy` (default `"random"`) | `arg_utils.py:196`；`stage_runtime.py:92` `_build_load_balancer_factory` | `StagePool` 分布式路由使用的负载均衡策略 |
| `--omni-heartbeat-timeout` (default 30.0s) | `arg_utils.py:197`；校验 `async_omni_engine.py:269-271` | 远端 replica 心跳超时阈值，超时后被视为消失，触发 `MembershipController` 反注册 |
| `enable_orch_monitor` kwarg / `--enable-orch-monitor` | `async_omni_engine.py:242`；`orchestrator_monitor.py:76` | 是否启用 `OrchestratorMonitor` 诊断采样（默认关闭，零开销） |
| `VLLM_OMNI_ORCH_MONITOR_PATH` (env) | `orchestrator_monitor.py:47` | 覆盖诊断 JSON 的落盘路径，默认 `./vllm_omni_orch_monitor_<MMDDHHMM>.json` |
| `VLLM_OMNI_REPLICA_ID` (env, 子进程内设置) | `stage_engine_core_proc.py:113` | 子进程标识自己的 Omni replica id，供日志/指标使用 |
| `log_stats` / `--log-stats` | `async_omni_engine.py:221,241`；`orchestrator.py:265-328` (`_init_metrics_state`) | 是否构建 `OmniPrometheusStatLogger` 暴露每 (stage, replica) 的 `vllm:*` 指标；关闭时不注册到 Prometheus registry |
| `stage_init_timeout` / `init_timeout` | `async_omni_engine.py:216-217`；`_wait_for_orchestrator_init` (`async_omni_engine.py:486`) | Orchestrator 启动与 stage 初始化的整体超时（默认 300s / 600s），超时会尝试优雅关闭并抛 `TimeoutError` |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` (env, 自动设置) | `stage_init_utils.py:435` (`prepare_engine_environment`) | 强制多进程 spawn 方式启动 stage 子进程，避免 fork 相关的 CUDA/线程状态问题 |
| `stage_connector_spec` / `SharedMemoryConnector` | `arg_utils.py:159, 253-257` | 单机内 stage 间默认的张量传递连接器（区别于跨节点的 `OmniConnector`/`mooncake`/`mori` 等） |
| `--enable-sleep-mode` | `arg_utils.py:173,185-189` | 是否为该 stage 开启 GPU 显存池睡眠模式（用于多 stage 共享 GPU 场景下的显存管理） |


---

## §3. vLLM-Omni AR 模块、核心调度器与前缀缓存（Prefix Caching）

> 代码基线：`vllm-project/vllm-omni` tag `v0.25.0rc1`（commit `d3c47efc`）。
> 覆盖范围：`vllm_omni/core/prefix_cache.py`、`vllm_omni/core/sched/*.py`、`vllm_omni/worker/gpu_model_runner.py`、`vllm_omni/worker/gpu_ar_model_runner.py`、`vllm_omni/worker/gpu_ar_worker.py`、`vllm_omni/reasoning/*`、`vllm_omni/request.py`、`vllm_omni/outputs.py`。

---

### 1. AR 模块相对上游 vLLM 的扩展点

vLLM-Omni 的 AR（AutoRegressive）模块**不是重写**vLLM 的 `Scheduler` / `GPUModelRunner`，而是通过继承（inheritance）在关键钩子上叠加 omni 专属逻辑，目的是让多阶段（multi-stage）、多模态（multimodal）、talker/thinker 架构的模型（Qwen2.5-Omni、Qwen3-Omni、Qwen3-TTS、Higgs Audio v3、MiMo-Audio 等）能复用 vLLM 原生的调度、批处理（continuous batching）、KV cache 管理与 CUDA Graph 执行流水线。

叠加的能力可以归纳为 4 类：

1. **文本 + 多模态 token 流的统一驱动**：`OmniGPUModelRunner._preprocess()`（`vllm_omni/worker/gpu_model_runner.py:1518`）在标准的 `input_ids`/`inputs_embeds` 组装流程之外，插入了 `model.preprocess()` / `model.preprocess_batch()` / `model.preprocess_decode_batch()` 钩子，让模型自己决定如何把 prompt token、`prompt_embeds`、`additional_information`（张量/标量元数据）拼装为该 step 的真正输入 embedding。
2. **Talker/Thinker 多码本（multi-codebook）音频 token**：`_init_talker_mtp()`（`vllm_omni/worker/gpu_model_runner.py:196`）检测模型上是否存在 `talker_mtp` 子模块（TTS 的 MTP=Multi-Token-Prediction 码本预测器），并为其分配独立的持久化 GPU buffer（`talker_mtp_input_ids` / `talker_mtp_inputs_embeds` / `last_talker_hidden` / `text_step`，见同文件 217-220 行），在 decode 阶段以**独立 batch**方式对这些一步长度为 1 的请求做 MTP 前向（`_talker_mtp_forward`，`vllm_omni/worker/gpu_model_runner.py:1779`），从而支持"每个文本 token 对应多个音频码本 token"的 talker 结构，且不破坏主 LM 的 CUDA Graph 捕获形状。
3. **隐藏状态（hidden states）跨阶段暴露**：AR 阶段的最终隐藏状态不再只用于计算 logits，还作为 `pooler_output`/`OmniModelRunnerOutput.multimodal_outputs` 的一部分向下一 pipeline stage（如 talker、code2wav）传递，参见 `GPUARModelRunner._build_omni_pooler_payload`（`vllm_omni/worker/gpu_ar_model_runner.py:892`）。
4. **异构生成器（generator）基本支持**：对于非标准 Transformer 解码结构（Convolution/LSTM 等一步生成的 code2wav 类模型），AR 模块提供了姊妹类 `GPUGenerationModelRunner` + `OmniGenerationScheduler`，跳过采样阶段，一步完成整段生成（详见第 3 节）。

#### AR 请求流（ASCII 图）

```
Client
  │  prompt + prompt_embeds(optional) + additional_information(optional)
  ▼
AsyncOmniEngine._upgrade_to_omni_request()        [vllm_omni/engine/async_omni_engine.py:120]
  │  上游 InputProcessor.process_inputs() 产出 EngineCoreRequest
  │  再补充 prompt_embeds / additional_information -> OmniEngineCoreRequest
  ▼
OmniARScheduler.schedule()                        [vllm_omni/core/sched/omni_ar_scheduler.py:212]
  │  调用 super().schedule() (vLLM 原生 continuous-batching 决策)
  │  用 OmniNewRequestData 重新包装 scheduled_new_reqs (256-279行)
  │  用 OmniSchedulerOutput 包裹整体输出 (290-293行, mixin 131行)
  ▼
GPUARWorker.init_device() -> GPUARModelRunner      [vllm_omni/worker/gpu_ar_worker.py:120]
  ▼
GPUARModelRunner.execute_model()                  [vllm_omni/worker/gpu_ar_model_runner.py:940]
  │  _update_states() 维护 persistent batch / prefix-cache 命中标记
  │  _preprocess() 组装 input_ids/inputs_embeds (+ talker MTP + 附加信息覆盖)
  │  _model_forward() 前向 (注入 omni kwargs, 缓存 OmniOutput)
  │  extract_multimodal_outputs() 拆出 text_hidden_states / multimodal_outputs
  │  omni_prefix_cache.schedule_async_write() 异步落盘 hidden states / mm outputs
  │  返回 None，ExecuteModelState 暂存 logits/hidden_states
  ▼
GPUARModelRunner.sample_tokens()                  [vllm_omni/worker/gpu_ar_model_runner.py:1854]
  │  _sample() 采样 (支持 model.prefer_model_sampler 自定义采样器)
  │  按请求切片 hidden_states + 前缀缓存合并结果 -> pooler_output
  │  返回 OmniModelRunnerOutput(sampled_token_ids, pooler_output, multimodal_outputs, ...)
  ▼
OmniARScheduler.update_from_output()               [vllm_omni/core/sched/omni_ar_scheduler.py:295]
  │  维护停止条件 / KV-transfer 触发 / structured-output / spec-decode 统计
  │  产出 OmniEngineCoreOutput(new_token_ids, pooling_output, multimodal_output, ...)
  ▼
MultimodalOutputProcessor (docs 描述) -> RequestOutput.multimodal_output
  ▼
Client / 下一 pipeline stage (talker / code2wav / 下游模型)
```

---

### 2. 关键组件一览表

| Component | File:Line | Role |
|---|---|---|
| `OmniARScheduler` | `vllm_omni/core/sched/omni_ar_scheduler.py:50` | 标准 AR 调度器；继承 `vllm.v1.core.sched.scheduler.Scheduler`；封装 `OmniNewRequestData`、KV-transfer 触发、连接器（connector）输出消费 |
| `OmniARAsyncScheduler` | `vllm_omni/core/sched/omni_ar_scheduler.py:928` | AR 调度器的异步调度变体；继承 `OmniARScheduler` + `vllm.v1.core.sched.async_scheduler.AsyncScheduler` |
| `OmniGenerationScheduler` | `vllm_omni/core/sched/omni_generation_scheduler.py:42` | 面向异构一步生成模型（Diffusion/Conv/LSTM code2wav）的"快路径"调度器 |
| `KVCacheTransferData` | `vllm_omni/core/sched/omni_ar_scheduler.py:40` | 跨阶段 KV cache 传输的数据封装（`layer_blocks`/`block_ids`/`metadata`） |
| `OmniSchedulerMixin` | `vllm_omni/core/sched/omni_scheduler_mixin.py:40` | AR/Generation 调度器共享的工具方法（连接器输出消费、超时清理、`SchedulerOutput` 包装、限流统计） |
| `OmniSchedulingCoordinator` | `vllm_omni/core/sched/omni_scheduling_coordinator.py:85` | 全量负载（full-payload）跨 stage 输入协调器：处理挂起的 chunk / 超时请求 |
| `OmniNewRequestData` | `vllm_omni/core/sched/output.py:11` | 扩展 `NewRequestData`，携带 `prompt_embeds`、`additional_information`、`external_req_id` |
| `OmniSchedulerOutput` | `vllm_omni/core/sched/output.py:88` | 扩展 `SchedulerOutput`，携带 `finished_requests_needing_kv_transfer`、`pending_input_registrations` |
| `omni_routed_experts_for_request` | `vllm_omni/core/sched/utils.py:9` | 依据 `slot_mapping` 从 `RoutedExpertsLists` 中切出单请求的 MoE 路由数据 |
| `OmniTensorPrefixCache` | `vllm_omni/core/prefix_cache.py:33` | Omni 张量前缀缓存核心类：缓存 hidden states / 多模态输出，异步 D2H 写入 |
| `_PendingAsyncWrite` | `vllm_omni/core/prefix_cache.py:17` | 一次异步 GPU→CPU 写入的挂起状态（CUDA event + CPU 暂存张量） |
| `OmniGPUModelRunner` | `vllm_omni/worker/gpu_model_runner.py:76` | 继承 `vllm.v1.worker.gpu_model_runner.GPUModelRunner`；AR/Generation 共享基类 |
| `GPUARModelRunner` | `vllm_omni/worker/gpu_ar_model_runner.py:289` | AR 专用 runner：两阶段 `execute_model()`/`sample_tokens()`，暴露 hidden states，管理前缀缓存合并 |
| `GPUGenerationModelRunner` | `vllm_omni/worker/gpu_generation_model_runner.py` | 一步生成 runner（无采样阶段） |
| `ExecuteModelState` | `vllm_omni/worker/gpu_ar_model_runner.py:272` | `execute_model()`→`sample_tokens()` 之间暂存的 NamedTuple（logits/hidden_states/spec_decode 元数据等） |
| `GPUARWorker` | `vllm_omni/worker/gpu_ar_worker.py:24` | 继承 `vllm.v1.worker.gpu_worker.Worker`；`init_device()` 中实例化 `GPUARModelRunner`（120行） |
| `OmniGPUWorkerBase` | `vllm_omni/worker/base.py:30` | AR/Generation worker 共享基类；封装 sleep/wake、内存 profiling |
| `OmniRequest` | `vllm_omni/request.py:17` | 继承 `vllm.v1.request.Request`，附带 `prompt_embeds_payload`、`additional_information`、`external_req_id` |
| `OmniModelRunnerOutput` | `vllm_omni/outputs.py:40` | 继承 `ModelRunnerOutput`；新增 `multimodal_outputs`、`inter_stage_outputs`、`kv_extracted_req_ids`、`omni_connector_output` |
| `OmniRequestOutput` | `vllm_omni/outputs.py:63` | 统一 pipeline-stage 与 diffusion 输出的外部输出结构 |
| `StepAudioReasoningParser` | `vllm_omni/reasoning/step_audio_reasoning_parser.py:20` | Step-Audio 模型的 `<think>`/`<|THINK_START|>` 推理段解析器，注册名 `step_audio` |
| `sanitize_min_tokens_stop_ids` | `vllm_omni/worker/sampling_utils.py:14` | 修正 codec-talker（lm_head 词表窄于 tokenizer 词表）下 `min_tokens` 越界导致的 CUDA assert |

---

### 3. 调度器（`vllm_omni/core/sched`）的 Omni 专属扩展

#### 3.1 `OmniARScheduler`：在不改变调度算法前提下"富化"输出

`OmniARScheduler`（`vllm_omni/core/sched/omni_ar_scheduler.py:50`）继承 `OmniSchedulerMixin` + 上游 `Scheduler`，其 `schedule()`（212 行）本质是对 `super().schedule()` 结果做**后处理包装**，而非重写调度算法：

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:  # 212行
    ...
    scheduler_output = super().schedule(throttle_prefills)   # 235行：完全复用 vLLM 调度决策
    ...
    for nr in scheduler_output.scheduled_new_reqs:            # 258行起
        omni_nr = OmniNewRequestData(..., prompt_embeds=..., additional_information=...)
    scheduler_output.scheduled_new_reqs = new_list
    return self._wrap_omni_scheduler_output(scheduler_output, finished_requests_needing_kv_transfer=finished_reqs)  # 290行
```

调度前会先做三件"omni 独有"的清理/预处理（217-250 行）：
- 清除 `FINISHED_ABORTED` 请求（上游对该状态会 `raise RuntimeError`，Omni 允许客户端断连时的异步 abort）；
- `_consume_pending_connector_output(model_mode="ar")`（`omni_scheduler_mixin.py:53`）：消费上一 step 模型侧连接器（connector）回传的 `stage_recv_req_ids`/`request_metadata`；
- `chunk_transfer_adapter.process_pending_chunks(...)`：处理跨阶段异步 chunk 到达。

`update_from_output()`（295 行）在标准的"新 token → 判断停止 → 生成 `EngineCoreOutput`"流程之上，额外叠加：
- **KV 传输触发器** `_process_kv_transfer_trigger()`（140 行）：按 `kv_transfer_criteria`（`prefill_finished` 或 `special_token`）判断是否需要把 KV cache 提取给下游 stage，并通过 `pending_stop_after_extraction` 延迟停止请求，确保 `kv_ready` 信号在请求仍存活时发出（162-170 行）；
- **多模态/中间态输出透传**：`mm_output = mm_outputs[req_index]`、`inter_stage_output = inter_stage_outputs[req_index]`（401-402 行）被塞进 `OmniEngineCoreOutput`；
- **routed experts 抽取**：`omni_routed_experts_for_request()`（437 行，工具函数见 `vllm_omni/core/sched/utils.py:9`）。

#### 3.2 `OmniGenerationScheduler`：一步式生成的"快路径"

`OmniGenerationScheduler`（`vllm_omni/core/sched/omni_generation_scheduler.py:42`）面向 code2wav 一类**非自回归、单次前向即完成**的异构生成器阶段。其 `schedule()`（59 行）不复用 `super().schedule()`，而是自实现一个简化调度循环：

- 一次性把请求 `prompt_token_ids` 全部纳入 token 预算（`required_tokens = len(prompt_token_ids) - num_computed_tokens`，108 行），零 token 请求分配 1 个占位符；
- 若预算不足以覆盖 `required_tokens` 且未开启 `enable_chunked_prefill`，直接 `break` 退回默认逐步调度（109-112 行）；
- `update_from_output()`（409 行）在拿到输出后立即将请求标记为 `RequestStatus.FINISHED_STOPPED` 并调用 `_free_request()` 释放，因为该类模型被建模为"运行一次 forward 即完成"。

#### 3.3 `OmniSchedulerMixin`：AR/Generation 共用胶水代码

`OmniSchedulerMixin`（`vllm_omni/core/sched/omni_scheduler_mixin.py:40`）把两个调度器里重复的逻辑收敛为共享方法：`_consume_pending_connector_output`（53 行）、`_process_pending_input_timeouts`（75 行，受环境变量 `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S` 控制，默认 600s，见 29 行）、`_capture_omni_connector_output`（113 行）、`_wrap_omni_scheduler_output`（131 行）、以及限流的 `make_stats`（154 行，每 1 秒才真正统计一次）。

#### 3.4 调度器如何选定（配置层面）

`_resolve_scheduler()`（`vllm_omni/config/stage_config.py:181`）依据 `StageExecutionType` 与 `async_scheduling` 映射到具体调度器类：

| `StageExecutionType` | `async_scheduling=True` | `async_scheduling=False` |
|---|---|---|
| `LLM_AR` | `OmniARAsyncScheduler` | `OmniARScheduler` |
| `LLM_GENERATION` | `OmniGenerationScheduler` | `OmniGenerationScheduler`（不区分） |
| `DIFFUSION` | `None`（走 diffusion 专属调度栈） | `None` |

#### 3.5 为什么这样设计

- **最小侵入**：`OmniARScheduler` 不重写调度算法本身（continuous batching / preemption / prefix-cache block 分配全部委托给上游 `Scheduler.schedule()`），只在其前后叠加 omni 数据搬运，天然继承上游未来的调度优化。
- **一步生成的语义隔离**：把"运行一次即完成"的模型单独建模为 `OmniGenerationScheduler`，避免污染 AR 调度器的 running/waiting 状态机（例如它不需要处理 spec-decode、KV-transfer 触发等 AR 特有逻辑）。

---

### 4. Omni 语境下的前缀缓存（`vllm_omni/core/prefix_cache.py`）

#### 4.1 目标：缓存的是"stage 输出张量"而非 KV cache 本身

`docs/design/feature/prefix_caching.md` 明确指出：vLLM-Omni 的前缀缓存**建立在**（builds on top of）vLLM 原生 KV-cache 前缀缓存之上，是"非侵入式的"（noninvasive）。它复用 vLLM KV cache manager 已经算好的 **block/slot mapping**，去缓存两类**per-token 特征张量**：

1. Stage 产出的最后一层 **hidden states**（用于跨 stage 传递，如 thinker→talker）；
2. 模型/stage 特定的**多模态输出张量**（如 talker 的音频 codec 特征）。

这与 vLLM 自身的 multimodal encoder cache（缓存 ViT/音频 encoder 输出）是**两套独立机制**，但在有前缀命中时会协同工作（见 `prefix_caching.md` 第 133-164 行的 "What About Multimodal Inputs?" 一节）。

#### 4.2 数据结构：镜像 KV cache 的 3D 布局

`OmniTensorPrefixCache.__init__`（`vllm_omni/core/prefix_cache.py:50`）为 hidden states 预分配形状为 `(num_blocks, block_size, hidden_size)` 的 **CPU pinned 张量**（`_get_cache_tensor`，118 行；`pin_memory=torch.cuda.is_available()`）。多模态输出的 cache 张量则**延迟初始化**：`maybe_init_missing_mm_cache_keys()`（84 行）在第一次前向拿到 `multimodal_outputs` 后，动态发现"首维等于 `seq_len` 的 2D+ 张量"作为可缓存 key（排除 1D 的 per-request 标量元数据，如 `ref_code_len`/`codec_streaming`，见 98-107 行注释），据此为每个 key 单独分配 `(num_blocks, block_size, feat_dim)` 的 cache（因为不同 mm 输出的 feature_dim 可能不同）。

复用 vLLM 的 slot mapping 意味着：只要知道 `block_table[req_idx, :num_cached_blocks]`（`_get_cached_block_ids`，728 行），就能把 flatten 后的 `cache[block_ids]` 直接当作该请求已缓存部分的 hidden states/mm 输出取出，无需为 Omni 单独维护 block 分配逻辑。

#### 4.3 写入路径：同步 vs. 异步双通道

- **同步写入** `update_omni_tensor_prefix_cache()`（345 行）：对 `hidden_states`/`multimodal_outputs` 按 `slot_mapping` 做 `index_copy_`（而非更慢的 `aten::index_put_`，见 283-288 行注释）散射进 cache；
- **异步写入流水线**（139-297 行）：`schedule_async_write()`（162 行）在**专用 CUDA stream** 上发起非阻塞 `.to("cpu", non_blocking=True)`，记录 CUDA event 后立即返回；下一次调用时先 `_consume_pending_write()`（266 行）阻塞在**上一次**的 event 上（此时上一步的 GPU 计算大概率已完成，因此同步开销接近于 0）再散射进 cache。`drain_ready_async_writes()`（252 行）提供非阻塞轮询接口，被 `GPUARModelRunner.execute_model()` 在每个 step 开头调用（`vllm_omni/worker/gpu_ar_model_runner.py:963-964`），保证读取 cache 时数据不早于上一 forward step 产出。这是典型的"生产者领先消费者一步"流水线设计，用于隐藏 D2H 拷贝延迟。

#### 4.4 延迟提交（deferred）机制

对于**只在 decode 阶段才产出**多模态输出的 talker（如 Higgs Audio v3 的 audio codes；见 `get_merged_multimodal_states` 616-621 行注释），若每步都做 CPU 缓存写入代价过高，`stage_deferred_mm_outputs()`（416 行）会把 GPU 上的 chunk（`detach()`）暂存在 `_deferred_mm_outputs` 字典里，直到请求结束时才调用 `commit_deferred_mm_outputs()`（456 行）一次性通过 `_get_slot_ids_for_token_range()`（553 行）映射到 block/slot 并写入 CPU cache。模型通过 `deferred_prefix_cache_mm_keys` 属性声明哪些 key 走这条路径（`GPUARModelRunner._deferred_prefix_cache_mm_keys`，`vllm_omni/worker/gpu_ar_model_runner.py:692`）。

#### 4.5 读取/合并路径：cache 命中 + 当前 forward 结果拼接

`_get_merged_tensors()`（673 行）为每个请求：
- 若其 `req_id` 在本 step 是"新调度且带前缀缓存命中"的请求（`_new_req_cache_hit_ids`，由 `add_prefix_cached_new_req_id()` 在 `OmniGPUModelRunner._update_states()` 中登记，见 `vllm_omni/worker/gpu_model_runner.py:549-550`：`new_req_data.num_computed_tokens > 0` 即视为命中），则取出 `cache[block_ids]`（对应已缓存的 prefix 部分）与本次前向新算出的 hidden_states 沿 token 维 `torch.cat`；
- 否则（cache miss）直接透传本次前向结果。

`get_merged_hidden_states()`（649 行）与 `get_merged_multimodal_states()`（607 行）分别对 hidden states 和 mm 输出做上述合并；后者对不可缓存的"透传"（passthrough）数据（如 1D per-request 标量）也会按请求切片（`_coerce_to_payload_dict`，577 行），行为与"未开启前缀缓存"路径保持一致，避免整批数据泄漏到单请求 payload 里。

#### 4.6 与 KV-cache 前缀缓存的关系（差异点）

| 维度 | vLLM 文本 KV cache 前缀缓存 | vLLM-Omni 张量前缀缓存 |
|---|---|---|
| 缓存对象 | attention KV（每层、每 head） | 单层 hidden states + 模型自定义 mm 输出（每 stage 一份，非逐层） |
| 存储位置 | GPU（KV cache pool） | CPU pinned memory（`OmniTensorPrefixCache`，`vllm_omni/core/prefix_cache.py:33`），因为只是"stage 间搬运"用途 |
| Block/slot 分配 | 由 KV cache manager 主动管理 | **复用**同一 block_table/slot_mapping，不单独分配（`vllm_omni/core/prefix_cache.py:41-48` 注释） |
| 覆盖粒度 | 支持 hybrid KV cache group | 当前**只支持单一 kv-cache group** 的 AR stage（`_get_merged_tensors` 690-694 行会对多 group 打 warning 并只用第一组） |
| 是否可关闭 | `enable_prefix_caching`（engine 级） | 每 stage 独立开关：stage config 的 `enable_prefix_caching`（见 §6），且需 `cache_config.enable_prefix_caching` 同时为真才会实例化 `OmniTensorPrefixCache`（`vllm_omni/worker/gpu_model_runner.py:169-175`） |
| 部分块（partial block）语义 | 不缓存未满的最后一块 | 同样不缓存；多模态输入跨块残留部分靠 vLLM 自身的 encoder cache 补齐（`prefix_caching.md` 144-160 行） |

---

### 5. AR Worker/Runner：前向驱动、多模态注入、采样与推理

#### 5.1 两阶段 execute/sample 分离

`GPUARModelRunner`（`vllm_omni/worker/gpu_ar_model_runner.py:289`）严格保留 vLLM 的 `execute_model()`→`sample_tokens()` 两段式协议：

- **`execute_model()`**（940 行）：
  1. `omni_prefix_cache.drain_ready_async_writes()`（963 行）非阻塞消费上一步的异步缓存写入；
  2. 处理跨 stage KV 传输完成信号（`kv_extracted_req_ids`）与 `finished_requests_needing_kv_transfer`（966-993 行）；
  3. `_update_states()`（继承自 `OmniGPUModelRunner`，`gpu_model_runner.py:450`）维护 persistent batch、清理已完成请求、登记前缀缓存命中；
  4. `_preprocess()`（`gpu_model_runner.py:1518`）组装 `input_ids`/`inputs_embeds`/`positions`/`model_kwargs`；
  5. `_model_forward()`（`gpu_model_runner.py:1908`）执行真正的模型前向，并把 `OmniOutput` 缓存到 `self._omni_last_model_output`；
  6. `extract_multimodal_outputs()`（`gpu_model_runner.py:828`）从 `OmniOutput` 中拆出 `text_hidden_states` 与 `multimodal_outputs` 字典；
  7. 若开启前缀缓存，`schedule_async_write()`（1321 行）异步落盘；
  8. 计算 `logits`（`compute_logits`），把所有中间结果打包进 `ExecuteModelState`（1389 行，`NamedTuple`，`vllm_omni/worker/gpu_ar_model_runner.py:272`），**返回 `None`** 以延迟采样。
- **`sample_tokens()`**（1854 行）：取出 `ExecuteModelState`，调用 `_sample()`（1413 行）采样，通过 `_resolve_req_hidden_states()`（1447 行）按请求切出（含前缀缓存合并结果）hidden states，构造 `pooler_output`（`_build_omni_pooler_payload`，892 行）与 `multimodal_outputs`（`_build_multimodal_outputs`，1467 行），最终返回 `OmniModelRunnerOutput`。

#### 5.2 多模态编码器输出注入（M-RoPE / mm embeddings）

`_preprocess()`（`gpu_model_runner.py:1518`）中，当 `self.supports_mm_inputs` 为真（1542 行）时：
```python
with self.maybe_get_ec_connector_output(...) as ec_connector_output:
    self._execute_mm_encoder(scheduler_output)                 # 运行视觉/音频 encoder（1548行）
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed,
)                                                                # 1554行：文本/多模态统一走 embedding 输入
```
这条路径直接调用的是**上游**（vLLM 原生）的 `_execute_mm_encoder`/`_gather_mm_embeddings`/`embed_input_ids`，Omni 未改写，只是在此之后（1638-1657 行）叠加了 `_collect_additional_information_for_prefill()`（1444 行）——把 `prompt_embeds_cpu` 覆盖到 `inputs_embeds` 的 prefill 区段（1460-1465 行），以及模型自定义的 `preprocess()`/`preprocess_batch()` 钩子（1659-1768 行）。

M-RoPE 位置计算复用上游 `_init_mrope_positions()`（`gpu_model_runner.py:309`，处理 image/video/audio grid 元数据得到多维位置编码），Omni 在其外围新增 `_fixup_precomputed_mrope_decode_positions()`（406 行）修正 decode 阶段的位置错位问题。

#### 5.3 Talker MTP（多码本音频 token）驱动

对于配备 `talker_mtp` 子模块的 TTS/omni 模型：decode 阶段单 token 请求会被收集进 `decode_batch_items`（`gpu_model_runner.py:1728`，条件 `span_len == 1 and not is_prefill`），批量调用 `model.preprocess_decode_batch()` 得到 `last_talker_hidden`/`text_step`，再统一调用 `_talker_mtp_forward()`（1779 行）驱动 `self.talker_mtp(...)` 一次性预测多个码本 token（`gpu_ar_model_runner.py`? 实际在 `gpu_model_runner.py:1136` 调用 `self.talker_mtp(...)`）。若开启 full CUDA Graph，`talker_mtp` 会被 `graph_wrapper_cls` 包裹（`_init_talker_mtp`，209-211 行）单独捕获。

#### 5.4 采样：标准采样器 vs. 模型自定义采样器

`_sample()`（`gpu_ar_model_runner.py:1413`）默认走 `self.sampler(logits, sampling_metadata)`；但若模型声明 `prefer_model_sampler=True` 且暴露 `model.sample()`（1422 行），则绕过标准 GPU 采样器、改由模型自身实现采样（例如 CosyVoice3 的 RAS 采样器、HunyuanImage3 的 stage-transition 采样器，见 `gpu_model_runner.py:262-268` 注释），此时仍需手动应用 `logit_bias_state`（min_tokens/allowed_token_ids，1426-1432 行）以保持与标准路径一致的约束语义。`sanitize_min_tokens_stop_ids()`（`vllm_omni/worker/sampling_utils.py:14`）额外修正了 codec-talker 场景下 `min_tokens` 逻辑处理器把越界 stop-token-id 写入越界索引导致的 CUDA device-side assert（对应上游 issue #4962）。

#### 5.5 推理（Reasoning）解析

Step-Audio 模型的 `<think>...</think>` 段（既有单 token 特殊标记 `<|THINK_START|>`/`<|THINK_END|>`，也有多 token 文本形式）由 `StepAudioReasoningParser`（`vllm_omni/reasoning/step_audio_reasoning_parser.py:20`）解析，注册名为 `step_audio`（`vllm_omni/reasoning/__init__.py:11`，通过 `ReasoningParserManager.register_lazy_module` 懒加载，在 `--reasoning-parser step_audio` 被引擎解析前完成注册）。该解析器同时支持基于 token id 与基于文本子串两种匹配路径（`_has_end_token_in_ids`/`_has_end_token_in_text` 等，112-161 行），因为 chat template 常用文本形式而部分下游只拿到 token id 序列。

---

### 6. 配置开关 / 环境变量

| 名称 | 位置 | 作用 |
|---|---|---|
| `enable_prefix_caching`（stage 级） | `vllm_omni/config/stage_config.py:433`（`OmniStageCacheConfig`，`vllm_omni/config/omni_config.py:285` 默认 `False`） | 逐 stage 开关 Omni 张量前缀缓存；最终体现为 `cache_config.enable_prefix_caching`，决定是否实例化 `OmniTensorPrefixCache`（`vllm_omni/worker/gpu_model_runner.py:169`） |
| `async_scheduling`（stage 级） | `vllm_omni/config/stage_config.py:` `OmniStageSchedulerConfig`（默认 `True`） | 决定 AR stage 使用 `OmniARScheduler` 还是 `OmniARAsyncScheduler`（`_resolve_scheduler`，`stage_config.py:181`） |
| `scheduler_cls` | `vllm_omni/config/stage_config.py:232,928` | 允许每 stage 显式覆盖调度器类（点分路径），默认由 `_scheduler_path(_resolve_scheduler(...))` 推导 |
| `async_chunk`（`model_config.async_chunk`） | 多处，如 `omni_ar_scheduler.py:85`、`omni_generation_scheduler.py:47` | 决定是否启用 `OmniChunkTransferAdapter`（流式跨 stage chunk 传输）而非全量 payload 协调器 |
| `model_config.omni_kv_config.kv_transfer_criteria` | `omni_ar_scheduler.py:103-114` | 定义 KV-transfer 触发条件：`type=prefill_finished` 或 `type=special_token`（配 `token_id`），以及 `stop_after_transfer` |
| `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S` | `vllm_omni/core/sched/omni_scheduler_mixin.py:29` | 全量 payload 协调器（`input_coordinator`）等待跨 stage 输入的超时秒数，默认 `600`，`<=0` 禁用超时保护 |
| `deferred_prefix_cache_mm_keys`（模型属性） | `vllm_omni/worker/gpu_ar_model_runner.py:692-696` | 模型声明哪些多模态输出 key 走"请求结束时一次性提交"的延迟前缀缓存路径 |
| `prefer_model_sampler`（模型属性） | `vllm_omni/worker/gpu_ar_model_runner.py:1422` | 模型声明使用自定义采样器而非标准 GPU `Sampler` |
| `logitsprocs_need_output_token_ids` / `supports_omni_query_start_loc` / `supports_omni_decode_step_metadata`（模型属性） | `vllm_omni/worker/gpu_model_runner.py:187-194`, `gpu_model_runner.py:1919` | 模型可选特性探测标志，控制是否启用对应 omni 专属 runner 行为 |
| `--reasoning-parser step_audio` | `vllm_omni/reasoning/__init__.py:11` | 启用 Step-Audio 专属推理段解析器 |


---

## §4. vLLM-Omni Diffusion (DiT) 运行时与执行模型

> 代码基线:`v0.25.0rc1`,`vllm_omni/diffusion/`。本章聚焦运行时/调度/执行子系统(`sched/`、`worker/`、`executor/`、`attention/`、`layers/mot/`、`distributed/autoencoders/`),不逐一展开各模型的 `models/*` 具体实现。

### 1. DiT 模块与 AR 模块的本质差异

vLLM-Omni 里有两套完全独立的推理引擎:AR(autoregressive,LLM token-by-token 解码,依赖 vLLM 原生 `PagedAttention`/KV cache/`vllm.v1` scheduler)和 DiT(Diffusion Transformer,并行去噪)。DiT 引擎位于 `vllm_omni/diffusion/`,是一套**平行实现**,不复用 vLLM 的 KV cache scheduler。

| 维度 | AR(自回归 LLM) | DiT(Diffusion) |
|---|---|---|
| 生成范式 | 逐 token 解码,每步产出 1 个新 token,输出长度不确定 | 固定 `num_inference_steps` 次去噪迭代,每步刷新**整张**latent,输出形状（分辨率/帧数）在 `prepare_encode` 阶段就已确定 |
| 状态载体 | KV cache(每层每 token 的 K/V,随生成动态增长) | `DiffusionRequestState.latents`(`vllm_omni/diffusion/worker/utils.py:88`),整块 latent tensor,无缓存增长,只做 in-place 覆盖 |
| 调度粒度 | token/序列级连续批处理(continuous batching),请求随时可插入/退出 | 支持两种模式:①整请求一次前向的 request-level batching;②`step_execution=True` 时的按 denoise-step 连续批处理(仍是实验特性) |
| “完成”判据 | EOS token 或 `max_tokens` | `step_index >= total_steps`,即 `DiffusionRequestState.denoise_completed`(`vllm_omni/diffusion/worker/utils.py:147-152`) |
| 批兼容性判据 | 词表/采样参数基本总能混批 | `SamplingParamsKey`/`RequestBatchSamplingParamsKey`(`vllm_omni/diffusion/sched/interface.py:37-119`)——形状(height/width/num_frames)、CFG、LoRA 身份必须完全一致才能同批,否则退化为串行 |
| 显存热点 | KV cache 随并发线性增长,是显存主要压力源 | 无 KV cache;显存压力来自 latent/attention 激活与 VAE 解码,`peak_memory_mb` 通过 `current_omni_platform.reset_peak_memory_stats()`(`vllm_omni/diffusion/worker/diffusion_model_runner.py:466, 691`)按批采样 |
| Attention | vLLM 自己的 attention backend 选择器(causal, paged) | 独立的 `vllm_omni.diffusion.attention.selector`(non-causal,role-aware,per-role 可配置)——文档明确二者互不影响(`docs/user_guide/diffusion/attention_backends.md:9`) |

一句话总结:AR 是"状态无限增长、步数不定、逐 token 调度";DiT 是"步数固定、状态原地刷新、以 denoise-step 或整请求为调度单元"。这直接决定了 DiT 需要一套单独的 scheduler(`vllm_omni/diffusion/sched/`)、executor(`vllm_omni/diffusion/executor/`)和 worker(`vllm_omni/diffusion/worker/`),而不能套用 vLLM AR 的 `SchedulerInterface`/`ModelRunner`。

### 2. 关键组件表

| 组件 | File:line | 角色 |
|---|---|---|
| `DiffusionEngine` | `vllm_omni/diffusion/diffusion_engine.py:134` | 顶层 orchestrator:持有 scheduler + executor,跑后台 `_busy_loop`(`:377`),对外暴露 `step()`/`step_streaming()`(`:240,258`) |
| `SchedulerInterface` | `vllm_omni/diffusion/sched/interface.py:195` | 调度器抽象契约:`add_request`/`schedule`/`update_from_output` |
| `_BaseScheduler` | `vllm_omni/diffusion/sched/base_scheduler.py:53` | waiting/running 队列、`SamplingParamsKey` 兼容性判断（`_can_schedule_waiting`, `:272`）等公共簿记 |
| `RequestScheduler` | `vllm_omni/diffusion/sched/request_scheduler.py:19` | request-mode 调度策略(`step_execution=False` 时使用) |
| `StepScheduler` | `vllm_omni/diffusion/sched/step_scheduler.py:30` | step-mode 调度策略,按 denoise-step 推进 `_request_progress`(`:35`) |
| `DiffusionSchedulerOutput` | `vllm_omni/diffusion/sched/interface.py:164` | 单次调度周期的产物:新请求、缓存请求 id、finished id 集合 |
| `DiffusionExecutor` (abstract) | `vllm_omni/diffusion/executor/abstract.py:16` | 定义 `execute_request`/`execute_batch`/`execute_step`(`:72,77,82`)三条执行入口 |
| `MultiprocDiffusionExecutor` | `vllm_omni/diffusion/executor/multiproc_executor.py:69` | 默认多进程实现,三条入口分别转发到 worker 侧 `execute_model`/`execute_model_batch`/`execute_stepwise`(`:310,351,370`) |
| `WorkerProc` | `vllm_omni/diffusion/worker/diffusion_worker.py:746` | 每 GPU 一个进程,`worker_busy_loop`(`:902`)通过 ZMQ/共享内存收发消息 |
| `DiffusionWorker` | `vllm_omni/diffusion/worker/diffusion_worker.py:182` | 单 GPU 内的模型执行封装,`execute_model`(`:409`)、`execute_model_batch`(`:432`)、`execute_stepwise`(`:454`) |
| `DiffusionModelRunner` | `vllm_omni/diffusion/worker/diffusion_model_runner.py:95` | 真正跑 pipeline 前向的 runner:`execute_model`(`:511`)/`execute_model_batch`(`:539`)/`execute_stepwise`(`:673`) |
| `InputBatch`（别名 `StepInputBatch`） | `vllm_omni/diffusion/worker/input_batch.py:581,759` | denoise-step 级张量批视图,`make_batch`(`:686`)复用/重建缓冲区,`scatter_latents`(`:739`)写回 |
| `DiffusionRequestBatch` | `vllm_omni/diffusion/worker/request_batch.py:57` | request-level 批的 pipeline-facing 包装,`forward()` 接受它并返回 `list[DiffusionOutput]` |
| `DiffusionRequestState` | `vllm_omni/diffusion/worker/utils.py:53` | runner 侧持久化的每请求状态(latents/timesteps/scheduler/extra) |
| `RunnerOutput`/`BatchRunnerOutput` | `vllm_omni/diffusion/worker/utils.py:179,197` | 执行结果载体,按 `request_id` 索引 |
| `SupportsStepExecution` 协议 | `vllm_omni/diffusion/models/interface.py:46` | 定义 `prepare_encode/denoise_step/step_scheduler/post_decode` 四段式契约 |
| `Attention` | `vllm_omni/diffusion/attention/layer.py:40` | role-aware attention 模块,构造时按 `role`/`role_category` 解析 backend |
| `get_attn_backend_for_role` | `vllm_omni/diffusion/attention/selector.py:95` | per-role → role_category → default → platform default 的四级回退解析 |
| `CacheBackend` 及子类 | `vllm_omni/diffusion/cache/selector.py:10` | TeaCache/Cache-DiT/MagCache/StepCache 统一 `enable()/refresh()` 接口 |
| `MoTQKVParallelLinear` | `vllm_omni/diffusion/layers/mot/mot_qkv_parallel_linear.py:24` | Mixture-of-Tokens 融合 QKV 投影,`forward`(`:128`)按 `text_indices` 是否为空切换 und/gen 路径 |
| `DistributedVaeExecutor` | `vllm_omni/diffusion/distributed/autoencoders/distributed_vae_executor.py:42` | tile/patch 并行 VAE 解码执行器 |
| `set_forward_context` | `vllm_omni/diffusion/forward_context.py:175` | 前向期间暴露并行组信息(sp/dp/cfg/pp)与 attn_metadata |

> 注:`docs/design/module/dit_module.md` 中写的 worker 路径是 `vllm_omni/diffusion/worker/gpu_worker.py` + `class GPUWorker`,但当前代码库中实际文件是 `vllm_omni/diffusion/worker/diffusion_worker.py`,类名为 `DiffusionWorker`(`:182`)。文档在这一处已经过时,本章以真实代码为准。

### 3. 单步去噪执行模型(Step Execution)

#### 3.1 四段式契约

`step_execution=True` 时,pipeline 必须实现 `SupportsStepExecution`(`vllm_omni/diffusion/models/interface.py:46-71`)的四个方法。当前仅 `QwenImagePipeline` 支持(`docs/user_guide/diffusion/step_execution.md:44-47`),其余 pipeline 在模型加载期就会因不满足 `isinstance(pipeline, SupportsStepExecution)`(`vllm_omni/diffusion/models/interface.py:99-102`)而提前失败,不会拖到首个请求才报错。

以 `pipeline_qwen_image.py` 为参照实现:

| 阶段 | 方法 | 位置 | 作用 |
|---|---|---|---|
| 一次性准备 | `prepare_encode(state)` | `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py:761` | 编码 prompt、初始化 latents/timesteps、深拷贝 `self.scheduler` 到 `state.scheduler`(`:789`)避免跨请求共享调度器状态 |
| 去噪前向 | `denoise_step(input_batch)` | 同文件 `:903` | 读取批视图 `InputBatch`,调用 `predict_noise_maybe_with_cfg` 走与 `forward()` 相同的 CFG 路径 |
| 调度器步进 | `step_scheduler(state, noise_pred)` | 同文件 `:950` | 仅做 `state.latents = scheduler_step_maybe_with_cfg(...)` 和 `state.step_index += 1` |
| 最终解码 | `post_decode(state)` | 同文件 `:971` | 调 `_decode_latents`(`:872`)执行 VAE 解码 |

#### 3.2 Runner 侧一次 tick 的执行流程

`DiffusionModelRunner.execute_stepwise`(`vllm_omni/diffusion/worker/diffusion_model_runner.py:673-802`)是单次 denoise-step 的核心驱动逻辑:

1. `_update_states(scheduler_output)`(`:567`):清理已完成请求的 `state_cache`,为新请求调用 `pipeline.prepare_encode` 前先构造 `DiffusionRequestState`,为已在跑的请求直接复用缓存状态。
2. `_prepare_batch_inputs(states, new_request_ids)`(`:612`):对新请求执行 `pipeline.prepare_encode(state)`(`:627`),再用 `InputBatch.make_batch(states, cached_batch=...)`(`:633`)组装/复用批张量视图。
3. `_prepare_attn_metadata(input_batch)`(`:664`):可选地为模型侧构造 attention metadata。
4. 在 `set_forward_context(...)`(`:695`)内调用 `pipeline.denoise_step(input_batch, states=states)`(`:701`)——**这是唯一的一次真正的 GPU 前向**,一次前向同时服务批内所有兼容请求。
5. 按每请求的 `row_num = req.latents.shape[0]` 对 `noise_pred` 切片,逐请求调用 `pipeline.step_scheduler(req, noise_pred[offset:offset+row_num])`(`:727`)推进状态。
6. 判定是否该请求需要解码(`req.chunk_denoise_completed` 或 `req.denoise_completed`,取决于 `streaming_output`),需要则调用 `pipeline.post_decode(req)`(`:737`)。
7. `_update_states_after(states, input_batch, interrupted)`(`:640`)把 `input_batch.latents` gather 回并 `scatter_latents`(`vllm_omni/diffusion/worker/input_batch.py:739`)写回每个请求的持久 `state.latents`;对已完成请求从 `state_cache` 中弹出。
8. 返回 `BatchRunnerOutput`(每请求一个 `RunnerOutput{step_index, finished, result}`)。

#### 3.3 ASCII 时序图(一个 step tick)

```
DiffusionEngine._busy_loop()                (vllm_omni/diffusion/diffusion_engine.py:377)
   │
   ├─ scheduler.schedule()                  (sched/step_scheduler.py:65 → base_scheduler.py:96)
   │     └─ DiffusionSchedulerOutput{scheduled_new_reqs, scheduled_cached_reqs, ...}
   │
   ├─ execute_fn = executor.execute_step     (executor/multiproc_executor.py:370)
   │     └─ collective_rpc("execute_stepwise", ...) ──ZMQ/shm──▶ WorkerProc.worker_busy_loop
   │                                                             (worker/diffusion_worker.py:902)
   │                                                                │
   │                                                                ▼
   │                                          DiffusionWorker.execute_stepwise (diffusion_worker.py:454)
   │                                                                │
   │                                                                ▼
   │                                    DiffusionModelRunner.execute_stepwise (diffusion_model_runner.py:673)
   │                                          ┌──────────────────────────────────────────┐
   │                                          │ _update_states()          (:567)          │
   │                                          │ _prepare_batch_inputs()   (:612)          │
   │                                          │   └─ pipeline.prepare_encode(state) [新请求│
   │                                          │        only, 一次性]                        │
   │                                          │ InputBatch.make_batch()   (input_batch.py:686)│
   │                                          │   └─ 一次 GPU 前向:                        │
   │                                          │      pipeline.denoise_step(input_batch)   │
   │                                          │        (:701)  ← 唯一 forward pass         │
   │                                          │   └─ 按请求切片 noise_pred                 │
   │                                          │      pipeline.step_scheduler(req, slice)  │
   │                                          │        (:727)  ← 每请求各自推进 step_index │
   │                                          │   └─ 若该请求 denoise_completed:            │
   │                                          │      pipeline.post_decode(req) (:737)      │
   │                                          │      → VAE decode                          │
   │                                          │ _update_states_after() + scatter_latents  │
   │                                          │  (:640, input_batch.py:739)                │
   │                                          └──────────────────────────────────────────┘
   │                                                                │
   │                                          BatchRunnerOutput{RunnerOutput per req}
   ▼
scheduler.update_from_output(sched_output, runner_output)   (sched/step_scheduler.py:68)
   └─ 按 req_output.step_index 推进 progress.current_step；step_index >= total_steps 时标记
      FINISHED_COMPLETED
```

**与 AR 的关键区别**:这里没有 KV cache append/prefill 之分——每个 tick 都是"整 latent 张量的一次完整前向 + 一次调度器数值积分",`total_steps` 在 `prepare_encode` 阶段就已经由 `len(timesteps)` 固定(`vllm_omni/diffusion/sched/step_scheduler.py:124-131`),调度器只是在"第几步"上做记账,不存在变长输出。

### 4. Continuous Batching vs Request-Level Batching

vLLM-Omni 文档明确区分两种批处理机制,二者互斥(由 `step_execution` 布尔值二选一,见 `diffusion_engine.py:178-182`):

| | Request-Level Batching(默认) | Continuous Batching(step-wise) |
|---|---|---|
| 前提 | `step_execution=False`,pipeline 声明 `supports_request_batch=True` | `step_execution=True`,pipeline 实现 `SupportsStepExecution` |
| 批的单位 | 一次完整 `pipeline.forward(batch)`,batch 是**静态**的一组独立请求 | 每个 denoise-step 都可重新决定批组成,请求可在 step 间加入/退出 |
| 调度器 | `RequestScheduler`(`sched/request_scheduler.py:19`) | `StepScheduler`(`sched/step_scheduler.py:30`) |
| 兼容性 key | `RequestBatchSamplingParamsKey`(`sched/interface.py:74-118`) | `SamplingParamsKey`(`sched/interface.py:37-70`,不含 `num_inference_steps`,允许步数不同/进度不同的请求同批) |
| Runner 批张量 | `DiffusionRequestBatch`(`worker/request_batch.py:57`) | `InputBatch`/`StepInputBatch`(`worker/input_batch.py:581,759`) |
| Executor 入口 | `execute_batch`(整请求一次 RPC,`multiproc_executor.py:351`) 或退化的 `execute_request`(逐请求 RPC,`:310`) | `execute_step`(`multiproc_executor.py:370`) |
| 何时批量增大 | `max_num_seqs>1` + 可选 `request_batch_max_wait_ms`(准入等待窗口,`diffusion_engine.py:443-497`) | `--step-execution --max-num-seqs 8`(`docs/design/feature/diffusion_continuous_batching.md:53-58`) |
| 稳定性 | 相对成熟——静态 batch,请求身份/abort/error 独立 | 官方标注为 **experimental**(`docs/design/feature/diffusion_continuous_batching.md:3-5`);`cache_backend` 不支持,KV transfer 等 request-mode 特性未接入 |
| 权衡 | 无法在请求进行中途插入新请求;不适合极长 step 数场景下的低 MFU 问题 | 能在低 MFU/突发流量场景下提升吞吐/利用率;不保证降低单请求延迟(`docs/design/feature/diffusion_continuous_batching.md:25-27`) |

两种机制共享同一条 FIFO 准入规则:`_can_schedule_waiting`(`sched/base_scheduler.py:272-277`)——只有当前 running 批为空,或等待请求的 key 与当前 running key 完全相等,才允许加入;队首不兼容请求会阻塞其后兼容请求(FIFO head-of-line blocking),这是两种批处理共同的限制。

`DiffusionEngine.__init__`(`diffusion_engine.py:200-205`)决定 `execute_fn` 走哪条路径:

```python
if self.step_execution:
    self.execute_fn = self.executor.execute_step
elif self.supports_request_batch:
    self.execute_fn = self.executor.execute_batch
else:
    self.execute_fn = self.executor.execute_request
```

request-batch 模式下,`_wait_for_request_batch_admission_locked`(`diffusion_engine.py:443`)在持锁状态下轮询等待窗口:仅当 `running==0`、`request_batch_max_wait_ms>0` 时生效,退出条件是等待队列达到 `max_num_running_reqs`、队列在 `stable_window_s` 内不再增长、或到达 deadline(`:459-486`)。默认值 `request_batch_max_wait_ms=0.0`(`data.py:792`)表示禁用等待,不引入额外延迟。

### 5. MoT 配置与 Attention Backend 选择

#### 5.1 MoT(Mixture-of-Tokens,注意不是 Mixture-of-Transformers——`docs/user_guide/diffusion/mot_config.md:2` 中原文即为 "Mixture-of-Tokens")

用于 BAGEL 等模型:文本 token 与 VAE(图像)token 共享同一个线性层但走不同权重分支。核心实现:

- `MoTQKVParallelLinear`(`vllm_omni/diffusion/layers/mot/mot_qkv_parallel_linear.py:24`):文本权重直接存在 `self.weight`(标准 `QKVParallelLinear` 创建),VAE 权重存放在子模块 `self.gen_exp` 中;`forward(input_, text_indices, vae_indices)`(`:128`)——当 `text_indices is None` 时完全复用父类 `forward`(纯文本/und 模式),否则调用 `_mot_gemm_dispatch` 走融合 GEMM(gen 模式,`:139`)。
- `MoTRowParallelLinear`(`vllm_omni/diffusion/layers/mot/mot_row_parallel_linear.py:22`)、`MoTRMSNorm`(`vllm_omni/diffusion/layers/mot/mot_layernorm.py:16`)是同样的双分支模式在 Row-parallel 投影和 RMSNorm 上的对应实现。
- 底层融合 kernel `invoke_mot_gemm`(`vllm_omni/diffusion/layers/mot/ops/mot_gemm.py:704`)依赖 Triton auto-tune 配置,三级加载顺序(`docs/user_guide/diffusion/mot_config.md:29-32`):①`$VLLM_TUNED_CONFIG_FOLDER/<filename>` 环境变量覆盖 → ②`vllm_omni/diffusion/layers/mot/configs/<filename>` 内置配置 → ③保守默认配置(到处能跑但次优)。配置按 `device_name=<GPU>,dtype=<DTYPE>.json` 命名,内部再按 `(K,N)` 矩阵形状 → batch size `M` 索引 tile 参数。
- 调优命令:`python benchmarks/kernels/mot_linear_benchmarks.py --model ByteDance-Seed/BAGEL-7B-MoT --tp-size 1 --dtype w16a16 --tune --save-dir vllm_omni/diffusion/layers/mot/configs/`(`docs/user_guide/diffusion/mot_config.md:39-42`)。

#### 5.2 Attention Backend 选择(role-aware selector)

每个 `Attention`(`vllm_omni/diffusion/attention/layer.py:40`)在构造时声明语义 `role`(如 `"self"`/`"cross"`/模型自定义字符串)和可选 `role_category`。解析函数 `get_attn_backend_for_role`(`vllm_omni/diffusion/attention/attention/selector.py:95`,实际路径 `vllm_omni/diffusion/attention/selector.py:95`)按四级优先顺序解析:

```
1. attention_config.per_role[role]            精确匹配      --diffusion-attention-config.per_role.<role>.backend
2. attention_config.per_role[role_category]   类别兜底       (例如 "ltx2.audio_to_video" → "cross")
3. attention_config.default                    全局默认      --diffusion-attention-backend / DIFFUSION_ATTENTION_BACKEND
4. current_omni_platform.get_diffusion_attn_backend_cls(...)  平台默认（按硬件挑最优 kernel）
```

`build_attention_config`(`vllm_omni/diffusion/data.py:1423-1451`)是唯一的规范化入口,在 `OmniDiffusionConfig.__post_init__` 中调用一次,负责把 CLI/dict 形式与 `DIFFUSION_ATTENTION_BACKEND` 环境变量(`data.py:1437`)统一解析为 `AttentionConfig`。`_cached_get_backend_cls`(`attention/selector.py:50-67`)按 `(backend_name, head_size)` 缓存平台解析结果,避免重复的硬件能力检查和日志刷屏。

可用后端(`docs/user_guide/diffusion/attention_backends.md:15-24`):`FLASH_ATTN`、`CUDNN_ATTN`(Blackwell 默认,mask-heavy DiT 上比 SDPA 快 2×)、`FLASHINFER_ATTN`、`TORCH_SDPA`(始终可用的保守 fallback)、`SAGE_ATTN`/`SAGE_ATTN_3`(INT8 量化,有损但视觉上通常无差异)、`FLASH_ATTN_HUB`/`FLASH_ATTN_3_HUB`(HuggingFace kernels hub,用于训练/rollout 数值对齐)。Ring/Ulysses 序列并行(`vllm_omni/diffusion/attention/parallel/ring.py`, `ulysses.py`)是构建在这些 backend 之上的**通信模式**而非 kernel 实现,二者正交。

### 6. VAE 编解码在 pipeline 中的位置

VAE 只出现在两个边界点,denoising loop 本身完全在 latent 空间进行,不接触 VAE:

```
prepare_encode()                    denoise loop (N 步，纯 latent)                post_decode()
   │                                        │                                          │
   text_encoder.encode_prompt()   →   latents = scheduler.step(                →  vae.decode(latents)
   latents = randn / vae.encode          transformer(latents, t, ...))              (only here)
   (image-to-image / edit 场景)           重复 num_inference_steps 次
```

- **编码侧**（可选，仅 image-to-image/edit 类 pipeline）:在 `prepare_encode`/`forward` 早期若有条件图像输入,会调用 VAE encode 得到 `image_latents`,拼进 `latent_model_input`(参见 `pipeline_qwen_image.py:842-843` 的 `torch.cat([latents, image_latents], dim=1)`)。
- **解码侧**:`_decode_latents`(`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py:872-901`)是唯一调用 `self.vae.decode(...)`(`:897`)的地方,由 `post_decode(state)`(`:971-983`)在 denoise 循环**结束之后**统一调用一次(或 `streaming_output` 场景下按 chunk 边界多次调用)。
- **分布式/大分辨率场景**:`DistributedVaeExecutor`(`vllm_omni/diffusion/distributed/autoencoders/distributed_vae_executor.py:42`)提供 tile/patch 并行解码——把 latent 切分成网格 tile(`GridSpec`,`:18`)、按 `DistributedOperator{split, exec, merge}`(`:36`)分发到 `dit_group`(`get_dit_group()`,`:12`)内各 rank 执行,再 gather/unpack(`_unpack_tiles`,`:93`)拼回完整输出,避免高分辨率视频/图像解码时的显存爆炸。各模型专属 VAE 实现(`autoencoder_kl_wan.py`、`autoencoder_kl_qwenimage.py`、`autoencoder_kl_hunyuan.py`、`autoencoder_kl_ltx2.py` 等)都在同一目录下按模型继承基础 `autoencoder_kl.py`。
- **显存开关**:`vae_use_slicing`/`vae_use_tiling`(`data.py:659-660`)控制 VAE 自身的 slicing/tiling 优化(与上面的分布式 tile 并行是两套机制,前者是单卡内存优化,后者是多卡并行)。

### 7. 关键配置项 / 环境变量

均定义于 `vllm_omni/diffusion/data.py` 的 `OmniDiffusionConfig`(`:573`起)及 `vllm_omni/diffusion/envs.py`:

| 配置项 / 环境变量 | 位置 | 作用 |
|---|---|---|
| `step_execution: bool` | `data.py:780` | 打开 step-wise 执行(四段式契约),CLI: `--step-execution` |
| `max_num_seqs: int = 1` | `data.py:786` | 两种批处理模式共享的容量上限,驱动 `_BaseScheduler.max_num_running_reqs`(`sched/base_scheduler.py:75-79`) |
| `request_batch_max_wait_ms: float = 0.0` | `data.py:792` | request-level batching 的准入等待窗口(毫秒),`0` 禁用 |
| `streaming_output: bool` | `data.py:783` | 打开后强制 `step_execution=True`(`diffusion_engine.py:171-174`),按 chunk 边界流式解码 |
| `cache_backend: str = "none"` | `data.py:600` | `"cache_dit"`/`"tea_cache"`/`"mag_cache"`/`"step_cache"`(`cache/selector.py:31-49`);step 模式下不支持(`diffusion_model_runner.py:681-682`) |
| `diffusion_attention_config` | `data.py:587` | 结构化 per-role attention 配置(`AttentionConfig`/`AttentionSpec`) |
| `DIFFUSION_ATTENTION_BACKEND`(env） | `data.py:1437` | 全局 attention backend 简写,等价于 `--diffusion-attention-config.default.backend` |
| `DIFFUSION_CACHE_BACKEND` / `DIFFUSION_CACHE_ADAPTER`(env） | `data.py:81-83` | `cache_backend` 未显式提供时的环境变量兜底 |
| `distributed_executor_backend: str = "mp"` | `data.py:614` | 目前只有 `"mp"`(`MultiprocDiffusionExecutor`)可用,`"ray"`/`"external_launcher"` 尚未实现(`executor/abstract.py:37-44`) |
| `parallel_config: DiffusionParallelConfig` | `data.py:143,597` | `tensor_parallel_size`/`sequence_parallel_size`/`ulysses_degree`/`ring_degree`/`data_parallel_size`/`pipeline_parallel_size` 等正交并行度 |
| `vae_use_slicing` / `vae_use_tiling` | `data.py:659-660` | 单卡 VAE 显存优化开关 |
| `enable_cpu_offload` / `enable_layerwise_offload` | `data.py:652,654` | DiT 与文本编码器互斥式 CPU offload / block 级 layerwise offload |
| `diffusion_kv_cache_dtype` | `data.py:768` | **注意**:这是 diffusion attention 自身的 FP8 量化开关,与 vLLM AR 的 `--kv-cache-dtype` 无关(DiT 无跨 step KV cache) |
| `MASTER_ADDR`/`MASTER_PORT`/`CUDA_HOME`/`LOCAL_RANK`(env） | `envs.py:16-28` | 分布式初始化用的运行时环境变量,同名逻辑照搬自 xDiT(`envs.py:1-3` 注明来源) |

---

**关键源文件速查**:调度 `vllm_omni/diffusion/sched/{interface,base_scheduler,request_scheduler,step_scheduler}.py`;执行 `vllm_omni/diffusion/executor/{abstract,multiproc_executor}.py`;worker `vllm_omni/diffusion/worker/{diffusion_worker,diffusion_model_runner,input_batch,request_batch,utils}.py`;attention `vllm_omni/diffusion/attention/{selector,layer}.py` + `attention/backends/*`;cache `vllm_omni/diffusion/cache/{selector,teacache,magcache,stepcache,cache_dit_backend}.py`;MoT `vllm_omni/diffusion/layers/mot/*`;VAE `vllm_omni/diffusion/distributed/autoencoders/*`;顶层引擎 `vllm_omni/diffusion/diffusion_engine.py`;配置 `vllm_omni/diffusion/data.py`。


---

## §5. vLLM-Omni 扩散加速与模型覆盖（Diffusion Acceleration & Model Coverage）

本章聚焦 `vllm_omni/diffusion/` 目录下的推理加速子系统——缓存（Cache-DiT / TeaCache / MagCache / StepCache）、异步分块（async_chunk）、Diffusion LoRA、CPU/层级 offload、帧插值（RIFE），以及当前支持的 Diffusion/DiT 模型家族全景。所有结论均标注 `file:line`，代码版本为 tag `v0.25.0rc1`。

---

### 1. 缓存加速：Cache-DiT 与 TeaCache

#### 1.1 共同思想

扩散模型的去噪循环在相邻 timestep 上，Transformer block 的中间激活（modulated input / hidden states）往往高度相似。两类算法都利用这一先验，**跳过冗余的 block 前向计算，直接复用前一步缓存的残差（residual）**，从而在几乎不损失质量的前提下换取 1.5×–2× 的加速。区别在于"用什么信号做跳过判断"和"以什么粒度跳过"。

统一入口是抽象基类 `CacheBackend`（`vllm_omni/diffusion/cache/base.py:33`），定义了 `enable(pipeline)` / `refresh(pipeline, num_inference_steps, verbose)` / `is_enabled()` 接口（`base.py:61-101`）；具体后端由 `get_cache_backend()` 按字符串路由（`vllm_omni/diffusion/cache/selector.py:11-49`）：

```python
# vllm_omni/diffusion/cache/selector.py:37-48
if cache_backend == "cache_dit":      return CacheDiTBackend(cache_config)
elif cache_backend == "tea_cache":    return TeaCacheBackend(cache_config)
elif cache_backend == "mag_cache":    return MagCacheBackend(cache_config)
elif cache_backend in ("step_cache", "stepcache", "step_cache_dit"):
                                       return StepCacheBackend(cache_config)
```

除 Cache-DiT / TeaCache 外，仓库还内置了 **MagCache**（`vllm_omni/diffusion/cache/magcache/backend.py:29` `MagCacheBackend`，基于逐步累积的幅值误差 `mag_threshold` 跳步）与 **StepCache**（`vllm_omni/diffusion/cache/stepcache/backend.py:86` `StepCacheBackend`，专用于 DreamZero 的速度余弦相似度跳步），本章以前两者为主。

#### 1.2 Cache-DiT：block 级动态缓存

**技术定位**：基于第三方库 `cache-dit`，在 block 粒度上做 **DBCache（Dynamic Block Cache）**、**TaylorSeer 校准预测** 和 **SCM（Step Computation Masking）** 三种策略的统一封装。

架构与算法输入/输出：

| 组件 | 位置 | 作用 |
|---|---|---|
| `CacheDiTAdapterConfig` | `vllm_omni/diffusion/cache/cache_dit_backend.py:42-53` | 模型侧声明式配置：block 属性名、forward pattern、是否分离 CFG |
| `CUSTOM_DIT_ENABLERS` 注册表 | `cache_dit_backend.py:57` | pipeline 类名 → 自定义 enabler 函数（用于双 transformer / 多 block-list 模型） |
| `_build_db_cache_config()` | `cache_dit_backend.py:123-143` | 把 `DiffusionCacheConfig` 映射为 cache-dit 的 `DBCacheConfig` |
| `enable_cache_for_dit()` | `cache_dit_backend.py:146-194` | 标准单 transformer 模型的默认 enabler，直接调用 `cache_dit.enable_cache(transformer, cache_config=db_cache_config)` |
| `enable_cache_for_wan22()` | `cache_dit_backend.py:239-332` | 双 transformer（Wan2.2 high/low-noise）定制 enabler，用 `BlockAdapter` 包装两个 transformer，并按 `boundary_ratio` 拆分 `num_inference_steps` |
| `CacheDiTBackend` | `cache_dit_backend.py:1024-1156` | 实现 `CacheBackend.enable/refresh`；`enable()` 优先查 `CUSTOM_DIT_ENABLERS`，否则读取 transformer 的 `_cache_dit_adapter_config` 自动构建 `BlockAdapter`（`maybe_build_block_adapter`, `cache_dit_backend.py:1061-1085`） |
| `may_enable_cache_dit()` | `cache_dit_backend.py:1158-1176` | 便捷入口 |

**算法参数（输入）**（`DBCacheConfig`，映射自 `DiffusionCacheConfig`，`vllm_omni/diffusion/data.py:412-450`）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `Fn_compute_blocks` | 1 | 前 N 个 block 强制计算（不缓存），保证特征质量 |
| `Bn_compute_blocks` | 0 | 后 N 个 block 强制计算 |
| `max_warmup_steps` | 4 | 预热步数，warmup 期间不做缓存 |
| `max_cached_steps` | -1（不限） | 超过后禁用缓存防止精度漂移 |
| `max_continuous_cached_steps` | 3 | 连续跳过步数上限 |
| `residual_diff_threshold` | 0.24 | **决策阈值**：block 间残差差异高于该值才重新计算 |
| `enable_taylorseer` / `taylorseer_order` | False / 1 | 是否用 TaylorSeer 多项式外推替代直接复用 |
| `scm_steps_mask_policy` / `scm_steps_policy` | None / "dynamic" | SCM 静态/动态跳步 mask 策略 |

**算法输出**：每次前向要么返回真实计算结果（同时更新缓存残差），要么直接返回 `cached_hidden_states + reused_residual`。`refresh()` 在 `num_inference_steps` 变化时重建 SCM mask 并调用 `cache_dit.refresh_context()`（`cache_dit_backend.py:80-106`）。

#### 1.3 TeaCache：hook 式、无侵入的时间步嵌入相似度缓存

**技术定位**：TeaCache（Timestep Embedding Aware Cache）不修改模型代码，而是通过 `ModelHook` 完全接管 transformer 的 `forward`，用**首个 block 的 modulated input**（归一化+timestep 调制后的特征）作为相似度探针。

核心类 `TeaCacheHook`（`vllm_omni/diffusion/cache/teacache/hook.py:30-278`）：

- `new_forward()`（`hook.py:87-189`）：调用模型专属的 extractor 函数取得 `CacheContext`（含 `modulated_input`、`hidden_states`、`run_transformer_blocks()` 可调用体、`postprocess()`），再执行通用决策逻辑；CFG 正/负分支各自维护独立状态（`hook.py:120-137`，兼容 CFG-parallel）。
- `_should_compute_full_transformer()`（`hook.py:191-238`）—— **算法核心**：
  1. 第一步必算；
  2. 计算相邻 timestep modulated input 的相对 L1 距离：`rel_distance = |x_t - x_{t-1}|.mean() / (|x_{t-1}|.mean() + 1e-8)`；
  3. 用模型专属的 5 次多项式系数 `coefficients` 对距离做非线性 rescale：`rescaled = poly1d(coefficients)(rel_distance)`；
  4. 累加 `accumulated_rel_l1_distance`；若 **低于阈值 `rel_l1_thresh`** 则复用缓存残差（跳过全部 block），否则清零累加器、执行完整 block 计算并重新缓存残差。

**算法输入/输出**：输入是 `modulated_input`（探针张量）+ 历史状态 `TeaCacheState`（`teacache/state.py`）；输出是 `hidden_states + previous_residual`（快路径）或真实 block 输出（慢路径），二者都经 `ctx.postprocess()` 还原为模型标准输出格式。

**配置** `TeaCacheConfig`（`vllm_omni/diffusion/cache/teacache/config.py:89-131`）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `rel_l1_thresh` | 0.2 | 决策阈值；0.2≈1.5×提速/低损，0.4≈1.8×，0.6≈2.0×但明显掉质量 |
| `coefficients` | None（按模型自动查表） | 5 元多项式系数 |
| `transformer_type` | "QwenImageTransformer2DModel" | 自动取 `transformer.__class__.__name__`，用于查 `_MODEL_COEFFICIENTS`（`config.py:9-86`，覆盖 FLUX、Qwen-Image、Bagel、SenseNova-U1、Z-Image、StableAudioDiT、HunyuanImage3、Flux2/Flux2Klein、LongCatImage 等） |

后端 `TeaCacheBackend`（`vllm_omni/diffusion/cache/teacache/backend.py:103-215`）：`enable()` 优先查 `CUSTOM_TEACACHE_ENABLERS`（Bagel/Flux2Klein/HunyuanImage3/SenseNovaU1 的特殊接入，`backend.py:95-100`），否则直接 `apply_teacache_hook(pipeline.transformer, teacache_config)`；`refresh()` 通过 hook registry 的 `reset_hook()` 清空状态（HunyuanImage3 因用 GPT+KV cache 架构，状态在去噪循环内部管理，`refresh()` 是 no-op，`backend.py:189-196`）。

#### 1.4 Cache-DiT vs TeaCache 对比

| 维度 | Cache-DiT | TeaCache |
|---|---|---|
| 缓存粒度 | Block 级（可选跳过部分 block，`Fn/Bn_compute_blocks`） | 整个 transformer 前向级（要么全跳，要么全算） |
| 决策信号 | Block 间残差差异（cache-dit 库内部计算） | 首 block modulated input 的相对 L1 距离 + 多项式 rescale |
| 接入方式 | 依赖模型声明 `_cache_dit_adapter_config` 或走自定义 enabler（半侵入） | 纯 hook 拦截 forward，模型只需提供一个 extractor 函数（零侵入） |
| 支持的复杂技巧 | TaylorSeer 外推、SCM 步跳、双 transformer（Wan2.2） | CFG-aware 状态分离、CFG-parallel 感知 |
| 典型加速 | 与阈值/warmup 相关，文档称约 1.5×-2× | 1.5×-2.0× |
| 后端类 | `CacheDiTBackend`（`cache_dit_backend.py:1024`） | `TeaCacheBackend`（`teacache/backend.py:103`） |

两者互斥地通过 `cache_backend` 参数选择（`"cache_dit"` / `"tea_cache"`），设计文档见 `docs/design/feature/cache_dit.md`、`docs/design/feature/teacache.md`，用户指南见 `docs/user_guide/diffusion/cache_acceleration/{cache_dit,teacache}.md`。

---

### 2. Async Chunk：多阶段流水线的分块异步加速

`async_chunk` **不是扩散去噪循环内部的加速手段**，而是 vLLM-Omni 多阶段模型（Thinker → Talker → Code2Wav，参见 Qwen3-Omni）流水线间的 **IO/计算重叠** 机制；其"Code2Wav"这类生成阶段可以是扩散/流匹配解码器（如 `cosyvoice3_audio`、`stable_audio`、`soulx_singer` 均属扩散类音频解码模型），因此该机制间接加速了这些扩散生成 stage 的端到端延迟。

**加速目标**：避免下一阶段必须等上一阶段整段输出完成才能启动；改为按 chunk（prefill 阶段 `chunk_size=num_scheduled_tokens`，decode 阶段 `chunk_size=1`）流式传递，重叠三段计算，显著降低 TTFP（time-to-first-audio，文档实测 ~92% 降低）。

关键实现（`docs/design/feature/async_chunk.md:262-276` 汇总，逐一核实）：

| 组件 | 文件 | 作用 |
|---|---|---|
| `OmniTransferAdapterBase` | `vllm_omni/distributed/omni_connectors/transfer_adapter/base.py` | 后台 `recv_loop`/`save_loop` 线程基类 |
| `OmniChunkTransferAdapter` | `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py:23` | chunk 生命周期管理；`load_async()`（`:124`）、`save_async()`（`:147`）、`process_pending_chunks()`（`:443`）、`restore_queues()`（`:609`） |
| `OmniARScheduler` | `vllm_omni/core/sched/omni_ar_scheduler.py:50` | AR 阶段调度器，chunk 就绪前置为 `WAITING_FOR_CHUNK` |
| `OmniGenerationScheduler` | `vllm_omni/core/sched/omni_generation_scheduler.py:42` | 生成阶段（含扩散/流匹配解码）调度器，同样接入 chunk transfer adapter |
| stage input processors | `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py` | `thinker2talker_async_chunk`、`talker2code2wav_async_chunk` |
| 配置字段 | `vllm_omni/config/model.py:121`（`async_chunk: bool = False`）、`vllm_omni/engine/arg_utils.py:161`（同名字段，`:442` 为 orchestrator 侧覆盖，`:512` 标注该字段是"orchestrator 读取后再分发给各 stage"的特殊字段） | 启用开关 |

启用方式（stage YAML）：

```yaml
async_chunk: true
stage_args:
  - stage_id: 1
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk
```

---

### 3. Diffusion LoRA 支持

vLLM-Omni 复用 vLLM 的 LoRA 基础设施（`LoRAModel`/`PEFTHelper`/`BaseLayerWithLoRA`），封装出扩散专用管理器 **`DiffusionLoRAManager`**（`vllm_omni/diffusion/lora/manager.py:36-734`）。

**应用流程**（LoRA 如何挂到 DiT 上）：

1. **模块发现**：`_compute_supported_lora_modules()`（`manager.py:113-135`）扫描 pipeline 中的 `LinearBase`/已替换的 `BaseLayerWithLoRA`，得到候选 suffix 集合；`_compute_packed_modules_mapping()`（`manager.py:137-197`）从模型 `load_weights()` 里已有的 `stacked_params_mapping` 反推出 packed（如 `to_qkv`）→ 子投影（`to_q/to_k/to_v`）映射，避免重复声明。
2. **层替换**：`_replace_layers_with_lora()`（`manager.py:340-436`）遍历 pipeline 的 `transformer`/`transformer_2`/`dit`/`bagel`/`unet`（及模型可选的 `_lora_components`）等组件，对匹配 `target_modules`（支持正则或列表）的 `nn.Linear`/`QKVParallelLinear`/`MergedColumnParallelLinear` 调用 `from_layer_diffusion()`（`vllm_omni/diffusion/lora/utils.py:58-87`），依次尝试 `DiffusionMergedQKVParallelLinearWithLoRA`、`DiffusionQKVParallelLinearWithLoRA`、`DiffusionMergedColumnParallelLinearWithLoRA`、`DiffusionColumnParallelLinearWithLoRA`、`DiffusionRowParallelLinearWithLoRA`、`DiffusionReplicatedLinearWithLoRA`（定义于 `vllm_omni/diffusion/lora/layers/*.py`）中第一个 `can_replace_layer()` 返回 True 的包装类，原层被 `replace_submodule()` 原地替换。
3. **加载与激活**：`_load_adapter()`（`manager.py:275-322`）用 `PEFTHelper.from_local_dir()` 读 `adapter_config.json`（`r`, `lora_alpha`, `target_modules`），`LoRAModel.from_local_checkpoint()` 装入 CPU；`_activate_adapter()`（`manager.py:518-632`）把 `lora_a/lora_b`（按 `lora_scale` 缩放 `lora_b`）写入各层的 LoRA buffer，支持 packed/多 slice 权重的拆分匹配。
4. **LRU 缓存 & 动态 rank**：`max_cached_adapters`（即 `OmniDiffusionConfig.max_cpu_loras`）控制 CPU 侧 LRU 淘汰（`_evict_for_new_adapter()`, `manager.py:644-665`）；`pin_adapter()` 可固定常驻；`_ensure_max_lora_rank()`（`manager.py:438-473`）在遇到更大 rank 的 adapter 时重新分配缓冲区。

配置入口在 `OmniDiffusionConfig`：`lora_path`（`vllm_omni/diffusion/data.py:642`）、`lora_scale`（`:643`，默认 1.0）、`max_cpu_loras`（`:644`，初始化校验 `:953-956` 要求 ≥1）。用户指南 `docs/user_guide/diffusion/lora.md` 给出 Python API 用例（`Omni(model=..., lora_path=...)` + 每请求 `LoRARequest`/`lora_scale`）；注意当前版本 **CLI（`vllm-omni serve`）engine_args 未暴露 lora 相关 flag**（`vllm_omni/engine/arg_utils.py` 中无 `lora` 字段），仅可通过 Python API 或配置文件设置。

---

### 4. CPU Offload 与帧插值（Frame Interpolation）

#### 4.1 CPU Offload（两种策略，互斥，layerwise 优先）

统一抽象 `OffloadBackend`（`vllm_omni/diffusion/offloader/base.py:69-99`），策略通过 `OffloadConfig.from_od_config()`（`base.py:29-66`）解析：

```python
# vllm_omni/diffusion/offloader/base.py:50-59
if enable_layerwise_offload:      strategy = LAYER_WISE   # 优先级更高
elif enable_cpu_offload:          strategy = MODEL_LEVEL
else:                             strategy = NONE
```

| 策略 | 实现 | 机制 | file:line |
|---|---|---|---|
| Model-level（Sequential） | `SequentialOffloadHook` + `ModelLevelOffloadBackend` | DiT 与 encoder 互斥驻留 GPU：encoder 前向时 DiT 挪到 CPU，反之亦然；VAE 常驻 GPU；用 pinned memory 加速 H2D | `vllm_omni/diffusion/offloader/sequential_backend.py:18`（hook），`:191`（backend），`apply_sequential_offload` `:120` |
| Layerwise（Blockwise） | `LayerwiseOffloadHook` + `LayerWiseOffloadBackend` | 一次仅 1 个 transformer block 驻留 GPU；用独立 CUDA 流 `copy_stream` 异步预取下一 block（`prefetch_layer`, `layerwise_backend.py:179-216`），当前 block 计算完立即释放；权重以打平后的 pinned CPU tensor 存储以便快速重物化 | `vllm_omni/diffusion/offloader/layerwise_backend.py:22`（hook），`:273`（backend），`apply_block_hook` `:252` |

模块发现：优先走 `SupportsComponentDiscovery` 协议（`_dit_modules`/`_encoder_modules`/`_vae_modules`/`_resident_modules`，`vllm_omni/diffusion/models/interface.py`），否则回退到属性名扫描（`transformer`/`transformer_2`/`dit`/... 等，`docs/user_guide/diffusion/cpu_offload_diffusion.md:169-171`）。Layerwise 要求 DiT 类声明 `_layerwise_offload_blocks_attrs`（如 `WanTransformer3DModel = ["blocks"]`，`Flux2Transformer2DModel = ["transformer_blocks", "single_transformer_blocks"]`）。工厂函数 `get_offload_backend()`（`vllm_omni/diffusion/offloader/__init__.py:32`）按 `OffloadConfig` 选择实现类。

配置：`OmniDiffusionConfig.enable_cpu_offload`（`vllm_omni/diffusion/data.py` 附近 `:652`）、`enable_layerwise_offload`（`:654`）、`pin_cpu_memory`（`:656`，默认 True）；CLI 同名字段见 `vllm_omni/engine/arg_utils.py:476-477`。已验证支持模型见 `docs/user_guide/diffusion/cpu_offload_diffusion.md:190-201` 表格（LongCatImage、NextStep-1.1、Ovis-Image、Qwen-Image、SDXL、SD3.5、Wan2.2 T2V/I2V、SoulX-Singer、Bagel 等，只支持单 GPU）。

#### 4.2 帧插值（RIFE）

后处理阶段（非去噪循环内）功能，在 diffusion worker 侧对已解码视频张量做光流插帧，避免额外占用 API server 的事件循环。

核心实现 `vllm_omni/diffusion/postprocess/rife_interpolator.py`：

- `FrameInterpolator` 类（`:357-432`）：惰性加载 RIFE 4.22.lite 模型（`_ensure_model_loaded`, `:364-380`，模型按 `(resolved_path, device)` 缓存于全局 `_MODEL_CACHE`），核心插值 `interpolate_tensor()`（`:399-432`）：对相邻两帧 `img0/img1` 递归二分插值 `_make_inference()`（`:382-397`，`n = 2**exp // 2` 次内插），输出帧数满足 `(N-1) * 2**exp + 1`，FPS 按 `2**exp` 倍增。
- 便捷函数 `interpolate_video_tensor()`（`:435-443`）。
- 调用点：`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:36`（import）、`:191-196`（`if sampling_params.enable_frame_interpolation: video, multiplier = interpolate_video_tensor(video, exp=..., scale=..., model_path=...)`），同样接入 `pipeline_wan2_2_i2v.py`。RIFE 模型代码 vendored 自 `hzwer/ECCV2022-RIFE` / `Practical-RIFE`（MIT）。

当前仅 `WanPipeline`（Wan2.2 T2V）与 `WanImageToVideoPipeline`（I2V）支持（`docs/user_guide/diffusion/frame_interpolation.md:27-30`）。请求参数：`enable_frame_interpolation`（bool）、`frame_interpolation_exp`（int，默认 1）、`frame_interpolation_scale`（float，默认 1.0）、`frame_interpolation_model_path`（本地目录或 HF repo，含 `flownet.pkl`，默认 repo `elfgum/RIFE-4.22.lite`，`rife_interpolator.py:27`）。

---

### 5. 支持的扩散模型家族一览

来源：`vllm_omni/diffusion/registry.py:22-311`（`_DIFFUSION_MODELS` 注册表，`arch → (mod_folder, mod_relname, cls_name)`），装配为 `DiffusionModelRegistry`（`registry.py:314-321`）。目录结构核对自 `vllm_omni/diffusion/models/`。

| 模型家族 | 目录 (`vllm_omni/diffusion/models/…`) | 类型 | 备注 |
|---|---|---|---|
| Qwen-Image (+Edit/EditPlus/Layered/DMD2) | `qwen_image/` | 文生图/图像编辑 | `QwenImagePipeline` 等 5 个变体；DMD2 为蒸馏加速版 |
| GLM-Image | `glm_image/` | 文生图 | `GlmImagePipeline` |
| Z-Image | `z_image/` | 文生图（少步蒸馏） | Cache-DiT 默认 warmup 参数即为 Z-Image 优化 |
| Ovis-Image | `ovis_image/` | 文生图 | `OvisImagePipeline` |
| FLUX / FLUX Kontext / FLUX DMD2 | `flux/` | 文生图/图像编辑 | `FluxPipeline`, `FluxKontextPipeline`, `FluxDMD2Pipeline` |
| FLUX.2 | `flux2/` | 文生图 | `Flux2Pipeline` |
| FLUX.2 Klein | `flux2_klein/` | 文生图（蒸馏/轻量） | `Flux2KleinPipeline` |
| Krea 2 | `krea2/` | 文生图 | `Krea2Pipeline` |
| LongCat-Image (+Edit) | `longcat_image/` | 文生图/图像编辑 | 多 block-list（`transformer_blocks`+`single_transformer_blocks`），是 Cache-DiT 自定义 enabler 参考实现 |
| HiDream-Image | `hidream_image/` | 文生图 | `HiDreamImagePipeline` |
| Ernie-Image | `ernie_image/` | 文生图 | `ErnieImagePipeline` |
| NextStep-1.1 | `nextstep_1_1/` | 文生图 | 不支持 cache 加速（`_NO_CACHE_ACCELERATION`, `registry.py:326-329`） |
| Stable Diffusion 3(.5) | `sd3/` | 文生图 | `StableDiffusion3Pipeline` |
| Stable Diffusion XL | `sdxl/` | 文生图 | `StableDiffusionXLPipeline` |
| OmniGen2 | `omnigen2/` | 文生图/统一生成 | `OmniGen2Pipeline` |
| Helios / Helios Pyramid | `helios/` | 文生图 | 两个 arch 名共用同一实现类 |
| DreamID-Omni | `dreamid_omni/` | 图像身份定制生成 | `DreamIDOmniPipeline` |
| SenseNova-U1 | `sensenova_u1/` | 统一图像/文本生成 | `SenseNovaU1Pipeline`，TeaCache/CacheDiT 均有专属 enabler |
| Wan2.2（T2V/I2V/S2V/VACE/DMD2） | `wan2_2/` | **视频** 生成 | `Wan22Pipeline`, `Wan22I2VPipeline`, `Wan22S2VPipeline`, `Wan22VACEPipeline`, `WanT2VDMD2Pipeline`, `WanI2VDMD2Pipeline`；唯一支持帧插值(RIFE)与双 transformer Cache-DiT 定制的家族 |
| LTX-2 / LTX-2.3（T2V/I2V/两阶段/DMD2） | `ltx2/` | 视频生成 | 多达 8 个 arch 变体（两阶段、DMD2 蒸馏） |
| Hunyuan Video 1.5（T2V/I2V） | `hunyuan_video/` | 视频生成 | `HunyuanVideo15Pipeline`, `HunyuanVideo15I2VPipeline` |
| Magi-Human | `magi_human/` | 数字人/人像视频生成 | `MagiHumanPipeline` |
| DreamZero | `dreamzero/` | 视频生成（自研 StepCache 速度余弦跳步） | 唯一使用 `step_cache` 后端的家族 |
| Hunyuan-Image-3 | `hunyuan_image3/` | 图像生成（GPT-based + KV cache） | 因架构特殊，TeaCache 走去噪循环内定制状态管理而非 hook |
| Bagel | `bagel/` | Omni 统一理解生成 | Cache-DiT/TeaCache 均有 `enable_cache_for_bagel` / `enable_bagel_teacache` 定制 enabler |
| Ming-Flash-Omni | `ming_flash_omni/` | 图像生成 | 注册为 `MingImagePipeline` |
| Lance | `lance/` | 图像/视频生成 | `LancePipeline` |
| AudioX | `audiox/` | 音频生成 | 不支持 cache 加速（`_NO_CACHE_ACCELERATION`） |
| Stable Audio | `stable_audio/` | 音频生成 | `StableAudioPipeline`，有专属 TeaCache 系数 |
| SoulX-Singer（SVS/SVC） | `soulx_singer/` | 歌声合成 | `PipelineSoulXSingerSVS`, `PipelineSoulXSingerSVC` |
| OmniVoice | `omnivoice/` | 语音合成 | `OmniVoicePipeline`（两个 arch 名指向同一类） |
| CosyVoice3 (audio DiT) | `cosyvoice3_audio/` | 音频 DiT 组件 | **未直接注册在 `DiffusionModelRegistry`**；作为多阶段 Omni 模型的 Code2Wav 生成阶段子模块被 `vllm_omni/model_executor/models/cosyvoice3/cosyvoice3_code2wav.py` 引用，配合 async_chunk 使用 |
| Mammoth-MoDa2 | `mammoth_moda2/` | 图像生成 DiT 组件 | 同上：未直接注册，被 `vllm_omni/model_executor/models/mammoth_moda2/` 多阶段模型复用（同名文件在两处均存在） |
| GR00T N1.5 | `gr00t/` | **具身智能/机器人策略** | `Gr00tN1d7Pipeline`；`policy.py` 面向动作生成 |
| InternVLA-A1 | `internvla_a1/` | **具身智能（VLA）** | `InternVLAA1Pipeline`；内部依赖 `model_cosmos.py`/`cosmos_ci_torch.py`，与 Cosmos 世界模型共享底层组件 |
| **Cosmos (v3)** | `cosmos3/` | **世界模型（World Model）** | `Cosmos3OmniDiffusersPipeline`（`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:1-26`）：单一 pipeline 覆盖 T2I/T2V/I2V/V2V、**控制迁移（`transfer.py`：edge/blur/depth/seg/wsm 引导视频生成）**、以及 **动作条件视频生成（`action.py`：`action_mode`，消费 RoboLab/OpenPI 机器人 observation，输出未来视频/action-only 结果）**——这是仓库中唯一显式实现"世界模型"语义（以动作预测未来观测）的 Diffusion 家族 |
| Diffusers Adapter | `diffusers_adapter/` | 通用适配层 | `DiffusersAdapterPipeline`，用于直接包装社区 diffusers pipeline |
| 编码器组件（非独立 pipeline） | `t5_encoder/`, `mistral_encoder/` | 文本编码器 | 被多个图像/视频家族复用，不在 registry 中单独注册 |

**图像家族小计**：Qwen-Image 系列、GLM-Image、Z-Image、Ovis-Image、FLUX/FLUX.2/Klein、Krea2、LongCat-Image、HiDream、Ernie-Image、NextStep-1.1、SD3/SDXL、OmniGen2、Helios、DreamID-Omni、SenseNova-U1、Hunyuan-Image-3、Bagel、Ming-Flash-Omni、Lance、Mammoth-MoDa2（子模块）。
**视频家族小计**：Wan2.2 全系列、LTX-2/2.3 全系列、Hunyuan Video 1.5、Magi-Human、DreamZero、Cosmos3（T2V/I2V/V2V/控制迁移/动作条件）。
**音频/语音家族小计**：AudioX、Stable Audio、SoulX-Singer、OmniVoice、CosyVoice3（子模块）。
**具身/世界模型小计**：GR00T N1.5、InternVLA-A1、Cosmos3（唯一显式"world model"，支持动作条件预测未来帧）。

---

### 6. 配置旋钮 / 环境变量汇总

集中定义于 `vllm_omni/diffusion/data.py`（`OmniDiffusionConfig`，Python API / 配置文件层）与 `vllm_omni/engine/arg_utils.py`（`OmniDiffusionEngineArgs`，CLI/orchestrator 层，字段在 `:455-490` 附近基本与前者同名镜像）。

| 类别 | 字段 | 默认值 | 位置 | 说明 |
|---|---|---|---|---|
| 缓存开关 | `cache_backend` | `"none"` | `data.py:600`, `arg_utils.py:469` | `"cache_dit"` / `"tea_cache"` / `"mag_cache"` / `"step_cache"` |
| 缓存开关 | `cache_config` | `{}` | `arg_utils.py:470`（CLI 为 JSON 字符串）；Python 侧为 `DiffusionCacheConfig` | 见第 1 节参数表 |
| 缓存环境变量 | `DIFFUSION_CACHE_BACKEND` / `DIFFUSION_CACHE_ADAPTER` | 未设置 | `vllm_omni/diffusion/data.py:81-83` | 当 `cache_backend` 未显式传入时的环境变量兜底 |
| 缓存汇总 | `enable_cache_dit_summary` | `False` | `data.py`/`arg_utils.py:470` | 打开后调用 `cache_summary()`（`cache_dit_backend.py:61-67`）打印命中率等统计 |
| LoRA | `lora_path` | `None` | `data.py:642` | 启动时预加载的静态 LoRA 路径（服务器本地路径） |
| LoRA | `lora_scale` | `1.0` | `data.py:643` | 静态/单请求缩放系数 |
| LoRA | `max_cpu_loras` | `1`（校验 ≥1） | `data.py:644`, `953-956` | LRU 缓存的最大 adapter 数 |
| CPU Offload | `enable_cpu_offload` | `False` | `data.py:652`, `arg_utils.py:476` | Model-level（Sequential）互斥卸载 |
| CPU Offload | `enable_layerwise_offload` | `False` | `data.py:654`, `arg_utils.py:477` | Layerwise（Block 级）卸载，二者同开时优先 layerwise（`offloader/base.py:50-56`） |
| CPU Offload | `pin_cpu_memory` | `True` | `data.py:656` | 是否用 pinned memory 加速 H2D/D2H |
| 帧插值（Sampling 参数，非引擎级） | `enable_frame_interpolation` | `False` | 见 `docs/user_guide/diffusion/frame_interpolation.md:38` | 通过 `/v1/videos` 请求或 `OmniDiffusionSamplingParams` 传入 |
| 帧插值 | `frame_interpolation_exp` | `1` | 同上 | 插帧指数，输出帧数 `(N-1)*2**exp+1` |
| 帧插值 | `frame_interpolation_scale` | `1.0` | 同上 | RIFE 推理 scale |
| 帧插值 | `frame_interpolation_model_path` | `None` | 同上 | 本地目录或 HF repo（默认 `elfgum/RIFE-4.22.lite`，`rife_interpolator.py:27`） |
| 多阶段异步 | `async_chunk` | `False` | `vllm_omni/config/model.py:121`, `arg_utils.py:161` | orchestrator 读取后按 stage 分发（`arg_utils.py:442,512`） |
| 多阶段异步 | `custom_process_next_stage_input_func` | 无 | stage YAML `engine_args` | 指向 chunk 处理函数（如 `qwen3_omni.thinker2talker_async_chunk`） |

**注意**：LoRA 相关字段（`lora_path`/`lora_scale`/`max_cpu_loras`）目前只出现在 `OmniDiffusionConfig`（Python API），未在 `vllm_omni/engine/arg_utils.py` 的 `OmniDiffusionEngineArgs` 中镜像为 CLI flag，即 v0.25.0rc1 尚不支持通过 `vllm-omni serve` 命令行直接指定静态 LoRA，需要用 Python `Omni(...)` 构造或每请求 `LoRARequest`。


---

## §6. vLLM-Omni 并行策略与 Composable Parallel（可组合并行）

> 版本基准：`v0.25.0rc1`，仓库路径 `vllm_omni/`。所有引用均为 `file:line` 格式，指向本次检出的实际源码。

### 1. Composable Parallel 是什么，以及为什么要"按 stage 组合并行"

vLLM-Omni 的推理管线是**多 stage（多引擎）**架构：一个 Omni 模型（如 Qwen2.5-Omni）被拆成 `thinker`（AR 语言模型）、`talker`（AR）、`code2wav`（DiT/生成）等多个 `model_stage`，每个 stage 由独立的 `LLMEngine` / `DiffusionEngine` 承载（`vllm_omni/config/stage_config.py:303` `StageDeployConfig`）。这些 stage 的计算特征天差地别：AR stage 是 token-by-token 自回归解码，DiT stage 是长序列、大 batch 的迭代去噪。因此**同一套并行策略不可能对所有 stage 都最优**——AR 语言模型阶段更关心 TP/PP/DP 这类 vLLM 原生的世界维度，而 DiT 阶段还需要 SP（Ulysses/Ring）、CFG-Parallel、VAE-Parallel、EP、HSDP 等 diffusion 专属的并行轴。

vLLM-Omni 用 **`composable_parallel`**（`vllm_omni/config/composable_parallel/`）把"每个 stage 用什么并行策略"声明化、可组合化：

- 一份 `strategy.yaml` 以 `model_stage` 名称为 key，为每个 stage 声明一组 **`StrategySpec`**（每个对应一条 mesh 轴：`axis` + `size` + `routing` + `l1_owner`），`vllm_omni/config/composable_parallel/spec.py:126-137`。
- `MeshAxisKind` 枚举了全部可声明的并行维度：`tp/dp/pp/ep/stage_replica` 是**当前已打通**（wired）的；`sp_ulysses/sp_ring/cfg/vae_pp/hsdp/stage_pp/cp` 是**保留但未接入 translator** 的维度，声明后会在 `translate_strategy_stack` 中直接抛 `AxisTranslationError`（`vllm_omni/config/composable_parallel/spec.py:35-55`，`vllm_omni/config/composable_parallel/translator.py:50`）。
- `translate_strategy_stack()`（`vllm_omni/config/composable_parallel/translator.py:312-378`）把一叠 `StrategySpec` 翻译成 `OmniParallelConfig`（`translator.py:111-161`），最终映射到真实的 `OmniEngineArgs` 字段名（`tensor_parallel_size` / `data_parallel_size` / `pipeline_parallel_size` / `enable_expert_parallel`），再由 `apply_strategy_specs()`（`vllm_omni/config/composable_parallel/apply.py:244-263`）把结果写回每个 stage 的部署配置。
- 该系统是 **opt-in** 的：只有传 `--strategy-config <path>` 才会在"registry 合并后的 stages"之上叠加派生尺寸；不传则 stage 完全照 deploy YAML 原样运行（`docs/configuration/composable_parallel.md:7-8`）。
- 优先级链条（从高到低）：CLI 覆盖 > strategy YAML > deploy YAML > parser 默认值，冲突时设备布局校验 `check_device_layout` 会在 CLI 覆盖后重新对 `tp*dp*pp*num_replicas` 校验（`docs/configuration/composable_parallel.md:93-100`）。

按 stage 组合并行之所以重要，本质是因为 omni 管线把"AR 理解/生成 token" 与 "DiT 迭代去噪" 强行串在同一个请求生命周期里：如果只有一套全局并行度，要么 AR stage 因为 TP 过大而通信开销压过其收益，要么 DiT stage 拿不到足够的 SP/CFG 并行度来压低单步延迟。Composable Parallel 让每个 stage 独立选择 TP/PP/DP/EP（工程化维度）而不影响其它 stage，同时把 DiT 专属维度（SP/CFG/VAE/HSDP）留在 `DiffusionParallelConfig`（见第 4 节）里按 stage 单独配置。

### 2. 七种并行策略逐项拆解

#### 2.1 Tensor Parallel（TP）

**是什么/为什么**：把 DiT/AR 模型内部大型 Linear 层（attention QKV/输出投影、FFN 上下投影）按列/行切分到多张 GPU，每卡只持有并计算权重的一部分，从而降低单卡显存并获得近线性加速。DiT 侧复用 vLLM 的 `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `ReplicatedLinear`（`docs/design/feature/tensor_parallel.md:32-40`），要求 `num_heads`、`num_kv_heads` 能整除 `tensor_parallel_size`（`tensor_parallel.md:148-156`）。

**适用模块**：DiT（Z-Image、FLUX、Qwen-Image transformer）与 AR（vLLM 原生 TP 机制直接复用）均适用。

**实现位置**：DiT 侧 TP 组的建立在 `vllm_omni/diffusion/distributed/parallel_state.py:866-881`（`initialize_model_parallel` 中构造 `vllm_parallel_state._TP`，复用 vLLM 的 `GroupCoordinator`/`init_model_parallel_group`）；模型示例见 `vllm_omni/diffusion/models/z_image/z_image_transformer.py`（`tensor_parallel.md:264`）。

#### 2.2 Pipeline Parallel（PP）

**是什么/为什么**：将去噪 Transformer 按层切分为若干顺序 stage，每个 PP rank 只持有 `[start_layer, end_layer)` 层，从而降低单卡模型内存，使更大的 DiT 能跨多卡运行。每个去噪 step 内，rank 0 发起前向、中间 rank 收发 `IntermediateTensors`、最后一个 rank 产出噪声预测并执行 scheduler step，再异步把 latents 传回 rank 0（`docs/design/feature/pipeline_parallel.md:32-45`）。

**适用模块**：DiT（Wan2.2 T2V/I2V pipeline 是参考实现）。AR 侧 PP 由 vLLM 原生 PP 机制承担（不在本 mixin 范围内）。

**实现位置**：`PipelineParallelMixin`（`vllm_omni/diffusion/distributed/pipeline_parallel.py:75`），`AsyncLatents` 包装类（`pipeline_parallel.py:20`）；PP 组的建立在 `parallel_state.py:826-834`（`init_model_parallel_group(..., parallel_mode="pipeline")` → `PipelineGroupCoordinator`，`vllm_omni/diffusion/distributed/group_coordinator.py:621`）。层切分默认由 `get_pp_indices()` 均衡分配，可用环境变量 `VLLM_PP_LAYER_PARTITION`（逗号分隔的每 rank 层数）覆盖（`pipeline_parallel.md:180-190`）。要求先继承 `PipelineParallelMixin` 后继承 `CFGParallelMixin`（顺序由 `__init_subclass__` 强制检查，见 `pipeline_parallel.md:99-101`）。

#### 2.3 Sequence Parallel（SP，Ulysses/Ring）

**是什么/为什么**：DiT 处理的图像 patch / 视频帧序列很长，SP 让每张卡只处理序列的一部分，attention 内部通过 Ulysses（all-to-all 交换 QKV heads）或 Ring（环形传递 KV）透明地完成跨卡通信；vLLM-Omni 术语中的 "Sequence Parallelism" 对应 diffusers 的 "Context Parallelism"（`docs/design/feature/sequence_parallel.md:25`）。声明式的 `_sp_plan`（类级 dict，用 `SequenceParallelInput`/`SequenceParallelOutput` 标注哪个 module 边界要切分/收集）是推荐方式，无需侵入 `forward()`（`sequence_parallel.md:85-114`）。`ulysses_mode="advanced_uaa"`（实验特性）放松了严格 Ulysses 对序列长度、head 数可整除的约束（`sequence_parallel.md:50-81`）。

**适用模块**：DiT（LongCat、Qwen-Image、Wan2.2、Z-Image transformer）。

**实现位置**：类型定义 `SequenceParallelInput`（`vllm_omni/diffusion/distributed/sp_plan.py:206`）、`SequenceParallelOutput`（`sp_plan.py:253`）、`SequenceParallelConfig`（`sp_plan.py:52`）；手动切分/聚合工具 `sp_shard`/`sp_gather` 在 `vllm_omni/diffusion/distributed/sp_sharding.py`；hook 机制实现在 `vllm_omni/diffusion/hooks/sequence_parallel.py`；SP 组（含 Ulysses/Ring 子群）由 `set_seq_parallel_pg()` 构造（`parallel_state.py:544-598`），并在 `initialize_model_parallel` 中封装为 `SequenceParallelGroupCoordinator`（`parallel_state.py:836-852`，coordinator 类见 `group_coordinator.py:981`）。

#### 2.4 Expert Parallel（EP）

**是什么/为什么**：MoE 模型（如 HunyuanImage3.0）把不同专家网络分布到不同设备，每卡只持有 `num_experts / ep_size` 个本地专家，token 通过 all-to-all（`allgather_reducescatter` 为默认后端）在设备间派发和收集（`docs/design/feature/expert_parallel.md:20-26`）。路由 gate 用 `ReplicatedLinear`（每卡全量复制），专家层用 `HunyuanFusedMoE` 工厂按平台解析实现（GPU: `FusedMoE`，NPU: `AscendFusedMoE`）。

**适用模块**：仅 DiT 中的 MoE 模型（专家并行不适用于稠密模型或 AR stage）。EP 是一个建立在 TP×SP×CFG×DP 之上的"稠密"标志位而非独立世界维度：`EP_SIZE = TP_SIZE × SP_SIZE × CFG_SIZE × DP_SIZE`（`expert_parallel.md:30-43`）；EP group 是"per pipeline stage"级别的，覆盖除 PP 外所有参与模型并行的 rank，通信模式是 EP 组内的 All-to-All（`expert_parallel.md:47-48`）。

**实现位置**：启用开关 `--enable-expert-parallel` → `enable_expert_parallel: bool = False`（`vllm_omni/diffusion/data.py:155-156`）；EP 组构造在 `parallel_state.py:900-910`（`rank_generator.get_ranks("tp-sp-cfg-dp")` → `vllm_parallel_state._EP`），仅在 `use_moe_parallel_mapping` 为真时建立；composable_parallel translator 中的稠密 EP 校验在 `vllm_omni/config/composable_parallel/translator.py:300-309`（`_validate_ep`，要求 `Broadcast` routing）与 `translator.py:358-368`（EP size 必须等于 `tensor_parallel_size * data_parallel_size`）。模型侧参考实现 `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`。

#### 2.5 CFG-Parallel（Classifier-Free Guidance Parallel）

**是什么/为什么**：标准 CFG 每步去噪需要 2 次（甚至 3/4 次，即 N-branch）前向——条件分支与无条件分支。CFG-Parallel 把这些分支分派到不同 rank 并发执行而非串行，随后用 `all_gather` 收集结果、每个 rank 本地按公式合并（保证结果确定且各 rank 一致），约获得 ~1.8x 加速（`docs/design/feature/cfg_parallel.md:19-30`，`docs/user_guide/diffusion/parallelism/overview.md:11`）。N-branch（如 Bagel/OmniGen2 用 3 分支，DreamID Omni 用 4 分支）通过 round-robin 规则 `branch i -> rank i % M` 分派（`cfg_parallel.md:64-78`）。

**适用模块**：DiT（管线级 mixin，覆盖任意需要 CFG 的去噪循环，如 Qwen-Image、Wan2.2）。

**实现位置**：`CFGParallelMixin`（`vllm_omni/diffusion/distributed/cfg_parallel.py:57`），核心方法 `predict_noise_maybe_with_cfg` / `predict_noise_with_multi_branch_cfg` / `scheduler_step_maybe_with_cfg` / `combine_cfg_noise` / `combine_multi_branch_cfg_noise`；分支分派辅助函数 `_dispatch_branches`（`cfg_parallel.py:39`）；CFG 组构造在 `parallel_state.py:818-825`（`init_model_parallel_group(..., parallel_mode="classifier_free_guidance")`）。模型侧覆盖示例：`vllm_omni/diffusion/models/qwen_image/cfg_parallel.py`。

#### 2.6 VAE Parallel（VAE Patch/Tile Parallelism 与 Spatial-Shard Decode）

**是什么/为什么**：VAE 编解码是纯空间/时间局部运算，把 latent 张量切成若干带 overlap 的空间 tile，分布到多 rank 并行编解码，再在 rank 0 融合（blend 消除接缝），从而分摊计算并降低单卡峰值显存（`docs/design/feature/vae_parallel.md:19-39`）。核心抽象是 `DistributedVaeExecutor`，接受模型无关的 `split`/`exec`/`merge` 三个函数（`vae_parallel.md:48-70`）。Wan VAE 还额外支持 **spatially-sharded decode**（`vae_parallel_mode` 为 `spatial_shard_height`/`spatial_shard_width`）：不是独立 tile 而是把一整张 feature map 沿 H/W 切分，卷积边界用 halo exchange（P2P）交换，仅支持 decode（`vae_parallel.md:451-464`）。

**适用模块**：VAE（自编码器），既服务于 T2I/T2V 的 decode，也服务于 I2V 的 encode。

**实现位置**：`vllm_omni/diffusion/distributed/vae_patch_parallel.py`（`VaePatchParallelism` 类 `:348`，分布式 tiled decode 辅助函数 `_distributed_tiled_decode` `:71`、`_distributed_patch_decode` `:208`、包装函数 `maybe_wrap_vae_decode_with_patch_parallelism` `:446`）；spatial-shard 卷积变体在 `vllm_omni/diffusion/distributed/autoencoders/wan_spatial_shard.py`（`WanDistConv2d`、`WanDistCausalConv3d`、`WanDistZeroPad2d`）；模型侧分布式 VAE：`vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py`（Z-Image，仅 decode）、`autoencoder_kl_wan.py`（Wan2.2，decode+encode）、`autoencoder_kl_qwenimage.py`（Qwen-Image，仅 decode）；配置项 `vae_patch_parallel_size` / `vae_parallel_mode` 定义在 `vllm_omni/diffusion/data.py:184-200`。

#### 2.7 HSDP（Hybrid Sharded Data Parallel）

**是什么/为什么**：基于 PyTorch FSDP2 对模型权重做混合分片——在"复制组"内做 all-gather 按需取回权重、在"分片组"内切分参数存储，从而在显存受限的 GPU 上跑大模型（如 Wan2.2 14B）。与 TP 不同，HSDP 分片的是**权重存储**而非切分计算；**HSDP 不能与 TP 同时使用**（`docs/design/feature/hsdp.md:16-30`，同一约束在配置校验中体现）。

**适用模块**：DiT（Wan2.2 transformer 为参考实现）；理论上可扩展到任意有重复 block（`_hsdp_shard_conditions` 匹配）的 nn.Module。

**实现位置**：`apply_hsdp_to_model()`（`vllm_omni/diffusion/distributed/hsdp.py:88`）、`shard_model()`（`hsdp.py:222`）、`HSDPInferenceConfig`（`hsdp.py:30`）、mesh 构造 `_create_hsdp_mesh()`（`hsdp.py:45`）；HSDP 的 "fully-shard" 通信组在 `parallel_state.py:892-899`（`init_model_parallel_group(..., parallel_mode="fully_shard")` → `get_fs_group()`/`get_fully_shard_world_size()`/`get_fully_shard_rank()`，分别定义于 `parallel_state.py:371-383`）；配置字段 `use_hsdp` / `hsdp_shard_size` / `hsdp_replicate_size` 及 standalone-HSDP 世界大小推导逻辑在 `vllm_omni/diffusion/data.py:202-217, 245-309`。

### 3. 汇总对比表

| 策略 | 切分对象 | 适用模块（AR/DiT/VAE） | 通信模式 | 关键实现 file:line |
|---|---|---|---|---|
| Tensor Parallel (TP) | 单层权重矩阵（列/行切分：QKV、FFN、输出投影） | AR（vLLM 原生）+ DiT | Column→Row 配对的 all-reduce | `vllm_omni/diffusion/distributed/parallel_state.py:866-881`；模型示例 `vllm_omni/diffusion/models/z_image/z_image_transformer.py` |
| Pipeline Parallel (PP) | Transformer 层区间 `[start_layer, end_layer)` | DiT（Wan2.2） | 相邻 PP rank 间异步 P2P send/recv `IntermediateTensors` | `vllm_omni/diffusion/distributed/pipeline_parallel.py:75`；组建立 `parallel_state.py:826-834`；coordinator `vllm_omni/diffusion/distributed/group_coordinator.py:621` |
| Sequence Parallel (SP, Ulysses/Ring) | 序列维（patch/frame）切分 | DiT | Ulysses: all-to-all 交换 heads；Ring: 环形 P2P 传 KV | `vllm_omni/diffusion/distributed/sp_plan.py:206,253`；组建立 `parallel_state.py:544-598,836-852`；coordinator `group_coordinator.py:981` |
| Expert Parallel (EP) | MoE 专家网络（专家维度） | DiT（仅 MoE 模型） | EP 组内 All-to-All（token 派发/收集），依托 `allgather_reducescatter` | `vllm_omni/diffusion/data.py:155-156`；组建立 `parallel_state.py:900-910`；校验 `vllm_omni/config/composable_parallel/translator.py:300-309,358-368` |
| CFG-Parallel | CFG 分支（条件/无条件，或 N-branch） | DiT | 各 rank 独立前向 + `all_gather` 收集结果，本地合并 | `vllm_omni/diffusion/distributed/cfg_parallel.py:57`；组建立 `parallel_state.py:818-825` |
| VAE Parallel（tile / spatial-shard） | 空间 tile（含 overlap）或整张 feature map 沿 H/W | VAE（encode + decode） | tile: gather 到 rank0 + blend 拼接；spatial-shard: 逐卷积 halo exchange (P2P) + all-gather 拼接 | `vllm_omni/diffusion/distributed/vae_patch_parallel.py:348`；spatial-shard `vllm_omni/diffusion/distributed/autoencoders/wan_spatial_shard.py` |
| HSDP | 全模型权重（FSDP2 分片） | DiT（Wan2.2） | 分片组内按需 all-gather 权重（前向前），复制组内梯度/权重同步（推理场景主要是取权重） | `vllm_omni/diffusion/distributed/hsdp.py:88,222`；组建立 `parallel_state.py:892-899` |

补充：`initialize_model_parallel()` 中 `RankGenerator` 用固定轴序 `order="tp-sp-pp-cfg-dp"` 生成正交 rank 分组（`vllm_omni/diffusion/distributed/parallel_state.py:791-799`），DiT 总 world size 由 `get_dit_world_size()` 计算为 `dp * cfg * sp * pp * tp` 的乘积（`parallel_state.py:395-403`）。

### 4. 每个 stage 如何声明并行配置

vLLM-Omni 有**两条并行的配置面**（都要覆盖，理解上不要混淆）：

1. **`StageDeployConfig`**（`vllm_omni/config/stage_config.py:303-403`）——deploy YAML/legacy `stage_args` 里，每个 stage 直接声明的引擎参数。其中既有通用维度（`tensor_parallel_size: int | None`，`stage_config.py:329`；`enable_expert_parallel: bool | None`，`:330`），也有 DiT 专属维度整段列出（`ulysses_degree`/`ulysses_mode`/`ring_degree`/`sequence_parallel_size`/`cfg_parallel_size`/`vae_patch_parallel_size`/`vae_parallel_mode`/`use_hsdp`/`hsdp_shard_size`/`hsdp_replicate_size`，`stage_config.py:354-363`）。`data_parallel_size`/`pipeline_parallel_size` 则被声明为**pipeline-wide**（顶层 `DeployConfig` 字段，`stage_config.py:435-436`），对所有 stage 统一生效，而不是逐 stage 覆盖。
2. **`composable_parallel` strategy YAML**——按 `model_stage` 名（不是 `stage_id` 整数）声明一叠 `StrategySpec`（`docs/configuration/composable_parallel.md:12,198-202`），经 `translate_strategy_stack` 转成 `OmniParallelConfig`，再由 `_apply_to_stage`（`apply.py`）写回对应 stage 的引擎参数，**仅覆盖它声明的轴**，其余引擎参数保持不变（`composable_parallel.md:5`）。

二者的关系：composable_parallel 是叠加在 registry 合并后 stage 配置之上的一层"声明式增量补丁"；stage_config.py 的字段是最终真正落地到每个 stage 引擎（`OmniEngineArgs`/`DiffusionParallelConfig`）的字段名。composable_parallel 目前只打通 `tp/dp/pp/ep/stage_replica` 五种轴（`translator.py:50`），因此 SP/CFG/VAE/HSDP 这类 DiT 专属并行目前**只能通过 `StageDeployConfig` 的字段（或顶层 `DiffusionParallelConfig`）逐 stage 直接配置，不能通过 strategy YAML 声明**（声明会立即报 `AxisTranslationError`）。

DiT 侧真正消费的运行时结构是 `DiffusionParallelConfig`（`vllm_omni/diffusion/data.py:143-324`），它做了跨维度的一致性校验，例如：
- `sequence_parallel_size == ulysses_degree * ring_degree`（`data.py:237-240`）；
- `cfg_parallel_size ∈ {1,2,3}`（`data.py:229-231`）；
- HSDP 与 TP/DP 互斥，`hsdp_shard_size=-1` 时按 `other_parallel_world_size / hsdp_replicate_size` 自动推导（`data.py:269-307`）。

### 5. Config 旋钮 / 环境变量 / CLI 一览

| 旋钮 | 层级 | 默认值 | 说明 / 来源 |
|---|---|---|---|
| `--tensor-parallel-size` / `tensor_parallel_size` | CLI / `StageDeployConfig` / `DiffusionParallelConfig` | `1` | TP 度；`vllm_omni/diffusion/data.py:152-153`；per-stage `stage_config.py:329` |
| `--pipeline-parallel-size` / `pipeline_parallel_size` | CLI（pipeline-wide） | `1` | PP 度；`data.py:146-147`；顶层 `DeployConfig.pipeline_parallel_size`（`stage_config.py:436`） |
| `--ulysses-degree` / `ulysses_degree` | CLI / stage | `1` | Ulysses SP 子群大小；`arg_utils.py:460` |
| `--ring-degree` / `ring_degree` | CLI / stage | `1` | Ring SP 子群大小；`arg_utils.py:462` |
| `ulysses_mode` | stage / `DiffusionParallelConfig` | `"strict"` | `"strict"` 或实验性 `"advanced_uaa"`；`arg_utils.py:461`，`data.py:167-179` |
| `--cfg-parallel-size` / `cfg_parallel_size` | CLI / stage | `1` | 取值必须 `∈ {1,2,3}`；`arg_utils.py:484`，`data.py:181-182,229-231` |
| `--enable-expert-parallel` / `enable_expert_parallel` | CLI / stage | `False` | 稠密 EP 开关；`data.py:155-156`，`stage_config.py:330` |
| `--vae-patch-parallel-size` / `vae_patch_parallel_size` | CLI / stage | `1` | VAE tile/spatial-shard 并行度；`arg_utils.py:485`，`data.py:184-185` |
| `--vae-parallel-mode` / `vae_parallel_mode` | CLI / stage | `"tile"` | `"tile"` / `"spatial_shard_height"` / `"spatial_shard_width"`；`arg_utils.py:486`，`data.py:187-200` |
| `--use-hsdp` / `use_hsdp` | CLI / stage | `False` | 开启 HSDP；与 TP/DP>1 互斥；`arg_utils.py:464`，`data.py:202-203,270-276` |
| `--hsdp-shard-size` / `hsdp_shard_size` | CLI / stage | `-1`（自动） | `arg_utils.py:465`，`data.py:213-214` |
| `--hsdp-replicate-size` / `hsdp_replicate_size` | CLI / stage | `1` | `arg_utils.py:466`，`data.py:216-217` |
| `VLLM_PP_LAYER_PARTITION` | 环境变量 | 均衡分配 | 逗号分隔的每 PP rank 层数，覆盖 `get_pp_indices()` 默认切分；`docs/design/feature/pipeline_parallel.md:180-190` |
| `--strategy-config PATH` | CLI（composable_parallel） | 无（opt-in） | 加载 strategy YAML，仅在 registry-based deploy 路径生效；`arg_utils.py:439`，`docs/configuration/composable_parallel.md:36` |
| `--omni-lb-policy POLICY` | CLI（orchestrator） | 无 | 与 `stage_replica` 轴派生的 `omni_lb_policy` 必须一致，否则 `AsyncOmniEngine` 构造时抛 `Conflicting load-balancer policy`；`composable_parallel.md:38,161` |
| `--stage-configs-path PATH` | CLI（legacy） | 无 | 旧版 `stage_args` YAML schema；与 `--strategy-config` 同时使用时后者被静默忽略并告警；`composable_parallel.md:37`，`vllm_omni/entrypoints/utils.py:372-380` |
| `--deploy-config PATH` | CLI | 无（自动按 `model_type` 加载内置 YAML） | 新 schema deploy YAML；`docs/configuration/stage_configs.md:89` |
| `--stage-overrides JSON` | CLI | 无 | 逐 stage JSON 覆盖，优先级高于 strategy YAML；`stage_configs.md:90` |


---

## §7. vLLM-Omni 解耦推理、Omni Connector、Omni Coordinator 与 Ray 分布式执行

> 代码基线：`vllm-omni` tag `v0.25.0rc1`（commit `d3c47efc`）。本章所有结论均标注 `file:line`，代码路径均相对仓库根目录。

### 1. vLLM-Omni 中"解耦"（Disaggregation）的含义

vLLM-Omni 把一个多模态请求建模成一条 **DAG 式 stage 流水线**（`stage_id` 0..N），而不是单一 LLM 的 prefill/decode 循环。以 Qwen3-Omni-MoE 的默认部署为例（`vllm_omni/deploy/qwen3_omni_moe.yaml:26-69`）：

- `stage_id 0`：Thinker（AR，语言理解/comprehension，`devices: "0"`）
- `stage_id 1`：Talker（AR，生成 codec token，`devices: "1"`）
- `stage_id 2`：Code2Wav（vocoder，`devices: "1"`）

"解耦"在 vLLM-Omni 中体现为两个正交维度：

1. **跨 stage 解耦（pipeline disaggregation）**：不同模型阶段（Thinker/Talker/Code2Wav，或更一般的 encoder/AR/DiT/vocoder）被拆分到不同进程甚至不同物理节点上运行，stage 之间通过 `input_connectors` / `output_connectors` 声明式地连线（`docs/design/feature/disaggregated_inference.md:60-87`）。每条 edge 若未显式配置 connector，系统会 **自动回退到 `SharedMemoryConnector`**（`vllm_omni/distributed/omni_connectors/utils/initialization.py:344-360`），并且**缺失 edge 会 fail-fast**（同文件 `:353-360`），而不是静默降级。
2. **同一 stage 内部再解耦（PD disaggregation）**：把某个 AR stage（如 Qwen3-Omni 的 Thinker）进一步拆成 `is_prefill_only` 和 `is_decode_only` 两个子 stage，分别部署在不同 GPU 上，通过 vLLM 原生的 `kv_transfer_config`（`MooncakeConnector`）搬运 KV cache（`docs/configuration/pd_disaggregation.md:24-107`，详见第 6 节）。

**动态资源分配**体现在 `stage_args[].runtime.devices` 字段（每个 stage/子 stage 独立声明 GPU 集合，如 `"0,1,2,3"` vs `"4,5,6,7"`，见 `docs/design/feature/omni_connectors/yuanrong_transfer_engine_connector.md:144-171`）、`engine_args.tensor_parallel_size`（各 stage 可独立设置 TP）、以及 `gpu_memory_utilization`（Thinker 用 0.9，Code2Wav 只用 0.1，`vllm_omni/deploy/qwen3_omni_moe.yaml:29,55`）——即不同计算强度的 stage 可以按需分配不同数量/规格的加速卡，而不是所有 stage 均摊同一份显存池。

**为什么对 Omni 吞吐量重要**：Omni 模型的各 stage 计算特征差异极大——Thinker/Talker 是访存密集的自回归 decode，Code2Wav/DiT 类 vocoder 是算力密集的批量前向。把它们绑定在同一进程/同一批调度器里会导致互相阻塞（如强制 talker 等 thinker 的调度节拍）。解耦后，每个 stage 可以按自己的批处理策略（`max_num_seqs`、`enable_chunked_prefill`、`scheduler_cls`）独立扩缩容和调度，OmniConnector 负责把张量/token/隐藏状态跨进程/跨节点搬运，而不牺牲流水线的整体吞吐。

### 2. OmniConnector 抽象

#### 2.1 核心接口

`OmniConnectorBase`（`vllm_omni/distributed/omni_connectors/connectors/base.py:12-113`）定义了统一的 `put`/`get` 契约：

```python
# vllm_omni/distributed/omni_connectors/connectors/base.py:20-56
def put(self, from_stage, to_stage, put_key, data) -> tuple[bool, int, dict | None]
def get(self, from_stage, to_stage, get_key, metadata=None) -> tuple[Any, int] | None
```

其中 `supports_raw_data: bool`（`base.py:18`）是关键旗标：默认 `False`（走 `OmniSerializer` 全量序列化），只有 RDMA 类连接器（Mooncake TE、Mori TE）覆写为 `True`，以支持 `torch.Tensor`/`bytes` 的零拷贝快路径。`OmniConnectorFactory`（`vllm_omni/distributed/omni_connectors/factory.py:24-56`）按名字注册/构造连接器实例，`_make_key()`（`base.py:105-112`）提供默认的 `{key}@{from_stage}_{to_stage}` 寻址格式，部分连接器（如 Yuanrong）覆写为自定义 key 格式。

工厂在 `factory.py:129-137` 注册的连接器名：`MooncakeStoreConnector`、`MooncakeTransferEngineConnector`、`SharedMemoryConnector`、`YuanrongConnector`、`YuanrongTransferEngineConnector`、`MoriTransferEngineConnector`（以及历史别名 `MooncakeConnector` → `MooncakeStoreConnector`）。

#### 2.2 连接器一览表

| Connector | File:line | 数据面 Transport | 控制面 | Fast-path (`supports_raw_data`) | 典型场景 |
|---|---|---|---|---|---|
| `SharedMemoryConnector` | `vllm_omni/distributed/omni_connectors/connectors/shm_connector.py:17` | POSIX `/dev/shm`（`shm_write_bytes`/`shm_read_bytes`），`fcntl.flock` 互斥 | 队列携带 `{shm:{name,size}}` 元数据 | 否，始终全量序列化 | 单机多进程默认连接器，无 edge 显式配置时自动生成 |
| `MooncakeStoreConnector` | `vllm_omni/distributed/omni_connectors/connectors/mooncake_store_connector.py:22` | Mooncake 分布式 KV 存储（TCP，可选 RDMA） | `store.put/get`，key 即 rendezvous 点，无额外 metadata | 否 | 多节点，运维简单，对性能要求不极致 |
| `MooncakeTransferEngineConnector` | `vllm_omni/distributed/omni_connectors/connectors/mooncake_transfer_engine_connector.py:70`（`put` `:362`，`get` `:611`） | Mooncake `TransferEngine.batch_transfer_sync_write`，RDMA(InfiniBand/RoCE) 或 TCP，托管内存池（CPU pinned 或 GPU/GPUDirect） | ZMQ 边带信道（握手/pull 请求/完成信号） | 是（`Tensor`/`bytes`/`ManagedBuffer`） | 多节点、大 payload（KV cache、隐藏状态）、要求最高吞吐，比 Store 版快约 58-60x |
| `MoriTransferEngineConnector` | `vllm_omni/distributed/omni_connectors/connectors/mori_transfer_engine_connector.py:110`（`put` `:379`，`get` `:527`） | AMD Mori `IOEngine`/`MemoryDesc`，RDMA(InfiniBand/RoCE) 或 XGMI(AMD Infinity Fabric)，托管内存池 | ZMQ 边带信道 | 是 | AMD GPU（MI300X）节点内零拷贝 GPU-to-GPU；跨节点尚未支持（见 `#1742`） |
| `YuanrongConnector` | `vllm_omni/distributed/omni_connectors/connectors/yuanrong_connector.py:19`（`put` `:62`，`get` `:89`） | Yuanrong Datasystem `KVClient`（TCP/RDMA） | etcd + Datasystem worker，key 为 `{request_id}:{from_stage}_{to_stage}` | 否 | 已有 Yuanrong Datasystem 基础设施的多节点部署 |
| `YuanrongTransferEngineConnector` | `vllm_omni/platforms/npu/omni_connectors/`（NPU 平台专属，经 `factory.py:94-103` 动态导入） | Yuanrong TransferEngine 直连，`protocol: "ascend"`，NPU 内存池（HCCN P2P） | ZMQ | 是 | 昇腾 NPU 间高性能 P2P KV cache 传输（仅支持 `memory_pool_device: "npu"`） |

补充说明：

- **D2H2D 现状**：目前所有连接器都工作在 D2H2D（Device→Host→Device）模式；`docs/design/feature/disaggregated_inference.md:108-112` 明确把纯 D2D（NCCL/UCX/IPC）列为未来路线图，当前尚未实现。
- **与 vLLM 原生分布式机制的关系**（`docs/design/feature/disaggregated_inference.md:89-101`）：vLLM 自身的 `vllm.distributed.kv_transfer`（KV cache 专用）、`vllm.distributed.ec_transfer`（encoder embedding 专用）、`vllm.distributed.device_communicators`（NCCL/SHM 底层原语）针对特定 artifact 做了优化；OmniConnector 则提供一个**统一的 `put`/`get` 抽象**，覆盖任意 stage 间 artifact（token、隐藏状态、codec chunk 等），并可以在需要时把 vLLM 的 KV 路径包装进同一接口（见 `OmniKVTransferManager`，`vllm_omni/distributed/omni_connectors/kv_transfer_manager.py:341`，它使用 `TRANSFER_ENGINE_CONNECTOR_NAMES`= `{MooncakeTransferEngineConnector, MoriTransferEngineConnector, YuanrongTransferEngineConnector}`，`vllm_omni/distributed/omni_connectors/utils/config.py:11-17`，作为 KV 传输可用的高性能后端集合）。
- **RDMA 内存池实现**：`MooncakeTransferEngineConnector`/`MoriTransferEngineConnector` 共用的内存管理原语是 `BufferAllocator`（`vllm_omni/distributed/omni_connectors/utils/memory_pool.py:13`，对齐子区间分配 + 相邻块合并）与 `ManagedBuffer`（`memory_pool.py:86`，零拷贝 1D `uint8` tensor 视图，支持 `.as_tensor()`/`.to_bytes()`/`.release()`）。
- **发送/接收调用点**：orchestrator 通过 `try_send_via_connector()`/`try_recv_via_connector()`（`vllm_omni/distributed/omni_connectors/adapter.py:14-101` / `:104-183`）把 `connector.put()` 的返回值（`success, serialized_size, metadata`）打包进队列通知（`connector_metadata` 字段），下游 stage 再用该 metadata 调用 `connector.get()`——这就是"轻量控制面 + 独立数据面"的通用模式。

### 3. Omni Coordinator：解耦 stage 的编排角色

Omni Coordinator 不是"数据搬运工"，而是**多副本注册中心 + 心跳健康检测器 + 负载均衡输入源**，为动态扩缩容的解耦 stage 提供成员管理（membership）。

- **核心服务** `OmniCoordinator`（`vllm_omni/distributed/omni_coordinator/omni_coordinator.py:19-370`）：
  - 用 ZMQ `ROUTER` socket（`:50-53`）接收 stage 副本上报的 `ReplicaEvent`（`update`/`heartbeat`，`vllm_omni/distributed/omni_coordinator/messages.py:19-32`）；
  - 用 ZMQ `PUB` socket（`:56-59`）向 hub（orchestrator 端）广播当前活跃副本列表 `ReplicaList`（`messages.py:51-61`）；
  - 后台线程 `_periodic_loop`（`omni_coordinator.py:244-286`）周期性检测心跳超时（`_check_heartbeat_timeouts`，`:143-163`，默认 `heartbeat_timeout=30.0`），把超时副本标记为 `ReplicaStatus.ERROR`；10 分钟（`gc_ttl=600.0`，`:147`）后彻底从注册表移除。
  - 广播采用 **合并 + 限速**（`_publish_min_interval=0.1s`，`:65`），避免高频状态变化打爆 PUB 信道。

- **生命周期封装** `OmniCoordinatorRuntime`（`vllm_omni/distributed/omni_coordinator/runtime.py:84-159`）：把 `OmniCoordinator` 启动为**独立子进程**（物理隔离，避免 GIL 争用），偏好 `fork` 上下文（`_get_coordinator_mp_context`，`runtime.py:54-71`，理由是 `spawn` 需要重新 import 整个 CLI/模型栈，可能超过 30s 启动超时）。它被 `DistStageRuntime._start_omni_master_server()` 实例化（`vllm_omni/engine/stage_runtime.py:889-912`），与 `OmniMasterServer` 配对：coordinator 的 ROUTER 地址交给 `OmniMasterServer` 转发给注册中的副本，PUB 地址交给 `MembershipController` 构造 `OmniCoordClientForHub`。

- **两个客户端**：
  - `OmniCoordClientForStage`（`vllm_omni/distributed/omni_coordinator/omni_coord_client_for_stage.py:19-259`）：stage 副本侧，DEALER socket 连接 coordinator ROUTER，构造时立即发一次 `update`（`:60`），随后每 5s 一次心跳（`_heartbeat_interval=5.0`，`:51`），带自动重连（`_reconnect`，`:68-107`，最多 3 次、间隔 5s）。心跳可挂载 `_on_heartbeat` 钩子实时刷新 `queue_length`（`create_stage_coord_client`，`:235-259`），这是负载均衡策略拿到"当前排队长度"的唯一实时来源。
  - `OmniCoordClientForHub`（`vllm_omni/distributed/omni_coordinator/omni_coord_client_for_hub.py:17-165`）：orchestrator 侧，SUB socket 订阅 coordinator PUB，缓存最新 `ReplicaList`，暴露 `get_replicas_for_stage(stage_id)`（`:151-155`）供负载均衡器筛选某个 stage 的可用副本。

- **负载均衡策略**（`vllm_omni/distributed/omni_coordinator/load_balancer.py:27-131`）：`LoadBalancingPolicy` 枚举 `RANDOM`/`ROUND_ROBIN`/`LEAST_QUEUE_LENGTH`，对应 `RandomBalancer`（`:64-71`）、`RoundRobinBalancer`（`:74-99`）、`LeastQueueLengthBalancer`（`:102-121`，选 `queue_length` 最小的副本，多个并列时随机打散）。通过 `--omni-lb-policy` 配置，Coordinator 提供的实时 `queue_length` 是 `LEAST_QUEUE_LENGTH` 策略生效的前提。

一句话总结：**Omni Coordinator 解决"解耦之后 stage 副本在哪、活不活、忙不忙"的问题**，是 OmniConnector（解决"数据怎么搬"）之外的编排层，二者共同支撑一个可以动态扩缩容的解耦流水线。

### 4. Ray-based 执行 vs. 默认多进程执行

vLLM-Omni 默认用 **`multiprocessing`**（`worker_backend: str = "multi_process"`，`vllm_omni/config/omni_config.py:807`、`vllm_omni/engine/arg_utils.py:430`）在单机上为每个 stage 拉起独立进程；Ray 是可选的多节点后端。

- **命令行开关**：`--worker-backend ray --ray-address auto`（`docs/design/feature/ray_based_execution.md:19-25`），对应 `ray_address: str | None = None`（`omni_config.py:808`、`arg_utils.py:431`，CLI flag `--ray-address` 在 `vllm_omni/entrypoints/cli/serve.py:338`）。**headless 模式（`--stage-id` 单 stage 启动）显式禁止 `worker_backend != "multi_process"`**（`serve.py:798-799: raise ValueError("headless mode requires worker_backend=multi_process")`），说明 Ray 后端目前只在"orchestrator 统一管理"路径下生效，而非每 stage 独立进程自举路径。

- **`vllm_omni/distributed/ray_utils/utils.py`（201 行）** 提供的能力，按"已接入调用点" vs "已定义但未见调用点"分两类：
  - **已接入**（在模型 runner 中真实调用）：
    - `is_ray_initialized()`（`utils.py:28-38`）：优先 `ray.is_initialized()`，否则回退检测 `RAY_RAYLET_PID` 环境变量（Ray worker 进程必设）。
    - `maybe_disable_pin_memory_for_ray()`（`utils.py:57-87`，contextmanager）：Ray worker 通常 `ulimit -l`（locked memory）较低，分配大块 pinned host memory 会失败；该函数在 Ray 环境下、分配超过 32MB 阈值且对象已开启 `pin_memory` 时临时关闭之。调用点：`vllm_omni/worker/gpu_ar_model_runner.py:334-342`、`vllm_omni/platforms/npu/worker/npu_ar_model_runner.py:119-127`。
    - **SHM 阈值联动**：`initialize_orchestrator_connectors(worker_backend="ray", ...)`（`vllm_omni/distributed/omni_connectors/utils/initialization.py:371-392`）在 `worker_backend == "ray"` 时把 `SharedMemoryConnector` 默认阈值设为 `sys.maxsize`（`:382`），强制小 payload 走 inline 而非 SHM——因为跨 Ray 节点的 stage 不共享 `/dev/shm`。这与 `docs/design/feature/ray_based_execution.md:50` 描述一致。
  - **已定义、当前仓库内未见调用点**（`create_placement_group`/`start_ray_actor`/`initialize_ray_cluster`/`kill_ray_actor`/`is_ray_task_alive`/`get_ray_task_error`/`get_ray_queue_class`，均在 `utils.py:93-201`）：
    - `initialize_ray_cluster(address)`（`utils.py:99-107`）：`ray.init(address=..., runtime_env={"env_vars": {"PYTHONPATH": ...}})`，用于连接/启动 Ray 集群并透传 `PYTHONPATH` 让 worker 能 import `vllm_omni`。
    - `create_placement_group(number_of_stages, address, strategy="PACK")`（`utils.py:110-130`）：为每个 stage 声明 `{"GPU": 1.0, "CPU": 1.0}` bundle，调用 `ray.util.placement_group(bundles, strategy)` 并阻塞等待 `pg.ready()`——这是**多节点 stage 放置**的调度单元。
    - `start_ray_actor(worker_entry_fn, placement_group, bundle_index, ...)`（`utils.py:157-181`）：定义 `@ray.remote(num_gpus=1) class OmniStageRayWorker`，用 `PlacementGroupSchedulingStrategy` 把 actor 绑定到指定 bundle，`.run.remote(worker_entry_fn, ...)` 执行 stage 的 worker 入口函数。
    - `is_ray_task_alive`/`get_ray_task_error`（`utils.py:184-201`）：轮询 Ray task 存活状态/取出 `RayTaskError`，用于监控远程 stage worker 的健康状况。
    - `__init__.py` 只导出 `calculate_total_bytes, is_ray_initialized, maybe_disable_pin_memory_for_ray`（`vllm_omni/distributed/ray_utils/__init__.py:4-9`），placement-group/actor 相关函数不在公开 `__all__` 中。

  **结论**：在 v0.25.0rc1 这个 tag，"Ray 作为多节点 stage 放置后端"的 **placement-group + actor 调度原语已经写好**（`create_placement_group`/`start_ray_actor`），但连接进 `DistStageRuntime`/orchestrator 主流程的粘合代码在本文档搜索范围内未找到调用点；已经稳定生效并被模型 runner 消费的，是 pinned-memory 兼容性处理和 SHM 阈值调整这两个"Ray 环境适配"层面的能力。文档（`docs/design/feature/ray_based_execution.md`）描述的"Ray 集群自动适配传输策略"（跨节点用 Mooncake TE/Store，同节点仍可用 SHM 或 Ray plasma）目前主要通过第 4 点的 SHM 阈值联动体现，而非专门的传输路由代码。

- **多进程执行对照**：默认路径下，`DistStageRuntime`（`vllm_omni/engine/stage_runtime.py:723-731`）用标准 `multiprocessing` 为每个 replica 启动 stage 进程（`launch_mode` 为 `"local"` 或 `"remote"`，`stage_runtime.py:360,397,425,505,509,521`），配合 `OmniMasterServer`（进程内 TCP server，负责远程 replica 注册）与 `OmniCoordinatorRuntime`（第 3 节）做成员管理；这是单机/多机通过手动 SSH+进程管理的"轻量"路径，Ray 则是把 stage 放置和进程生命周期都交给 Ray 调度器管理的"重量"路径。

### 5. 解耦式 Omni 部署 ASCII 示意图

以 Qwen3-Omni（AR Thinker → AR Talker → Vocoder/Code2Wav）跨节点部署 + PD 解耦为例：

```
┌───────────────────────────── Node A (prefill) ─────────────────────────────┐
│  Stage 0: Thinker (prefill-only)                                           │
│  is_prefill_only=true, worker_type=ar, tp=1..N, devices="0"                │
│  kv_transfer_config: kv_connector=MooncakeConnector, kv_role=kv_producer   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                     │  KV cache (MooncakeConnector / RDMA)
                                     ▼
┌───────────────────────────── Node B (decode) ──────────────────────────────┐
│  Stage 1: Thinker (decode-only)                                           │
│  is_decode_only=true, kv_role=kv_consumer, engine_output_type=latent      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                     │  OmniConnector.put/get(engine_inputs)
                                     │  transport = MooncakeTransferEngineConnector
                                     │  (RDMA data-plane + ZMQ control-plane)
                                     ▼
┌───────────────────────────── Node C (AR / Talker) ──────────────────────────┐
│  Stage 2: Talker (AR, codec-token generation)                              │
│  sync_process_input_func = thinker2talker_token_only                      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                     │  OmniConnector.put/get(codec tokens)
                                     │  transport = MoriTransferEngineConnector
                                     │  (intra-node RDMA/XGMI) or SharedMemoryConnector
                                     ▼
┌───────────────────────────── Node C/D (DiT / Vocoder) ──────────────────────┐
│  Stage 3: Code2Wav / DiT-style vocoder (final_output=true)                 │
│  sync_process_input_func = talker2code2wav_token_only                     │
└──────────────────────────────────────────────────────────────────────────────┘

           ▲                              ▲                              ▲
           │ ReplicaEvent(update/heartbeat│ over ZMQ ROUTER/DEALER       │
           └──────────────┬───────────────┴──────────────┬───────────────┘
                           │                              │
                 ┌─────────────────────┐         ┌─────────────────────┐
                 │   OmniCoordinator   │────────▶│ OmniCoordClientForHub│
                 │ (独立子进程, ROUTER  │  PUB    │  (orchestrator 端,   │
                 │  + PUB socket)       │  广播   │  缓存 ReplicaList)   │
                 └─────────────────────┘         └──────────┬──────────┘
                                                             │ get_replicas_for_stage()
                                                             ▼
                                                   ┌─────────────────────┐
                                                   │    LoadBalancer      │
                                                   │ (Random/RoundRobin/  │
                                                   │  LeastQueueLength)   │
                                                   └─────────────────────┘
```

（图中的 stage 编号对应 `docs/configuration/pd_disaggregation.md:26-155` 的四段式流水线：0=prefill thinker, 1=decode thinker, 2=talker, 3=code2wav；connector 选择依据 `docs/design/feature/disaggregated_inference.md:20-29` 的 Use-Case 表。）

### 6. PD（Prefill/Decode）解耦细节与配置

针对 Qwen3-Omni Thinker 的 PD 拆分（`docs/configuration/pd_disaggregation.md`）：

- **拆分方式**：把原单一 Thinker stage 替换为两个 stage：
  - Prefill stage（`stage_id: 0`）：`is_prefill_only: true`（`pd_disaggregation.md:31`），`kv_transfer_config.kv_role: "kv_producer"`（`:50-56`）。文档明确指出 **orchestrator 会强制该 stage `max_tokens=1`**（`:114-115`），即只跑一次 prefill 导出 KV，不做自回归 decode。
  - Decode stage（`stage_id: 1`）：`is_decode_only: true`（`:70`），`kv_role: "kv_consumer"`（`:89-91`），`engine_input_source: [0]`（`:95`）从 prefill stage 消费。
- **KV 传输后端**：两侧都必须配置 `kv_transfer_config`（`:113`），示例用 `kv_connector: "MooncakeConnector"`（vLLM 原生 KV connector 命名空间，不同于 OmniConnector 工厂里的 `MooncakeConnector` 别名，二者同名但作用域不同——一个是 vLLM `kv_transfer` 子系统的 connector，一个是 OmniConnector 工厂别名）；`kv_parallel_size`/`kv_rank` 标识 producer/consumer 编号，`mooncake_bootstrap_port` 指定握手端口（`:50-56`, `:88-94`）。
- **下游 stage 重编号**：插入 prefill/decode 两段后，Talker 从 `stage_id 1` 变为 `2`，Code2Wav 从 `2` 变为 `3`（`:117-141`），并显式声明 `runtime.edges`（`:145-155`）。
- **约束条件**（`:164-171`）：
  - `MooncakeConnector` **不支持** prefill/decode 之间的异构 TP（heterogeneous TP），二者 `tensor_parallel_size` 必须一致；
  - 若 Thinker 需要 TP=2，prefill 和 decode 都要 TP=2，且分配不同 GPU 集合（如 `"0,1"` vs `"2,3"`）；
  - 至少需要 3 张 GPU（prefill + decode + talker/code2wav）。
- **异构 TP 的例外**：在 KV cache 直接传输（非 vLLM `kv_transfer_config`，而是 OmniConnector 层的 `omni_kv_config`）场景下，`MooncakeTransferEngineConnector`/`YuanrongTransferEngineConnector` 通过"1 receiver → N senders"模式支持异构 TP（sender TP > receiver TP），receiver 侧用 `metadata` 中的 `source_host`/`source_port` 主动向多个 sender rank 拉取分片（`docs/design/feature/omni_connectors/mooncake_transfer_engine_connector.md:727-729` 描述该限制，`yuanrong_transfer_engine_connector.md:138-180` 给出 4 卡 AR→DiT TP 对齐的实例，`rank_mapping: {from_tp: 4, to_tp: 4}`）。

### 7. 配置项与环境变量速查

| 配置/环境变量 | 位置 | 作用 | 默认值 |
|---|---|---|---|
| `worker_backend` | `vllm_omni/config/omni_config.py:807`、`vllm_omni/engine/arg_utils.py:430` | `"multi_process"` / `"ray"` | `"multi_process"` |
| `ray_address` | `omni_config.py:808`、`arg_utils.py:431`；CLI `--ray-address`（`entrypoints/cli/serve.py:338`） | Ray 集群地址（如 `"auto"`） | `None` |
| `shm_threshold_bytes` | `vllm_omni/distributed/omni_connectors/connectors/shm_connector.py:31`；orchestrator 层 `omni_config.py` `VllmOmniOrchestratorConfig.shm_threshold_bytes` | SHM 与 inline 阈值（字节）；`worker_backend="ray"` 时被强制设为 `sys.maxsize`（`initialization.py:381-384`） | `65536` |
| `inline_small_payloads` | `shm_connector.py:32` | 是否允许小 payload inline（跳过 SHM+锁开销） | `False` |
| `stage_init_timeout` / `init_timeout` | `omni_config.py:VllmOmniOrchestratorConfig` | stage 启动/整体初始化超时 | `300` / `600` |
| `omni_heartbeat_timeout` | `omni_config.py`；`OmniCoordinator(heartbeat_timeout=...)`（`omni_coordinator.py:36,46`） | 副本心跳超时判定 | `30.0` |
| `omni_lb_policy` | `omni_config.py`；`load_balancer.py:27-36` | `random` / `round-robin` / `least-queue-length` | `"random"` |
| `zmq_port` | 各 RDMA connector `extra.zmq_port`（如 `mooncake_transfer_engine_connector.md:33`） | ZMQ 边带信道基础端口，实际端口 = `base + purpose_offset(0/100) + stage_offset + dp_index*tp_size + tp_rank`（`+200` 若调用方是 orchestrator） | `50051` |
| `memory_pool_device` / `memory_pool_size` | RDMA connector `extra` | `"cpu"`(pinned)/`"cuda"`(GPUDirect)/`"npu"`(Ascend) 及池大小 | `"cpu"` / 4GB(CPU)/2GB(GPU) |
| `RDMA_DEVICE_NAME` | 环境变量，`mooncake_transfer_engine_connector.md:123` | 覆盖 RDMA 设备名（如 `mlx5_0`） | 未设置 |
| `MC_IB_PCI_RELAXED_ORDERING` | 环境变量 | GPUDirect PCIe relaxed ordering 开关 | 未设置 |
| `kv_connector` / `kv_role` / `kv_rank` / `kv_parallel_size` | stage `engine_args.kv_transfer_config`（`pd_disaggregation.md:50-56`） | PD 解耦的 vLLM 原生 KV 传输配置 | 无 |
| `is_prefill_only` / `is_decode_only` | stage config（`pd_disaggregation.md:31,70`） | 标记 PD 拆分的两侧 | `False` |
| `TRANSFER_ENGINE_CONNECTOR_NAMES` | `vllm_omni/distributed/omni_connectors/utils/config.py:11-17` | 判定某 connector 是否属于"高性能传输引擎类"（影响 KV 传输后端选择逻辑） | 固定集合，非可配置项 |


---

## §8. vLLM-Omni 的 Entrypoints、OpenAI 兼容 Serving API 与流式输出

> 基于标签 `v0.25.0rc1`，代码路径以仓库根为准（本次分析检出在 `/private/tmp/.../omni-v0250rc1`）。所有结论均标注 `file:line`。

### 1. Entrypoint 架构：从 upstream `api_server` 复用到异构输出扩展

vLLM-Omni **不是重写**一个独立的 API Server，而是在 upstream vLLM 的 `build_app` / `setup_server` 基础上做“覆盖 + 追加路由”的**寄生式扩展**：

- `vllm_omni/entrypoints/openai/api_server.py:38-39` 直接 `from vllm.entrypoints.openai.api_server import build_app as build_openai_app` 与 `setup_server as setup_openai_server`，复用 upstream 的 FastAPI app 构建、CORS/lifespan/中间件、`/v1/completions`、`/v1/embeddings` 等纯文本端点。
- `omni_run_server_worker`（`vllm_omni/entrypoints/openai/api_server.py:460-592`）是真正的组装点：
  1. `build_async_omni(args, client_config=...)`（`api_server.py:596-632`）构造 `AsyncOmni` 引擎客户端（多阶段流水线的统一入口，见第 5 节）；
  2. `app = build_openai_app(args, supported_tasks)`（`api_server.py:500`）拿到 upstream 的 app；
  3. `_remove_route_from_app(app, "/v1/chat/completions", {"POST"})` 与 `_remove_route_from_app(app, "/v1/models", {"GET"})`（`api_server.py:503-504`）**摘掉** upstream 的纯文本 handler；
  4. `app.include_router(router)`（`api_server.py:505`）挂载 Omni 自己的 `router`（`api_server.py:154`），其中同名路径（`/v1/chat/completions`、`/v1/audio/speech`、`/health`、`/v1/models`）用 `_remove_route_from_router`（`api_server.py:238`）在**模块加载期**再摘一遍，确保 Omni 版本优先；
  5. `omni_init_app_state(engine_client, app.state, args)`（`api_server.py:511`，实现于 `705-1139` 附近）向 `app.state` 注入 `openai_serving_chat` / `openai_serving_speech` / `openai_serving_audio_generate` / `openai_serving_video` 等异构 serving 实例；
  6. `shutdown_unsupported_routes(app, engine_client.endpoint_restrictions)`（`api_server.py:514-515`，实现于 `vllm_omni/config/endpoint_policy.py:63-80`）按模型能力**按需摘除**不支持的端点（如批量 chat completions），返回 400 而非 500。

这与 upstream vLLM 的关键差异在于：upstream `api_server` 面向单一输出模态（token 文本流），一个 `engine_client: AsyncLLM` 对应一种 `RequestOutput`；vLLM-Omni 把 `engine_client` 换成 `AsyncOmni`（`vllm_omni/entrypoints/async_omni.py:110`），它编排一条由多个「stage」（AR 文本 stage、扩散图像/视频 stage、TTS stage 等）组成的流水线，一次 `generate()` 调用可以在不同阶段产出文本 / 图像 / 音频 / 视频等异构 `OmniRequestOutput`（见 `vllm_omni/outputs.py` 中 `OmniRequestOutput`）。因此 Entrypoint 层必须：

- 为每种输出模态单独定义 serving 类（`OmniOpenAIServingChat`、`OmniOpenAIServingSpeech`、`OmniOpenAIServingAudioGenerate`、`OmniOpenAIServingVideo`），而不是像 upstream 那样只有 `OpenAIServingChat`/`OpenAIServingCompletion`；
- 为**纯扩散模型**（单一 diffusion stage，无 AR 阶段）提供一套简化初始化分支：`omni_init_app_state` 中 `is_pure_diffusion` 判断（`api_server.py:726-733`），走 `OmniOpenAIServingChat.for_diffusion(...)`（`api_server.py:764-767`）等 `for_diffusion` 工厂方法，跳过 tokenizer/tool-parser 等 LLM 专属初始化；
- 用一个 `_TimestampMiddleware`（纯 ASGI 外层包装，`api_server.py:544-562`）给每个 HTTP 请求打上 `request_timestamp`，供各 serving 类计算端到端时延（跨 stage 的排队/推理时间）。

### 2. 端点清单

| Endpoint | HTTP 路由 | Handler 类 / 函数 File:line | 输出模态 | 是否流式 |
|---|---|---|---|---|
| Chat Completions | `POST /v1/chat/completions` | `create_chat_completion` `api_server.py:1160` → `OmniOpenAIServingChat.create_chat_completion` `serving_chat.py:287` | 文本 / 图像 / 音频（按 `modalities` 参数混合） | 是（SSE `text/event-stream`），见 `api_server.py:1220`、`serving_chat.py:1192` |
| Speech (TTS) | `POST /v1/audio/speech` | `create_speech` `api_server.py:1238` → `OmniOpenAIServingSpeech.create_speech` `serving_speech.py:4010` | 音频（wav/pcm/…） | 支持三种模式：非流式 `Response`（`serving_speech.py:4111`）、raw chunked audio `StreamingResponse`（`serving_speech.py:4070-4079`）、SSE `speech.audio.*` 事件（`serving_speech.py:4090-4101`） |
| Speech Batch | `POST /v1/audio/speech/batch` | `create_speech_batch` `api_server.py:1290` → `OmniOpenAIServingSpeech.create_speech_batch` `serving_speech.py:4184` | 音频（批量 base64） | 否 |
| Speech Streaming（文本增量输入） | `WS /v1/audio/speech/stream` | `streaming_speech` `api_server.py:1557` → `OmniStreamingSpeechHandler.handle_session` `serving_speech_stream.py:81` | 音频（WAV/PCM 二进制帧或 base64 JSON） | 是（WebSocket 双工） |
| Audio Generate | `POST /v1/audio/generate` | `create_audio_generate` `api_server.py:1333` → `OmniOpenAIServingAudioGenerate.create_audio_generate` `serving_audio_generate.py:40` | 通用音频（音效/音乐，扩散模型） | 否（`Response`，`serving_audio_generate.py:159`） |
| Voice 管理 | `GET/POST/DELETE /v1/audio/voices[/{name}]` | `list_voices` `api_server.py:1368`、`upload_voice` `api_server.py:1422`、`delete_voice` `api_server.py:1510` | 元数据（JSON） | 否 |
| Image Generations | `POST /v1/images/generations` | `generate_images` `api_server.py:1710` | 图像（base64 PNG，或按 `response_format=file` 返回文件流） | 单阶段扩散：否；多阶段（AR+DiT）经 `openai_serving_chat.generate_diffusion_images` `serving_chat.py:2988` 内部仍非 SSE，返回聚合结果 |
| Image Edits | `POST /v1/images/edits` | `edit_images` `api_server.py:1935` | 图像（base64 PNG） | 仅多阶段（AR+DiT，如 HunyuanImage3 IT2I）支持 `stream=true` → SSE，单阶段模型 `stream=true` 直接 400（`api_server.py:1992-1996`） |
| Videos（异步任务） | `POST /v1/videos` | `create_video` `api_server.py:3044` → `_run_video_generation_job` `api_server.py:2796` | 视频（MP4 文件，异步落盘） | 否（任务式，需轮询） |
| Videos Sync | `POST /v1/videos/sync` | `create_video_sync` `api_server.py:3087` → `OmniOpenAIServingVideo.generate_video_bytes` `serving_video.py:296` | 视频（原始 MP4 字节） | 否（阻塞到完成，`asyncio.wait_for` 超时 `VIDEO_SYNC_TIMEOUT_S`） |
| Videos 查询/下载/删除 | `GET /v1/videos`、`GET /v1/videos/{id}`、`GET /v1/videos/{id}/content`、`DELETE /v1/videos/{id}` | `list_videos` `api_server.py:3158`、`retrieve_video` `api_server.py:3195`、`download_video` `api_server.py:3275`、`delete_video` `api_server.py:3223` | 元数据 / 文件流 | 否（`FileResponse`，`api_server.py:3305`） |
| Video Output Streaming | `WS /v1/realtime/video` | `streaming_video_output` `api_server.py:1595` → `OmniStreamingVideoOutputHandler.handle_session` `serving_video_output_stream.py:85` | 视频（fragmented MP4 二进制帧） | 是（WebSocket，逐 chunk 推送 `m4s` 分片） |
| Video Chat Streaming（视频输入理解） | `WS /v1/video/chat/stream` | `streaming_video_chat` `api_server.py:1578` → `create_streaming_video_handler` `serving_video_stream.py` | 文本 / 音频 | 是（WebSocket，见 `docs/serving/video_stream_api.md:36-51`） |
| Realtime（通用） | `WS /v1/realtime` | `realtime_websocket` `api_server.py:1612` → `RealtimeConnection` `realtime_connection.py` | 文本/音频（OpenAI Realtime 风格） | 是 |
| Robot OpenPI Realtime | `WS /v1/realtime/robot/openpi` | `realtime_robot_openpi` `api_server.py:1625` → `RobotRealtimeConnection` | 机器人动作序列 | 是 |
| Sleep/Wakeup | `POST /v1/omni/sleep`、`POST /v1/omni/wakeup` | `omni_sleep` `api_server.py:3376`、`omni_wakeup` `api_server.py:3387` | 控制面（无内容输出） | 否 |
| Health/Models | `GET /health`、`GET /v1/models` | `health` `api_server.py:1650`、`show_available_models` `api_server.py:1681` | 元数据 | 否 |

### 3. 异构 / 多模态输出的序列化方式

vLLM-Omni **没有**统一的“对象存储 + URL 回传”方案，不同模态各自选择最合适的编码：

- **图像**：几乎全部走 **base64 PNG（或指定格式）内联在 JSON**。`generate_images` 用 `encode_image_base64` / `encode_image_base64_with_compression`（`image_api_utils.py`，调用点 `api_server.py:1799`、`1893`）把 PIL Image 编码进 `ImageData.b64_json` 字段（`protocol/images.py:172`）。若客户端显式要求 `response_format=ResponseFormat.FILE`，则调用 `ImageGenerationResponse.stream_response()`（`protocol/images.py:194-226`）：单张图 `base64.b64decode` 后包成 `StreamingResponse(io.BytesIO(...), media_type="image/png")` 并带 `Content-Disposition: attachment`；多张图打包成 zip（`zipfile.ZipFile`）后同样以 `StreamingResponse` 返回。Chat Completions 里的图像 delta 走同样的 base64 编码（`serving_chat.py:2660-2668` 组装 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`）。
- **音频**：非流式场景走 **`Response(content=audio_bytes, media_type=...)` 原始二进制**（wav/pcm/mp3/…），例如 `serving_speech.py:4111`、`serving_audio_generate.py:159`；Chat Completions 的音频 choice 用 `CreateAudio(..., base64_encode=True)` → `OpenAIChatCompletionAudio(data=audio_base64, ...)`（`serving_chat.py:2548-2571`），对齐 OpenAI `chat.completion.audio` schema。批量语音接口 `create_speech_batch` 则把每条结果 base64 编码后放入 JSON 数组（`serving_speech.py:4218`, `api_server.py:1312` 用 `exclude_none=True` 裁剪失败/成功各自缺失的字段）。
- **视频**：**不走 base64**，而是**文件引用 + 落盘存储**。`POST /v1/videos` 创建任务记录并 `asyncio.create_task` 后台生成（`api_server.py:3063-3074`），生成完成后经 `STORAGE_MANAGER`（`vllm_omni/entrypoints/openai/storage.py:59` `LocalStorageManager` / `210` 全局单例）把 MP4 写入 `VLLM_OMNI_SERVER_STORAGE__PATH` 指定目录，`download_video`（`api_server.py:3299-3305`）用 `FileResponse(path=file_handle.path, media_type=job.media_type)` 流式返回文件；`/v1/videos/sync` 则直接把生成的 `video_bytes` 通过 `Response(content=video_bytes, media_type="video/mp4")`（`api_server.py:3145-3155`）一次性返回，并把 `stage_durations` / `peak_memory_mb` 塞进自定义响应头（`X-Stage-Durations`、`X-Peak-Memory-MB`）而非 body。
- **WebSocket 二进制帧**：语音流式（`serving_speech_stream.py`）与视频流式输出（`serving_video_output_stream.py`）都用 `websocket.send_bytes(chunk)` 直接推送裸二进制（PCM / fMP4 `m4s` 分片），配合 `websocket.send_json` 发送带外元信息（`type: "audio.start"`、`"video.start"` 等），是二进制帧 + JSON 控制帧交替的协议（协议文档内嵌注释见 `serving_speech_stream.py:1-28`、`serving_video_output_stream.py:1-18`）。

### 4. 流式输出的实现细节

#### 4.1 文本/图像/音频统一 SSE（Chat Completions）

`OmniOpenAIServingChat.chat_completion_stream_generator`（`serving_chat.py:1192`）是核心流式生成器，按 `omni_res.final_output_type` 分支处理：

- 文本分支产出标准 `chat.completion.chunk`（`serving_chat.py:1871` 等），`yield f"data: {data}\n\n"`；
- 音频分支（`serving_chat.py:1911` 起，经 `_create_audio_choice(..., stream=True)` `serving_chat.py:2509-2592`）把当前 delta tensor 转成 base64 PCM/WAV 塞进 `DeltaMessage(content=audio_base64)`；
- 图像分支（`serving_chat.py:1949-1986`，经 `_create_image_choice(..., stream=True)` `serving_chat.py:2594`）把每次产出的图像整帧 base64 编码后作为一条 delta 下发（图像不是逐 pixel 增量，而是逐张完整帧）；
- 循环结束后统一 `yield "data: [DONE]\n\n"`（`serving_chat.py:2099`），并按需追加 `finish_reason=stop` 补丁块（`serving_chat.py:1995-2010`）以保证每个 `modality` 都收到终止信号。

顶层路由 `create_chat_completion`（`api_server.py:1160-1220`）对非流式响应做了三层 fallback 序列化（`model_dump` → `model_dump_json` → 强制忽略警告的 `model_dump`）以规避多模态字段导致的 Pydantic 序列化警告（`api_server.py:1186-1218`）；流式响应直接 `return StreamingResponse(content=generator, media_type="text/event-stream")`（`api_server.py:1220`）。

图像编辑的多阶段流式（`_stream_diffusion_image_chunks` `serving_chat.py:3123`）复用相同套路：AR 阶段的 caption/CoT 文本先以 `data: {...}\n\n` 增量下发，最终 DiT 阶段产出图像时再下发一条包含 `b64_json`（`encode_image_base64_with_compression` 调用 `serving_chat.py:3162`）的 chunk，最后 `yield "data: [DONE]\n\n"`（`serving_chat.py:3221`）。

#### 4.2 Speech 三种流式模式（`/v1/audio/speech`）

`create_speech`（`serving_speech.py:4010-4111`）按请求参数分流：

1. **raw audio streaming**（`stream_format='audio'`）：`_generate_audio_chunks`（`serving_speech.py:2700-2855`）逐块把 codec tensor 转 PCM bytes，`response_format=='wav'` 时先 `yield` 一个占位 header（`_create_wav_header`，`serving_speech.py:2775-2780`），再持续 `yield audio_bytes`；封装为 `StreamingResponse(..., media_type="audio/wav"|"audio/pcm")`（`serving_speech.py:4070-4079`），即 HTTP chunked transfer，非 SSE。
2. **SSE 流**（`stream_format='sse'` 或 `stream=True`）：`_generate_audio_sse_events`（`serving_speech.py:2857-2918`）复用 `_generate_audio_chunks` 拿到 PCM bytes，再 base64 编码后包成 OpenAI 风格事件 `event: speech.audio.delta\ndata: {...}\n\n`，结束时发 `speech.audio.done`（携带 usage 统计）或 `speech.audio.error`（`serving_speech.py:2891-2918`）。
3. **WebSocket 增量文本→语音**（`/v1/audio/speech/stream`）：`OmniStreamingSpeechHandler.handle_session`（`serving_speech_stream.py:81`）先等待 `session.config`（超时保护 `_DEFAULT_CONFIG_TIMEOUT`，`serving_speech_stream.py:167-201`），按句切分文本，每句产出 `audio.start` JSON + 若干 `send_bytes` 二进制帧（或 `word_timestamps=true` 时改为 base64 JSON `audio.chunk`，见协议注释 `serving_speech_stream.py:12-27`）+ `audio.done`，最终 `session.done`。

#### 4.3 视频输出流式（`/v1/realtime/video`）

`OmniStreamingVideoOutputHandler.handle_session`（`serving_video_output_stream.py:85`）接收 `session.start` 控制消息（含 `format: "m4s"` 等，`serving_video_output_stream.py:6-9`），推理产出帧后逐 chunk `await websocket.send_bytes(chunk)`（`serving_video_output_stream.py:122`）推送 fragmented-MP4 片段，`session.ping`/`session.pong`（`281`）做保活，`session.done`（`104`附近）结束会话；`create_streaming_video_encoder`（`vllm_omni/entrypoints/openai/video_api_utils.py`）负责把 diffusion 输出的帧张量编码为 `m4s` 二进制。

#### 4.4 视频输入流式理解（`/v1/video/chat/stream`）

按 `docs/serving/video_stream_api.md:36-51` 描述的协议：客户端持续推送 `video.frame` / `audio.chunk`（base64），服务端在 `video.query` 触发时下发 `response.text.delta` / `response.audio.delta`（base64 WAV）等增量事件，环境变量 `VLLM_VIDEO_ASYNC_CHUNK`（`video_stream_api.md:81`）控制是否逐 token/逐 chunk wire-level 流式或服务端聚合后一次性下发。

### 5. Request → Stage-Engine 派发

所有实时/非流式 serving 类最终都收敛到同一个入口：`AsyncOmni.generate()`（`vllm_omni/entrypoints/async_omni.py:259-418`）。以 Chat Completions 为例：

1. `OmniOpenAIServingChat._create_chat_completion`（`serving_chat.py:306`）完成模型校验、chat template 渲染、tool/reasoning parser 准备后，构建每个 stage 的 `sampling_params_list`（`serving_chat.py:560-587`，通过 `coerce_param_message_types` 按 `request.stream` 决定 cumulative 还是 delta 输出）；
2. 调用 `self.engine_client.generate(prompt=engine_prompt, request_id=request_id, sampling_params_list=sampling_params_list, output_modalities=output_modalities, arrival_time=request_timestamp)`（`serving_chat.py:588-594`），这里 `engine_client` 就是 app.state 上的 `AsyncOmni` 实例；
3. `AsyncOmni.generate`（`async_omni.py:259`）：
   - 校验引擎未处于 sleep 状态（`async_omni.py:319-325`）；
   - 对 diffusion stage 拒绝 list-prompt 批量请求（`async_omni.py:328-334`，每个 diffusion 请求必须独立提交，由调度器做 co-batch）；
   - 通过 `_compute_final_stage_id` / `_compute_final_output_stage_ids`（`async_omni.py:370-371`）确定哪个/哪些 stage 的输出是“终态”（用于 e2e 指标与结果收敛）；
   - 生成内部 `request_id`（追加随机后缀，`async_omni.py:311`），构造 `ClientRequestState`（`async_omni.py:382-388`）记录跨 stage 的状态和 metrics；
   - 若 `prompt` 是 `AsyncGenerator`（流式输入场景），走 `_add_streaming_input_request`（`async_omni.py:400-407`）逐 chunk `streaming_update`；否则 `self.engine.add_request_async(...)`（`async_omni.py:409-416`）把请求一次性提交给内部 `AsyncOmniEngine`（管理各 `StageEngineCoreClient`，通过 ZMQ/进程间通信驱动真正的 stage worker 进程）；
   - `generate()` 是一个 `AsyncGenerator[OmniRequestOutput, None]`，随着各 stage 陆续产出 partial/final 结果不断 `yield`，serving 层的 `chat_completion_stream_generator` 或 `chat_completion_full_generator`（`serving_chat.py:2101`）据此决定是逐块下发 SSE 还是等待耗尽后一次性返回完整 `ChatCompletionResponse`。

图像/视频的独立端点复用同一条链路但走更薄的封装：`generate_images`（非多阶段分支）调用 `_generate_with_async_omni`（`api_server.py:2382-2412`），内部同样是 `engine_client.generate(sampling_params_list=..., **kwargs)` 循环取最后一个 `output` 作为最终结果（`api_server.py:2403-2407`）；多阶段（AR+DiT）分支则直接复用 `openai_serving_chat.generate_diffusion_images`（`api_server.py:1786-1792`，实现于 `serving_chat.py:2988`），保证 `/v1/images/generations` 与 `/v1/chat/completions` 在多阶段场景下走同一套 prompt 构造逻辑（代码注释 `api_server.py:1739-1740` 明确指出这是为了避免二者行为分叉）。视频异步任务则是 `OmniOpenAIServingVideo.generate_video_bytes`（`serving_video.py:296`）内部再调用 stage 分发，由 `asyncio.create_task` 包一层做后台执行（`api_server.py:3063-3074`）。

### 6. CLI / Config 关键旋钮（`vllm serve ... --omni`）

`OmniServeCommand`（`vllm_omni/entrypoints/cli/serve.py:77`）在 upstream `make_arg_parser`（`serve.py:186`）基础上追加 `OmniConfig` 参数组（`serve.py:191-739`），核心项：

| 分类 | Flag | File:line | 作用 |
|---|---|---|---|
| 模式开关 | `--omni` | `serve.py:196` | 启用 vLLM-Omni 多阶段/扩散模式 |
| 拓扑/部署 | `--deploy-config` | `serve.py:251` | 新格式 deploy YAML（stages + engine_args），取代废弃的 `--stage-configs-path`（`serve.py:244`） |
| 拓扑/部署 | `--stage-overrides` | `serve.py:267` | 覆盖 YAML 中单个 stage 的字段 |
| 分布式头/尾 | `--stage-id` / `--omni-master-address` / `--omni-master-port` | `serve.py:280,344,350` | headless worker 加入某个 stage，三者需同时提供（校验见 `serve.py:110-111`） |
| 分布式头/尾 | `--omni-dp-size-local` | `serve.py:370` | 单进程内该 stage 的数据并行副本数，仅在拥有 stage 的进程（head/headless）里 >1 生效（`serve.py:119-126`） |
| 负载均衡 | `--omni-lb-policy` | `serve.py:381` | 校验对应 `LoadBalancingPolicy` 枚举（`serve.py:154-163`） |
| 与 upstream 并行参数互斥 | `--data-parallel-size` 等 8 个 upstream 参数 | 校验于 `serve.py:133-152` | 显式传入即报错，强制并行度只能来自 per-stage YAML |
| Sleep 模式 | `--enable-sleep-mode` | `serve.py:203` | 打开 GPU 显存池 sleep/wakeup（对应 `/v1/omni/sleep`、`/v1/omni/wakeup`） |
| TTS | `--task-type` | `serve.py:212` | TTS 模型默认 task（CustomVoice/VoiceDesign/Base） |
| TTS | `--forced-aligner` / `--forced-aligner-config` | `serve.py:223,233` | 开启流式 TTS 的词级时间戳对齐模型 |
| 扩散并行 | `--usp/--ulysses-degree`、`--ring/--ring-degree`、`--cfg-parallel-size`、`--vae-patch-parallel-size` | `serve.py:449-473,649-664` | 扩散模型的序列并行 / CFG 并行 / VAE 并行配置 |
| 扩散性能 | `--diffusion-attention-backend`、`--diffusion-kv-cache-dtype`、`--cache-backend`、`--step-execution` | `serve.py:518,631,544,564` | 扩散 attention 后端、KV cache 精度、cache-dit 加速、逐步执行策略 |
| 图像/输出限制 | `--max-generated-image-size` | `serve.py:688` | 服务端强制的最大生成图像宽高（对应 `_check_max_generated_image_size`，`api_server.py:1759,1843`） |
| 存储 | `VLLM_OMNI_SERVER_STORAGE__PATH`（环境变量，非 CLI flag） | `docs/serving/videos_api.md:203-211` | 视频/语音等异步产物的本地落盘目录（对应 `storage.py:210` 的 `STORAGE_MANAGER`） |
| headless 部署 | `--headless` | `serve.py:104-107` | 走 `run_headless(args)`（`serve.py:752`）而非起 HTTP server，用于纯 stage worker 进程 |

启动路径：`OmniServeCommand.cmd`（`serve.py:86-107`）非 headless 时 `uvloop.run(omni_run_server(args))`（`serve.py:107`），即前文第 1 节的入口。


---

## §9. vLLM-Omni 配置体系、Pipeline Registry 与 Stage Config 深度解析

> 基于 tag `v0.25.0rc1`（commit `d3c47efc`）源码，全部结论均标注 `file:line`。

### 0. 全局心智模型

vLLM-Omni 把一个"多模态大模型"拆成若干个 **Stage**（例如 Qwen2.5-Omni 的 `thinker → talker → code2wav`），每个 Stage 是一个独立的 vLLM `LLMEngine` / `DiffusionEngine` 进程，Stage 之间通过 Connector（共享内存或 Mooncake）传递隐藏状态/token。整个体系由三层配置对象拼接而成：

1. **Pipeline 拓扑（代码写死，不可配置）** —— `PipelineConfig` / `StagePipelineConfig`，声明在 `vllm_omni/model_executor/models/<model>/pipeline.py`，注册进 `vllm_omni/config/pipeline_registry.py` 的 `OMNI_PIPELINES` 字典。
2. **Deploy YAML（用户可编辑的部署参数）** —— `DeployConfig` / `StageDeployConfig`，声明在 `vllm_omni/deploy/<model_type>.yaml`。
3. **CLI / 运行时覆盖** —— `OmniEngineArgs` / `OrchestratorArgs`（`vllm_omni/engine/arg_utils.py`）。

`merge_pipeline_deploy()`（`vllm_omni/config/stage_config.py:831`）把 (1)+(2) 合并为遗留的 `list[StageConfig]`；同时存在一套并行的"结构化"新路径 `VllmOmniConfig.from_registry()`（`vllm_omni/config/omni_config.py:1226`，标注为 RFC #4021 Phase 2，用于在 cutover 前验证与旧路径的等价性）。两条路径当前共存。

---

### 1. Omni Config 模型 —— OmniConfig / StageConfig 对象体系

#### 1.1 拓扑层（frozen，不可被用户覆盖）

| 类 | 文件:行 | 作用 |
|---|---|---|
| `StagePipelineConfig` | `vllm_omni/config/stage_config.py:208` | 单个 stage 的**固定拓扑**：`stage_id`、`model_stage`（如 `"thinker"`）、`execution_type`（`LLM_AR` / `LLM_GENERATION` / `DIFFUSION`）、`input_sources`（上游 stage id 元组）、`final_output`/`final_output_type`、`owns_tokenizer`、`engine_output_type`（`latent`/`text`/`audio`）、`custom_process_next_stage_input_func` 等处理函数挂载点。`frozen=True`，即代码级常量。 |
| `PipelineConfig` | `vllm_omni/config/stage_config.py:242` | 一个模型的**完整拓扑**：`model_type`、`model_arch`、`stages: tuple[StagePipelineConfig,...]`，以及用于消歧的 `hf_architectures`、`hf_config_predicate`、`diffusers_class_name`、`endpoint_restrictions`。`validate()`（`stage_config.py:281`）检查重复 stage_id、非法 `input_sources`、必须存在的入口 stage。 |

#### 1.2 部署层（用户在 YAML 里编辑）

| 类 | 文件:行 | 作用 |
|---|---|---|
| `StageDeployConfig` | `vllm_omni/config/stage_config.py:302` | 每个 stage **随部署环境变化**的旋钮：`devices`、`tensor_parallel_size`、`gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens`、`max_model_len`、`enforce_eager`、扩散专属字段（`ulysses_degree`/`cfg_parallel_size`/`cache_backend`…）、`output_connectors`/`input_connectors`、`engine_extras`（兜底透传）。 |
| `DeployConfig` | `vllm_omni/config/stage_config.py:406` | 一份 deploy YAML 的顶层对象：`async_chunk`、`connectors`、`edges`、`stages: list[StageDeployConfig]`、`platforms`（npu/rocm/xpu 覆盖）、`pipeline`（覆盖自动探测的 registry key）；以及**pipeline-wide** 字段 `trust_remote_code`/`distributed_executor_backend`/`dtype`/`quantization`/`enable_prefix_caching`/`enable_chunked_prefill`/`data_parallel_size`/`pipeline_parallel_size`（对所有 stage 生效，见 `stage_config.py:742` 的 `_PIPELINE_WIDE_ENGINE_FIELDS`）。 |
| `load_deploy_config()` | `stage_config.py:602` | 加载 YAML → `DeployConfig`；`resolve_deploy_yaml()`（`stage_config.py:576`）先处理 `base_config:` 继承（`stages:`/`platforms:` 按 `stage_id` 深合并，其余标量 overlay 覆盖 base）。 |

#### 1.3 合并产物（引擎实际消费）

| 类/函数 | 文件:行 | 作用 |
|---|---|---|
| `merge_pipeline_deploy(pipeline, deploy, cli_overrides)` | `stage_config.py:831` | 遗留路径核心：拓扑 + Deploy + Platform overrides → `list[StageConfig]`。执行 `_apply_platform_overrides`、单 stage 时强制 `async_chunk=False`、校验多 stage pipeline 必须有 `async_chunk_process_next_stage_input_func` 才能开 `async_chunk`（`stage_config.py:858-868`）。 |
| `StageConfig` | `stage_config.py:913` | 单 stage 最终态（legacy，供新旧 loader 共用），持有 `yaml_engine_args`/`yaml_runtime`/`yaml_extras`/`runtime_overrides` 四个字典；`to_omegaconf()`（`stage_config.py:936`）把它们拍平成引擎实际使用的 OmegaConf `DictConfig`，并处理 CLI 覆盖优先级与 `runtime.max_batch_size` 的废弃迁移。 |

#### 1.4 新结构化路径（`vllm_omni/config/omni_config.py`，RFC #4021 Phase 2）

顶层文档字符串（`omni_config.py:3-8`）明确说明这是"additive"、用于在旧路径切换前验证等价性。

| 类 | 文件:行 | 作用 |
|---|---|---|
| `VllmOmniConfig` | `omni_config.py:1213` | 顶层结构化配置：`pipeline_config: PipelineConfig`、`stage_configs: tuple[StageConfigType,...]`、`orchestrator_config: VllmOmniOrchestratorConfig`。`from_registry()`（`omni_config.py:1226`）是构建入口，等价于遗留路径的 `_create_from_registry`。 |
| `BaseVllmOmniStageConfig` / `VllmOmniARStageConfig` / `VllmOmniGenerationStageConfig` / `VllmOmniDiffusionStageConfig` | `omni_config.py:820`, `912`, `917`, `922` | 按 execution_type 特化的 stage 结构化配置，聚合以下子配置对象；`StageConfigType`（`omni_config.py:929`）是三者的 `TypeAlias` 联合。 |
| `OmniStageModelConfig` / `OmniStageLoadConfig` / `OmniStageCacheConfig` / `OmniStageSchedulerConfig` / `OmniStageConnectorConfig` / `OmniStageRuntimeConfig` / `OmniStageParallelConfig` / `OmniStageDiffusionParallelConfig` | `omni_config.py:247/267/277/291/308/322/335/349` | 按关注点切分的 pydantic dataclass（`@config` 装饰器来自 `vllm.config.utils.config`），分别管理采样默认值、load_format、显存/prefix cache、调度器（`max_num_seqs`/`max_num_batched_tokens`）、Connector 拓扑、进程放置（`devices`/`num_replicas`）、并行度（`tensor_parallel_size`/`data_parallel_size`/`pipeline_parallel_size`/`world_size` 自动计算，`omni_config.py:344`）。扩散并行额外含 `ulysses_degree`/`ring_degree`/`cfg_parallel_size`/HSDP 校验（`omni_config.py:364-413`）。 |
| `VllmOmniOrchestratorConfig` | `omni_config.py:801` | Orchestrator 进程专属：`stage_init_timeout`、`omni_master_address/port`、`omni_lb_policy`（默认 `"random"`）、`shm_threshold_bytes` 等。 |

#### 1.5 `OmniModelConfig` —— 单 stage 的 vLLM `ModelConfig` 扩展

`vllm_omni/config/model.py:83` 的 `OmniModelConfig(ModelConfig)` 是 **每个 stage 内部真正喂给 vLLM 引擎**的对象，新增 `stage_id`/`model_stage`/`model_arch`/`worker_type`/`engine_output_type`/`hf_config_name`（用于从多组件 HF config 中取出 `thinker_config`/`talker_config` 子配置，`model.py:196`）。`OmniModelArchConfigConvertor`（`model.py:21`）处理量化配置从 stage 专属子配置提取（避免 talker/code2wav 误继承 thinker 的量化）。构造走 `from_vllm_model_config()`（`model.py:244`），用 `object.__new__` 绕过昂贵的 `ModelConfig.__post_init__` 重复校验。

---

### 2. Pipeline Registry —— HF 模型名 → Stage 拓扑

`vllm_omni/config/pipeline_registry.py:96` 定义 `OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc]`，value 要么是常量 `PipelineConfig`，要么是接受可选 `PretrainedConfig` 并返回 `PipelineConfig` 的 resolver（用于同一 `model_type` 但结构随 HF config 变化的模型，如 Qwen3-Omni 的 `resolve_qwen3_omni_pipeline`，`pipeline_registry.py:100`）。`register_pipeline()`（`pipeline_registry.py:140`）供树外插件注册。

#### 2.1 model_type 解析流程（`StageConfigFactory`，`vllm_omni/config/config_factory.py:47`）

`try_infer_model_type()`（`config_factory.py:106`）按优先级探测：
1. `get_config(model, trust_remote_code)`（HF `AutoConfig`）读到的 `model_type`；
2. 直接读 `config.json` 的 `model_type` / `architecture`（VoxCPM2 风格，`config_factory.py:150-156`）；
3. `model_index.json` 的 `_class_name` 匹配某个 `PipelineConfig.diffusers_class_name`（GLM-Image 风格，`config_factory.py:164-180`）；
4. 模型路径 basename 做最长子串匹配 registry key（CosyVoice3 风格，`config_factory.py:184-195`）。

`get_pipeline_config()`（`config_factory.py:201`）之后按优先级解析出 `PipelineConfig`：deploy YAML 显式 `pipeline:` 字段（最高优先级，`_get_deploy_override_pipe_config`，`config_factory.py:257`）> `model_type` 直接命中 `OMNI_PIPELINES` > 遍历所有已注册 pipeline，用 `hf_config.architectures` 与 `PipelineConfig.hf_architectures` 求交集消歧（并可选执行 `hf_config_predicate` 二次过滤，例如 MiniCPM-o 4.5 vs 2.6 都上报 `architectures=["MiniCPMO"]`，靠 `version=="4.5"` 区分，见 `stage_config.py:262-265` 文档注释）。

`create_from_model()` → `_create_from_registry()`（`config_factory.py:274/322`）是最终产出 `list[StageConfig]` 的入口：加载 `_DEPLOY_DIR/<model_type>.yaml`（默认 `vllm_omni/deploy/`），若不存在则退回 `DeployConfig()` 默认值并告警；调用 `merge_pipeline_deploy`；再叠加可选的 composable-parallel `strategy_specs`（`_apply_strategy_specs`，`config_factory.py:386`）；最后合并 CLI overrides 并重新校验设备布局（`_reconcile_strategy_with_cli`，`config_factory.py:415`）。

#### 2.2 已注册 Pipeline 一览（节选，`pipeline_registry.py:96-137`）

| Registry key | 模型家族 / model_arch | Stages（`model_stage`） | 定义文件 |
|---|---|---|---|
| `qwen2_5_omni` | `Qwen2_5OmniForConditionalGeneration` | `thinker`(LLM_AR) → `talker`(LLM_AR) → `code2wav`(LLM_GENERATION) | `vllm_omni/model_executor/models/qwen2_5_omni/pipeline.py:18` |
| `qwen2_5_omni_thinker_only` | 同上 | 仅 `thinker` | 同上 `:64` |
| `qwen3_omni_moe` | `Qwen3OmniMoeForConditionalGeneration`（resolver：`resolve_qwen3_omni_pipeline`） | `thinker` → `talker` → `code2wav` | `vllm_omni/model_executor/models/qwen3_omni/pipeline.py:23` |
| `qwen3_tts` | `Qwen3TTSTalkerForConditionalGeneration` / `Qwen3TTSCode2Wav` | `qwen3_tts` → `code2wav` | `vllm_omni/model_executor/models/qwen3_tts/pipeline.py:17` |
| `bagel` / `bagel_think` | `OmniBagelForConditionalGeneration` | `thinker`(LLM_AR) → `dit`(DIFFUSION) | `vllm_omni/model_executor/models/bagel/pipeline.py:29/65` |
| `bagel_single_stage` | `BagelForConditionalGeneration` | 仅 `dit` | 同上 `:98` |
| `glm_image` | `GlmImageForConditionalGeneration` / `GlmImagePipeline`（`diffusers_class_name` 检测） | `ar` → `dit` | `vllm_omni/model_executor/models/glm_image/pipeline.py:16` |
| `hunyuan_image_3_moe` / `hunyuan_image3_ar` / `hunyuan_image3_dit` | HunyuanImage-3 | `AR` → `dit`（或单独拆分变体） | `vllm_omni/model_executor/models/hunyuan_image3/pipeline.py:20/52/73` |
| `dreamzero` | — | 单 stage `diffusion` | `vllm_omni/model_executor/models/dreamzero/pipeline.py:12` |
| `Gr00tN1d7` | — | 单 stage `diffusion` | `vllm_omni/model_executor/models/gr00t/pipeline.py:12` |
| 其它（`aura_omni`、`covo_audio`、`cosyvoice3`、`mimo_audio`、`ming_tts(_moe)`、`voxtral_tts`、`glm_tts`、`fish_qwen3_omni`、`ming_flash_omni*`、`moss_tts_*`、`omnivoice`、`mammoth_moda2*`、`minicpmo_4_5`、`higgs_audio_v2/v3`、`dynin_omni`、`indextts2`、`voxcpm2`） | 见对应 import | 各 1~3 stage | `pipeline_registry.py:36-89` 逐一 import |

> 说明：单 stage 扩散模型（无多 stage 拆分需求）默认走 `async_omni_engine.py` 的 `_create_default_diffusion_stage_cfg` 兜底，不进 registry（`pipeline_registry.py:20-22`）。

---

### 3. Stage Config —— 每 Stage 的并行/显存/模型设置与 `config_factory.py` 的角色

#### 3.1 Deploy YAML 字段（用户编辑面）

依据 `docs/configuration/stage_configs.md:37-51`（新 schema）：

| 字段 | 归属 | 默认值 | 说明 |
|---|---|---|---|
| `stage_id` | 每 stage | — | 匹配 `PipelineConfig.stages[*].stage_id` |
| `tensor_parallel_size` | 每 stage | `1` | TP 度 |
| `gpu_memory_utilization` | 每 stage | `0.9` | 单 stage 显存预算 |
| `max_num_seqs` | 每 stage | `64` | 并发序列数 |
| `max_num_batched_tokens` | 每 stage | `32768` | prefill 预算 |
| `enforce_eager` | 每 stage | `false` | 关闭 CUDA graph |
| `devices` | 每 stage | `"0"` | 逻辑设备号（经 `CUDA_VISIBLE_DEVICES`/`ASCEND_RT_VISIBLE_DEVICES` 映射） |
| `input_connectors`/`output_connectors` | 每 stage | `null` | 引用顶层 `connectors:` 命名的 Connector |
| `trust_remote_code`/`distributed_executor_backend`/`dtype`/`quantization`/`enable_prefix_caching`/`enable_chunked_prefill`/`data_parallel_size`/`pipeline_parallel_size` | **pipeline-wide**（顶层） | 见 `stage_config.py:428-437` | 对所有 stage 统一生效，stage 内 `engine_extras` 可覆盖单个 stage |

#### 3.2 `StageConfigFactory` 的角色（`config_factory.py:47`）

它是"pipeline registry + deploy YAML + CLI"三者的**装配厂**：
- 探测/解析 `model_type` → `PipelineConfig`（第 2 节流程）；
- 加载对应 `DeployConfig`；
- 调用 `merge_pipeline_deploy` 生成 `list[StageConfig]`；
- 应用 composable-parallel `strategy_specs`（可选）；
- 合并显式 CLI overrides（`_merge_cli_overrides`，`config_factory.py:555`）到每个 `StageConfig.runtime_overrides`；
- 通过 `_reconcile_strategy_with_cli`（`config_factory.py:415`）在 CLI 覆盖落地后，重新校验 `tensor_parallel_size * data_parallel_size * pipeline_parallel_size * num_replicas` 与 `devices` 数量是否匹配（"device-layout guard"）。
- `create_default_diffusion()`（`config_factory.py:494`）是未注册 pipeline 的扩散模型兜底路径。

#### 3.3 GPU 显存与并行的实践规则（`docs/configuration/gpu_memory_utilization.md`）

- 公式：`requested_memory = total_gpu_memory × gpu_memory_utilization`，需满足 `free_memory ≥ requested_memory`（第 15-26 行）。
- 同 GPU 多 stage 时各 stage `gpu_memory_utilization` 之和不能超过 1.0（第 77-96 行）；不同 GPU 的 stage 各自独立可达 1.0。
- `tensor_parallel_size > 1` 时模型按 TP 切分，单卡显存占比相应下降（第 100-114 行）。

#### 3.4 Composable Parallel（`vllm_omni/config/composable_parallel/`）

在 Deploy YAML 之上叠加一层**声明式并行策略**（opt-in，`--strategy-config`）：`strategies: {model_stage_name: [StrategySpec,...]}`，每个 `StrategySpec` 声明 `axis`（`tp`/`dp`/`pp`/`ep`/`stage_replica`，其余保留）+ `size` + 可选 `routing`/`l1_owner`（`docs/configuration/composable_parallel.md:22-30`）。`stage_replica` 不是 vLLM 世界维度，而是驱动 omni 的 `StagePool` 复制整个 stage 引擎并设置 pipeline 级 `omni_lb_policy`（第 189-196 行），该值最终由 `AsyncOmniEngine._apply_strategy_lb_policy`（`vllm_omni/engine/async_omni_engine.py:1129` 附近）应用一次。优先级（从高到低，第 91-100 行）：CLI overrides > Strategy YAML > Deploy YAML > Parser defaults。

#### 3.5 PD 分离（`docs/configuration/pd_disaggregation.md`）

不是内置 bundled YAML，而是一份 stage-config **配方**：把 thinker 拆成 `is_prefill_only: true` 和 `is_decode_only: true` 两个 stage，双方都必须声明 `kv_transfer_config`（`kv_connector: "MooncakeConnector"`、`kv_role`、`kv_rank`、`kv_parallel_size`），且 TP 度必须一致（第 24-172 行）；orchestrator 会强制 prefill stage 以 `max_tokens=1` 运行（第 114-115 行）。

---

### 4. Custom Pipeline 特性 —— 用户自定义扩散 Pipeline

`docs/features/custom_pipeline.md` 描述的是**扩散（diffusion）Worker 级**扩展点，而非 Stage 拓扑级扩展（Stage 拓扑扩展见第 2 节 `register_pipeline`）：

| 组件 | 文件 | 作用 |
|---|---|---|
| `WorkerWrapperBase` | `vllm_omni/diffusion/worker/diffusion_worker.py` | 通过 `worker_extension_cls` 动态继承，创建带自定义方法的 `DiffusionWorker` 实例。 |
| `load_format="custom_pipeline"` | `vllm_omni/diffusion/model_loader/diffusers_loader.py` | 除 `"default"`（走 model registry）/`"dummy"`（跳过加载）外，第三种加载模式：按 `custom_pipeline_name` 加载用户自定义 Pipeline 类。 |
| `CustomPipelineWorkerExtension` | `vllm_omni/diffusion/worker/diffusion_worker.py` | mixin，提供 `re_init_pipeline(custom_pipeline_args)`，先清理旧 pipeline 再用自定义实现重建。 |

用法示例（`docs/features/custom_pipeline.md:62-107`）：用户子类化既有 Pipeline（如 `QwenImageEditPipeline`）覆写 `forward()`，通过 `Omni(model=..., diffusion_load_format="dummy", custom_pipeline_args={"pipeline_class": "custom_pipeline.CustomPipeline"})` 注入；还可传 `worker_extension_cls` 给 Worker 挂自定义方法（内部参数）。这套机制作用在**单个 diffusion stage 内部**，与前述"注册一个新的多 stage `PipelineConfig`"（`register_pipeline`）是两个不同层次的"custom pipeline"。

---

### 5. Omni 如何接入上游 vLLM —— `patch.py` 与 `plugins/`

`vllm_omni/__init__.py:22` 在包导入时（先导入 `version` 做版本告警，再 patch）执行 `from . import patch`，即模块级副作用完成所有 monkey-patch。`vllm_omni/patch.py`（486 行）中的补丁分为"能力补齐"与"upstream bug 规避"两类：

| Patch | 位置 | Why（原因摘要） |
|---|---|---|
| `ModelConfig.is_mm_prefix_lm` | `patch.py:50-76` | HunyuanImage-3 需要图像 token 双向注意力，但 vLLM 内置的 `MM_PREFIX_LM_MODELS` 白名单不含 `hunyuan_image_3_moe`；该检查发生在 vLLM core（调度器/attention backend 选择）里，模型级 hook 够不到，只能在 `ModelConfig` 层打补丁（通过 `__dict__` 绕过 `cached_property` 在 pydantic dataclass 下的限制）。 |
| `ModelOptNvFp4LinearMethod.process_weights_after_loading` | `patch.py:120-255` | ModelOpt 0.44 导出的 NVFP4 W4A4 权重在 FP8 per-block scale 里偶发 NaN 字节，传播到 FlashInfer FP4 GEMM 导致模型输出坍缩成 `!!!!`；在 PWAL 之前原地 clamp。带自愈探测（`_already_patched_upstream` 通过反射 `co_names` 判断上游是否已修复）和逃生开关 `VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP`。 |
| `GlmImageTextConfig.__init__` | `patch.py:264-281` | GLM-Image 用 M-RoPE `mrope_section: [8,12,12]`，但 transformers 未在 `rope_parameters` 暴露；vLLM 的 `uses_mrope` 检测依赖该字段存在。 |
| `RequestStatus` 扩展 `WAITING_FOR_CHUNK`/`WAITING_FOR_INPUT` | `patch.py:284-292` | omni 特有的调度状态（等待跨 stage 分片 / 等待非 stage-0 输入），用 `aenum.extend_enum` 追加枚举值。 |
| 全局类替换：`EngineCoreOutput(s)`/`TokensPrompt`/`MRotaryEmbedding`/`Request`/`StreamingUpdate`/`EngineCoreRequest` → Omni 对应子类 | `patch.py:294-314` | 遍历 `sys.modules` 中所有 `vllm*` 模块，把已导入的引用替换为 Omni 扩展版本，实现"数据结构级"的透明替换而不改 vLLM 源码。 |
| Chat template 注册表补 `qwen3_omni_moe` | `patch.py:323-338` | Qwen3-Omni 的 `chat_template` 存于独立 `chat_template.json`，旧版 transformers 不会加载，vLLM 回退表里没有 `qwen3_omni_moe` 项。 |
| `ScaledMMLinearKernel.apply_weights` 强制 contiguous | `patch.py:357-364` | 扩散 batching 下激活可能非连续，FP8 ScaledMM 的 `x.view(-1,...)` 要求连续。 |
| `FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm` 补形状 reshape | `patch.py:391-404` | FlashInfer FP8 kernel 忽略 `output_shape`，3D 激活会被拍扁成 2D，破坏按维度 reshape 的 DiT（如 Wan2.2）。 |
| `CuMemAllocator._python_free_callback` | `patch.py:444-483` | 原始实现的"睡眠态双重释放保护"只对 ROCm 生效（`is_rocm()` 判断），CUDA 上进程退出时对已释放内存二次 `cuMemRelease` 报 `CUDA_ERROR_INVALID_VALUE`；补丁去掉平台限制。 |

`vllm_omni/plugins/__init__.py` 实现的是**vLLM 官方插件机制的 omni 分组**，不是 monkey patch：定义两个 entry-point group ——`OMNI_DEFAULT_PLUGINS_GROUP = "vllm_omni.general_plugins"`（`plugins/__init__.py:14`，所有进程都加载）与 `OMNI_PLATFORM_PLUGINS_GROUP = "vllm_omni.platform_plugins"`（`:18`，惰性加载于 `current_omni_platform` 首次访问时）。`load_omni_general_plugins()`（`plugins/__init__.py:61`）先调用 vLLM 自身的 `load_general_plugins()`，再按 `VLLM_PLUGINS` 环境变量过滤加载 omni 分组下的 entry points；带全局幂等标记 `omni_plugins_loaded`。该函数被 `OmniEngineArgs.__post_init__`（`vllm_omni/engine/arg_utils.py:210`）调用，确保每个 Stage 进程（含 worker 子进程）启动引擎参数解析时都触发一次插件装载。

---

### 6. `endpoint_policy.py` 与 `server_settings.py`

#### 6.1 `endpoint_policy.py`（91 行）—— 按 Pipeline 关闭不支持的 OpenAI 兼容路由

- `OmniServingCapability`（`endpoint_policy.py:21`）枚举当前可被"关停"的能力，值为 `RouteTarget(path, methods)`（如 `/v1/chat/completions/batch`、`/v1/completions`）。
- `EndpointRestriction(capability, reason)`（`:36`）是 `PipelineConfig.endpoint_restrictions` 元组里的元素，由 `StageConfigFactory.get_pipeline_endpoint_restrictions()`（`config_factory.py:62`）按 model 解析出来。
- `UNSUPPORTED_ROUTES`（`:44`）是全局硬编码的临时限制（batched chat completions 尚未支持）。
- `shutdown_unsupported_routes(app, endpoint_restrictions)`（`:65`）在 FastAPI app 初始化后，把受限路由从路由表移除并替换成返回 400 的 `rejection_handler`（`build_rejection_handler`，`:52`），错误信息使用 pipeline 特定的 `reason`。

#### 6.2 `server_settings.py`（58 行）—— 基于 `pydantic-settings` 的服务端存储配置

- `FileBackend`（`server_settings.py:21`）：`path`（默认 `/tmp/storage`）、`file_concurrency`（默认 4）、`file_ttl`/`ttl_sweep_interval`（默认在设置了 `file_ttl` 但未设 `ttl_sweep_interval` 时补 300 秒，`:46-50`）。
- `_resolve_deprecated_env()`（`:9`）实现旧环境变量到新环境变量的兼容迁移并发 `DeprecationWarning`：`VLLM_OMNI_STORAGE_PATH` → `VLLM_OMNI_SERVER_STORAGE__PATH`，`VLLM_OMNI_STORAGE_MAX_CONCURRENCY` → `VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY`。
- `ServerSettings(BaseSettings)`（`:53`）：`env_prefix="VLLM_OMNI_SERVER_"`，`env_nested_delimiter="__"`，全局单例 `SERVER_SETTINGS_CONFIG = ServerSettings()`（`:58`），供输出文件（生成的图片/音频）落盘路径与并发控制读取。

---

### 7. 关键配置旋钮 / 环境变量 / CLI 参数一览

#### 7.1 顶层 CLI（`vllm serve <model> --omni ...`，来自 `vllm_omni/entrypoints/cli/serve.py`）

| Flag | 文件:行 | 说明 |
|---|---|---|
| `--omni` | `serve.py:195` / `arg_utils.py:181` | 启用 Omni 引擎特性 |
| `--deploy-config PATH` | `serve.py:250` | 新 schema deploy YAML；与 `--stage-configs-path` 互斥，优先级更高 |
| `--stage-configs-path PATH` | `serve.py:243` | **已废弃**：legacy `stage_args` YAML |
| `--strategy-config PATH` | `serve.py:257` | Composable-parallel 策略 YAML，仅对 registry-based 模型生效 |
| `--stage-overrides JSON` | `serve.py:266` | 逐 stage JSON 覆盖，如 `'{"0":{"gpu_memory_utilization":0.5}}'` |
| `--async-chunk` / `--no-async-chunk` | `serve.py:273` | 覆盖 deploy YAML 的 `async_chunk` |
| `--stage-id INT` | `serve.py:279` | 单进程只启动某个 stage（配合 `--headless`） |
| `--omni-master-address` / `-oma` | `serve.py:343` | Orchestrator（Stage 0）监听地址，worker stage 用它注册 |
| `--omni-master-port` / `-omp` | `serve.py:349` | 同上端口 |
| `--omni-lb-policy` | `serve.py:381` | Orchestrator 级负载均衡策略，需与 `LoadBalancingPolicy` 枚举匹配（`serve.py:154-163`），且不能与 composable-parallel 派生值冲突 |
| `--headless` | `serve.py:104` | Worker stage 模式，不起 API server |
| `--enable-sleep-mode` | `serve.py:202` | GPU 显存池睡眠模式 |
| `--stage-init-timeout` / `--init-timeout` | `serve.py:296/301` | 单 stage / 全 pipeline 初始化超时（秒），默认 300/600 |
| `--worker-backend` | `serve.py:330` | `multi_process`（默认）或 `ray` |
| `--shm-threshold-bytes` | `serve.py:307` | 共享内存 Connector 阈值，默认 65536 |
| `--task-type` | `serve.py:211` | TTS 模型任务类型：`CustomVoice`/`VoiceDesign`/`Base` |
| `--tts-max-instructions-length` | `stage_configs.md:522` | TTS 语音风格指令最大字符数，默认 500 |

#### 7.2 Stage-level engine args（体现在 deploy YAML `stages:` 或 `OmniEngineArgs`，`arg_utils.py:153-202`）

| 字段 | 默认值 | 位置 |
|---|---|---|
| `tensor_parallel_size` | `1` | `StageDeployConfig.tensor_parallel_size`（`stage_config.py:329`） |
| `gpu_memory_utilization` | `0.9`（新 schema）/ `0.8`（旧 schema文档示例） | `stage_config.py:331`；`docs/configuration/stage_configs.md:41` |
| `max_num_seqs` | `64`（新）/ `1`（旧文档默认） | `stage_config.py:332` |
| `max_num_batched_tokens` | `32768` | `stage_config.py:333` |
| `enforce_eager` | `false`（新）/ `true`（旧文档，当前仅支持 eager） | `stage_config.py:337` |
| `devices` | `"0"` | `StageDeployConfig.devices`（`stage_config.py:317`） |
| `async_scheduling` | `null`（继承 `OmniStageSchedulerConfig.async_scheduling=True`） | `stage_config.py:338`；`omni_config.py:298` |
| `enable_expert_parallel` | `False` | `omni_config.py:341` |

#### 7.3 环境变量

| 变量 | 作用 | 位置 |
|---|---|---|
| `VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP` | 跳过 NVFP4 NaN clamp 补丁安装（诊断用逃生舱） | `patch.py:198` |
| `VLLM_OMNI_SERVER_STORAGE__PATH` / 旧 `VLLM_OMNI_STORAGE_PATH` | 生成文件落盘目录 | `server_settings.py:23/37` |
| `VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY` / 旧 `VLLM_OMNI_STORAGE_MAX_CONCURRENCY` | 文件并发写入上限 | `server_settings.py:24/41-43` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | 当 stage `max_model_len` 超过 HF 默认时自动置 1（`docs/configuration/stage_configs.md:45`） | 文档 |
| `CUDA_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` | 把 `devices:` 里的逻辑索引映射为物理 GPU | `stage_configs.md:369-371` |
| `VLLM_PLUGINS` | 控制 `vllm_omni.general_plugins`/`vllm_omni.platform_plugins` 分组下哪些插件被加载 | `plugins/__init__.py:27` |

#### 7.4 优先级总表（合并自 stage_configs.md / composable_parallel.md）

1. 单 stage 显式 CLI 参数（stage-based CLI 范式下每个进程只管一个 stage）/ `--stage-overrides` JSON
2. 全局显式 CLI flag（如 `--gpu-memory-utilization`）
3. Composable-parallel `--strategy-config` 派生的 sizing（写入即视为"显式"，与已有值冲突会抛 `StrategyApplyError`）
4. `platforms.<npu|rocm|xpu>.stages` 覆盖
5. `base_config:` overlay YAML
6. Deploy YAML 基础值
7. Parser 默认值 / dataclass 默认值

---

### 8. 小结：谁该改哪个文件

| 需求 | 应该动的文件 |
|---|---|
| 新增一个多 stage 模型架构 | 新建 `vllm_omni/model_executor/models/<model>/pipeline.py` 定义 `PipelineConfig`，在 `pipeline_registry.py` 的 `OMNI_PIPELINES` 里注册（或调用 `register_pipeline`） |
| 调整某模型的显存/并行/批大小等部署参数 | 编辑（或新建 overlay）`vllm_omni/deploy/<model_type>.yaml`，走 `--deploy-config` |
| 声明式改变某 stage 的 TP/DP/PP/副本数 | 写一个 `strategy.yaml`，走 `--strategy-config`（composable_parallel） |
| 临时/一次性覆盖单个 stage 的参数 | `--stage-overrides '{"<id>": {...}}'` 或 stage-based CLI 直接传对应 flag |
| 自定义单个 diffusion stage 的前向逻辑 | 参照 `docs/features/custom_pipeline.md`，走 `custom_pipeline_args` + `WorkerWrapperBase`/`CustomPipelineWorkerExtension` |
| 修复/绕过 vLLM 上游 bug 或补齐能力 | `vllm_omni/patch.py`（monkey-patch，模块导入即生效）或 `vllm_omni/plugins/`（entry-point 插件机制） |
| 关闭某模型不支持的 OpenAI 端点 | `PipelineConfig.endpoint_restrictions` + `endpoint_policy.py` |


---

## §10. vLLM-Omni v0.25.0rc1 平台后端 / 量化 / Attention / 工具链 拆解

> 代码位置均基于 tag `v0.25.0rc1` 的 checkout worktree（`vllm_omni/...`）。

### 1. Platform / Backend 覆盖表

`vllm_omni/platforms/interface.py:31` 定义抽象基类 `OmniPlatform(Platform)`（继承自 vLLM 原生 `Platform`），并扩展了一批 Omni 特有接口：`get_omni_ar_worker_cls`（AR stage worker）、`get_omni_generation_worker_cls`（generation/code2wav stage worker）、`get_diffusion_attn_backend_cls`（DiT attention 后端选择）、`get_diffusion_worker_cls` / `get_diffusion_model_runner_cls`（DiT stage worker/runner，默认落到 `vllm_omni/diffusion/worker/diffusion_worker.py`）、`get_profiler_cls`、`get_graph_wrapper_cls`（图捕获包装器）。`vllm_omni/platforms/__init__.py:132` 的 `resolve_current_omni_platform_cls_qualname()` 通过探测 `pynvml`/`amdsmi`/`torch.npu`/`torch.xpu`/`torchada` 依次判定当前硬件，实现"一个进程只能激活一种 Platform"的互斥逻辑（`vllm_omni/platforms/__init__.py:150`）。

| Platform | 关键文件:行 | Omni 专属能力 | AR stage | DiT stage |
|---|---|---|---|---|
| CUDA (默认) | `vllm_omni/platforms/cuda/platform.py:20` `CudaOmniPlatform(OmniPlatform, CudaPlatformBase)` | DiT attention 后端自动选型：按 SM 能力/cuDNN 版本/FlashInfer 可用性区分 Blackwell(sm_100/103/120/121)→CUDNN_ATTN 或 FLASHINFER_ATTN、Hopper/Ampere→FLASH_ATTN、否则 SDPA（`platform.py:56-207`）；支持 `FLASH_ATTN_HUB`/`FLASH_ATTN_3_HUB`（HuggingFace `kernels` 库懒加载 FA kernel，`platform.py:122-147`）；`SAGE_ATTN_3`（Blackwell-only SageAttention3，`platform.py:159-174`）；`get_default_ir_op_priority` 为 `rms_norm` 引入 `oink` 算子优先级（`platform.py:251-262`） | `vllm_omni.worker.gpu_ar_worker.GPUARWorker` | `DiffusionWorker`（基类默认） |
| ROCm/AMD | `vllm_omni/platforms/rocm/platform.py:18` `RocmOmniPlatform(OmniPlatform, RocmPlatform)` | 覆盖 vLLM v0.19+ 默认 `ROCM_ATTN`，AR 阶段强制使用 `TRITON_ATTN`/`ROCM_AITER_FA`（注释见 `platform.py:25-51`，实际逻辑在 `vllm_omni/engine/stage_init_utils.py`）；DiT attention 依赖 `aiter`（仅 gfx942/gfx950，`platform.py:79-117`）；`apply_patches()` 打 GroupNorm patch（`vllm_omni/platforms/rocm/patch/worker/patch_groupnorm.py`） | `GPUARWorker`（复用 GPU worker） | `DiffusionWorker` |
| NPU/Ascend (Huawei) | `vllm_omni/platforms/npu/platform.py:24` `NPUOmniPlatform(OmniPlatform, NPUPlatform)`（继承自 vllm-ascend） | 专属 AR/generation worker 与 model runner（下表）；`ACLGraphWrapper` 替代 CUDAGraph（`platform.py:193-196`）；`set_ascend_forward_context` 替代通用 `set_forward_context`（`platform.py:198-214`）；DiT attention 依赖 `mindiesd`（FLASH_ATTN）否则 SDPA（`platform.py:100-131`）；310P (低算力昇腾) 专属 patch（`vllm_omni/platforms/npu/_310p/`，含 Qwen3-TTS code2wav conv2d 兼容补丁 `patch/qwen3_tts.py` 和 worker patch `patch/worker.py`）；HunyuanImage-3 fused MoE 的 NPU 专属实现 `AscendHunyuanFusedMoE`（`vllm_omni/platforms/npu/models/hunyuan_fused_moe.py`）；YuanRong 分布式 KV/权重传输连接器（`vllm_omni/platforms/npu/omni_connectors/yuanrong_transfer_engine_connector.py:88` `YuanrongTransferEngineConnector(OmniConnectorBase)`） | `vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker`（`npu_ar_worker.py:11`）+ `NPUARModelRunner`（`npu_ar_model_runner.py:103`） | `vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker`（`npu_generation_worker.py:11`）+ `NPUGenerationModelRunner`（`npu_generation_model_runner.py:41`），二者共享 `OmniNPUWorkerBase`（`worker/base.py:16`，把 vllm-ascend profiler 换成 `OmniProfiler`） |
| XPU/Intel GPU | `vllm_omni/platforms/xpu/platform.py:16` `XPUOmniPlatform(OmniPlatform, XPUPlatform)` | 专属 AR/generation worker（下方）；DiT attention 按设备架构黑名单跳过 FA（Intel Max 1100/1550, `architecture==13136561920` 时禁用，`platform.py:39-41`）；`get_default_ir_op_priority` 引入 `xpu_kernels` 优先级（`platform.py:106-119`） | `vllm_omni.platforms.xpu.worker.xpu_ar_worker.XPUARWorker`（`xpu_ar_worker.py:10`） | `vllm_omni.platforms.xpu.worker.xpu_generation_worker.XPUGenerationWorker`（`xpu_generation_worker.py:10`） |
| MUSA/Moore Threads | `vllm_omni/platforms/musa/platform.py:16` `MUSAOmniPlatform(OmniPlatform, MUSAPlatformBase)`（依赖 `vllm_musa` 包） | DiT attention 通过 `mate` 包提供 FLASH_ATTN，要求 compute capability ≥ 3.1（`platform.py:44-108`）；`supports_float64()` 返回 `False`（`platform.py:115-118`，MUSA 尚不支持 float64） | `GPUARWorker`（复用 GPU worker） | `DiffusionWorker` |

### 2. Quantization — AR 与 DiT 量化方法

`vllm_omni/quantization/factory.py:350` 的 `build_quant_config()` 是统一入口：支持字符串、dict（含 `method`/`quant_method` key）、per-component dict（如 `{"transformer": {"method": "fp8"}, "vae": None}`）、已构造的 `QuantizationConfig` 对象或 `None`。相较上游 vLLM 只面向单一稠密/MoE LLM 的量化注册表（`QUANTIZATION_METHODS`），Omni 的核心创新是**面向多阶段(stage)/多组件(component)模型的按前缀路由量化**——这是上游没有的能力，因为上游模型是单一 checkpoint，而 Omni 模型（Qwen3-Omni、BAGEL、GLM-Image 等）在同一进程内跑多个 stage，且 DiT stage 的 transformer/text_encoder/VAE 需要独立量化决策。

#### 2.1 Omni 专属量化方法（`_OVERRIDES` 注册表，`factory.py:135-143`）

| 方法 | 构建函数 file:line | 支持平台 | 用途 |
|---|---|---|---|
| `int8` | `factory.py:84` `_build_int8` → `int8_config.py:77` `DiffusionInt8Config(QuantizationConfig)` | CUDA + NPU（`int8_config.py:149-163` 按 `current_omni_platform.is_cuda()/is_npu()` 分派 `Int8LinearMethod`/`NPUInt8LinearMethod`，各自有 online(`Int8OnlineLinearMethod` `int8_config.py:385`)/offline(`int8_config.py:314`) 变体） | DiT transformer 的 W8A8 动态或离线 INT8 |
| `mxfp8` | `factory.py:91` `_build_mxfp8` → `mxfp8_config.py:70` `DiffusionMXFP8Config` | 仅 NPU（`NPUMxfp8LinearMethod` `mxfp8_config.py:297`, online 变体 `mxfp8_config.py:421`） | W8A8 MXFP8（Wan2.2 系列验证过） |
| `mxfp4` | `factory.py:98` `_build_mxfp4` → `mxfp4_config.py:92` `DiffusionMXFP4Config` | 仅 NPU（`NPUMxfp4LinearMethod` `mxfp4_config.py:177`）；ROCm 分支存在但仅限 gfx950/MI355X 且报 `NotImplementedError`（`mxfp4_config.py:161-166`） | W4A4 单尺度 MXFP4，仅支持 online |
| `mxfp4_dualscale` | `factory.py:105` `_build_mxfp4_dualscale` → `mxfp4_config.py:747` `DiffusionMXFP4DualScaleMixedConfig` | 仅 NPU（`NPUMxfp4DualScaleLinearMethod` `mxfp4_config.py:494`） | W4A4 双尺度 + 逐层 BF16 fallback（`ignored_layers`/`num_bf16_fallback_layers` 控制），offline 校准 `mul_scale` 精度最佳 |
| `inc`/`auto-round`/`auto_round` | `factory.py:121` `_build_inc` → `inc_config.py:52` `OmniINCConfig(INCConfig)` | 主要 CUDA/XPU（AutoRound checkpoint 驱动） | 预量化 W4A16 checkpoint |

此外 `vllm_omni/quantization/quack_fp8.py:101` `install_quack_fp8_patch()` 为 Blackwell 数据中心卡（`tcgen05` 指令集，`quack_fp8.py:18` `_is_quack_capable`）接入 NVIDIA `quack` CuteDSL FP8 GEMM 内核（带融合 bias epilogue），是 CUDA 专属的加速路径而非量化方法本身。

#### 2.2 Per-component 路由（`component_config.py:75` `ComponentQuantizationConfig(QuantizationConfig)`）

按前缀最长匹配将 `get_quant_method()` 分派到不同 stage/子模块的 `QuantizationConfig`（`component_config.py:1-9` 文档字符串示例：`{"transformer": fp8_config, "vae": None}`）。`resolve_encoder_quant_config()`（`component_config.py:32`）专门处理 ModelOpt 系checkpoint 只量化 Thinker LM、视觉/音频 encoder 保持 BF16 的情况（`PRE_QUANTIZED_METHODS` 集合，`component_config.py:27`）。

#### 2.3 与上游差异（DiT 量化是 Omni 的新增能力）

上游 vLLM 的量化框架服务于纯文本/纯视觉 LLM 的线性层；Omni 在 `docs/user_guide/quantization/overview.md:1-238` 中明确将模型分为三类并各自定义量化 scope：
- **纯 DiT 模型**（Qwen-Image、Wan2.2）：默认量化目标是 diffusion transformer，text encoder/VAE/scheduler 默认 BF16（`overview.md:32-48`）。
- **多阶段 Omni/TTS 模型**（Qwen3-Omni、Qwen3-TTS）：量化仅作用于 AR 语言模型 stage（thinker/LM），talker/audio encoder/code2wav 默认 BF16（`overview.md:50-65`）。
- **多阶段 DiT 模型**（BAGEL、GLM-Image）：必须显式挂到目标 stage，而非全局应用（`overview.md:67-81`）。

硬件支持矩阵（`overview.md:16-25`）显示 MXFP8/MXFP4 目前**仅 Ascend NPU** 支持，msModelSlim 也仅 Ascend；ROCm 仅 MXFP4(gfx950) 验证过。`quantized_kvcache.md:1-39` 描述的 **FP8 Flash Attention 运行时量化**（`diffusion_kv_cache_dtype`，区别于 `--kv-cache-dtype`）目前也**仅 NPU 后端**实现（`vllm_omni/platforms/npu/quant/kv_quant_npu.py:26` `is_quantized_kv_cache`，基于 QuaRot Hadamard 旋转 + `mindiesd`/`msmodelslim` 的 `fa_block_quant_preprocess`）。

`vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py`（CLI，见 `overview.md:157-196`）是量化质量回归工具：同 seed/prompt 对比 BF16 基线与量化候选的输出图像/视频帧，报告 `cosine_similarity`/`mae`/`psnr_db`/`relative_l2`/`avg_generation_time_s`/`max_peak_memory_mb`（`overview.md:198-238`）。另有 `vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py`、`merge_mxfp8_checkpoint.py` 用于合并离线量化 checkpoint 分片。

### 3. Attention 后端（`vllm_omni/attention/`）

不同于 `vllm_omni/diffusion/attention/backends/`（DiT 的 FA/SDPA/CUDNN/FlashInfer/SageAttn 后端注册表，见第1节 `DiffusionAttentionBackendEnum`），`vllm_omni/attention/` 目录是一个**模型特化的 AR decode 加速内核**：Fish-Speech 慢速 AR 模型的自定义 KV cache decode attention。

| 组件 | file:line | 说明 |
|---|---|---|
| Triton kernel | `vllm_omni/attention/fish_kvcache_triton.py:239` `fish_decode_kvcache_attn_triton` | 纯 Triton 实现的单 token query + paged KV cache decode attention（无 `vllm` 依赖），`is_available()`(`:22`)/`load_error()`(`:26`) 做懒加载探测 |
| 适用性判定 | `vllm_omni/attention/fish_kvcache_attn.py:56` `can_use_fish_kvcache_attn()` | 严格前置条件检查：`max_query_len==1`（纯 decode）、无 cascade/alibi/sliding window、`head_dim==128`、`block_size==16`、dtype ∈ {fp16, bf16}、tensor 均需 contiguous（`fish_kvcache_attn.py:56-110`）；`FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN=1024`(`:10`) 划分小/长序列两条路径；工作区缓存 `_WORKSPACE_CACHE`（线程安全，`:16-17`） |
| 安装/monkey-patch | `vllm_omni/attention/fish_kvcache_backend.py:351` `install_fish_kvcache_attn_backend(model)` | 仅对 `model_arch == "FishSpeechSlowARForConditionalGeneration"` 生效（`fish_kvcache_backend.py:37` `is_fish_kvcache_attn_active_for_model`）；通过 `types.MethodType` 替换每层 `layer.self_attn.attn.impl.forward`（`:373-405`），带 fallback 统计 `_FALLBACK_COUNTS`/`get_fish_kvcache_attn_stats()`（`:48-57`） |
| 开关 | env `VLLM_OMNI_FISH_KVCACHE_ATTN`（`fish_kvcache_attn.py:12-45`） | `required`/`1`/`true`/... 启用；`0`/`false`/... 禁用；`required` 时不可用会直接 `RuntimeError` |

### 4. Benchmarks & Profiler 工具清单

Omni 的基准工具分布在两处：`vllm_omni/benchmarks/`（打包进 wheel、被 `vllm bench serve --omni` CLI 挂载的运行时组件）与仓库顶层 `benchmarks/`（开发者手动运行的独立脚本，不随包分发）。

| 工具 | File:line | 测量内容 | CLI 示例 |
|---|---|---|---|
| Serve bench 入口（AR + omni） | `vllm_omni/benchmarks/serve.py:19` `main()` | 委托给 `vllm.benchmarks.serve.main_async`，注入 `--print-stage`/`extra_body` 的 stage 级指标开关（`should_request_stage_metrics`, `:27-30`） | `vllm bench serve --omni --backend openai-chat-omni --dataset-name random --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf`（`docs/cli/bench/serve.md:206-219`） |
| Patch：数据集/后端注册 | `vllm_omni/benchmarks/patch/patch.py`（1706 行，monkey-patch `vllm.benchmarks.datasets.get_samples`） | 注册 `daily-omni`/`seed-tts`/`sound-effect`/`ttsd`/`random-mm` 等数据集与 `openai-chat-omni`/`openai-image-edits-omni` 后端 | — |
| 指标计算/汇总 | `vllm_omni/benchmarks/metrics/metrics.py:1` (965 行，`_MULTIMODAL_BENCHMARK_FIELDS` 扩展 `BenchmarkMetrics` dataclass，`:17-24`) | 生成 audio TTFP/RTF/duration/frames 等多模态百分位指标；引用 `vllm_omni/metrics/definitions.py` | — |
| 音频连续性 | `vllm_omni/benchmarks/audio_continuity.py:35` `compute_continuity_stats` | 模拟播放器实时消费 PCM，检测流式音频"卡顿"（underrun）事件，配合 `audio_ttfp`/`audio_rtf` | — |
| Daily-Omni 视听推理评测 | `vllm_omni/benchmarks/data_modules/daily_omni_dataset.py:1`（1013 行）+ `daily_omni_eval.py:1`（417 行，选择题打分，对齐上游 `Lliar-liar/Daily-Omni` 评测脚本） | 684 视频 / 1197 选择题的视听推理 QA 准确率，按视频时长/类别拆分 | `--dataset-name daily-omni` |
| Seed-TTS zero-shot 数据/评测 | `vllm_omni/benchmarks/data_modules/seed_tts_dataset.py:1`（481 行）+ `seed_tts_eval.py:1`（835 行） | TTS WER（Whisper-large-v3 英文 / FunASR paraformer-zh 中文）、SIM（WavLM speaker embedding cosine）、UTMOS 预测 MOS | `--seed-tts-wer-eval` |
| Sound-effect / 对话 TTS 数据集 | `sound_effect_dataset.py:1`（128 行）/ `ttsd_dataset.py:1`（140 行） | 环境声合成 bench（无 SIM）/ 多说话人对话 TTS（`[S1]`/`[S2]` 标签） | — |
| Random 多模态请求生成 | `vllm_omni/benchmarks/data_modules/random_multi_modal_dataset.py:14` `process_audio` | 合成图像/视频/音频请求压测视觉/音频模型 | `--dataset-name random-mm --random-mm-bucket-config '{...}'` |
| Multi-Stage bench 展示 | `docs/cli/bench/serve.md:367-467` | `--print-stage` 输出各 stage（thinker/talker/code2wav）单独的 `stage_gen_time`/TTFC/TPOP/ICL | `vllm bench serve --omni --print-stage --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,ttfc,tpoc,icl` |
| Diffusion 在线服务 bench | `benchmarks/diffusion/diffusion_benchmark_serving.py:1`（1468 行，adapted from fastvideo） | T2I/I2I/T2V/I2V 在线服务延迟/吞吐（`/v1/chat/completions`、`/v1/images/edits`、`/v1/images/generations`、`/v1/videos`） | `python3 benchmarks/diffusion/diffusion_benchmark_serving.py --task t2v ...` |
| DiT attention 后端诊断 | `benchmarks/diffusion/bench_attention_backends.py:3`（328 行） | 逐后端(CUDNN_ATTN/FLASH_ATTN/SDPA子kernel)对比同一 attention shape 的延迟，定位 CUDNN_ATTN 性能异常 | `python benchmarks/diffusion/bench_attention_backends.py --preset hv15` |
| DiT 量化质量 | `benchmarks/diffusion/quantization_quality.py:5`（487 行） | BF16 vs 量化候选的 LPIPS 感知距离，产出可粘贴进 PR 的 Markdown 表 | `python benchmarks/diffusion/quantization_quality.py --model ... --quantization fp8` |
| TTS 通用 bench CLI | `benchmarks/tts/bench_tts.py:2`（327 行） | 基于 `model_configs.yaml` 的模型感知默认值，包一层 `vllm bench serve --omni` | `python benchmarks/tts/bench_tts.py --model Qwen/Qwen3-TTS-... --task voice_clone` |
| Fish-Speech 说话人缓存 bench | `benchmarks/fish-speech/bench_speaker_cache.py:1`（292 行） | 对比内联 ref_audio vs 已上传声音缓存的 TTFP 改善 | `python bench_speaker_cache.py --ref-audio ... --num-prompts 20` |
| MoT GEMM kernel 自动调优 | `benchmarks/kernels/mot_linear_benchmarks.py:3`（1015 行） | BAGEL 等 Mixture-of-Tokens 模型的 Triton GEMM kernel 配置搜索/落盘 | `python benchmarks/kernels/mot_linear_benchmarks.py --model ByteDance-Seed/BAGEL-7B-MoT --tune` |
| 文生图/图生图精度 bench | `benchmarks/accuracy/text_to_image/gbench.py`（927 行）/ `benchmarks/accuracy/image_to_image/gedit_bench.py`（787 行） | GenEval/GEdit-Bench 风格的自动化图像质量评测 | `python benchmarks/accuracy/text_to_image/run_gebench.py` |
| Torch Profiler（跨平台） | `vllm_omni/profiler/omni_torch_profiler.py:30` `OmniTorchProfilerWrapper(WorkerProfiler)` | 自定义 trace 命名（stage/rank）、后台 gzip 压缩、Excel 汇总表（summary/by_shape/by_stack sheet, `:343-384`）、跨后端 memory history/snapshot（CUDA/NPU/XPU/MUSA, `:288-309`） | `vllm serve ... --profiler torch` |
| Diffusion pipeline profiler | `vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py:81` `DiffusionPipelineProfilerMixin` | 按路径字符串 wrap pipeline 内部方法（`wrap_methods_by_paths`, `:64`）逐步统计 DiT 各子模块耗时 | — |

平台专属 profiler override：`vllm_omni/platforms/npu/profiler.py`（`NPUTorchProfilerWrapper`，见 `platform.py:189-190`）与 `vllm_omni/platforms/xpu/profiler.py`（`XPUTorchProfilerWrapper`，`platform.py:102-104`），分别处理 NPU/XPU profiler activity 与内存快照 API 差异。

### 5. Metrics — Omni 专属 Prometheus 指标

设计文档 `docs/design/metrics.md` 详述架构：Omni 在单进程内跑多个 engine（stage × replica），需要额外一层 pipeline 级 + 模态级 + 跨 stage 传输级指标，全部使用 `vllm:omni_` 前缀以与上游 `vllm:*`（reshape 后带 `stage`+`replica` label）区分（`metrics.md:23`）。

| 分类 | 指标 family | file:line | 说明 |
|---|---|---|---|
| Pipeline (4个) | `vllm:omni_num_requests_running/waiting`, `vllm:omni_requests_success_total`, `vllm:omni_e2e_request_latency_s` | `vllm_omni/metrics/prometheus.py:42` `OmniPrometheusMetrics`；计数器 `prometheus.py:100` `OmniRequestCounter` | 跨全部 stage 的端到端请求状态（`metrics.md:159-162`） |
| Audio (7个) | `vllm:omni_audio_ttfp_s/duration_s/rtf/frames_total/underrun_s/continuity_ok_total/skipped_requests_total` | `vllm_omni/metrics/modality.py:80` `OmniModalityMetrics`；观测函数 `observe_modality_at_finalize`(`:189`)/`observe_audio_first_packet`(`:249`)/`observe_audio_streaming_finalize`(`:271`) | 音频专属 SLO：首包延迟、real-time factor（`compute_audio_rtf`, `vllm_omni/metrics/definitions.py:256`）、连续性欠载（复用 `benchmarks/audio_continuity.py` 的播放器模拟） |
| 跨 stage 传输 (4个) | `vllm:omni_transfer_size_bytes/tx_s/rx_s/in_flight_s` | `vllm_omni/metrics/transfer.py:58` `OmniTransferMetrics` | 每次物理传输 hop（chunk 级，非累计）的序列化/网络在途时间，归因 e2e 延迟与 stage `gen_time` 之和之间的 gap |
| LLM stage 级（wrap 上游 `vllm:*`） | `vllm_omni/metrics/stat_logger.py:182` `OmniPrometheusStatLogger(PrometheusStatLogger)` | 用 `_RelabelGauge/_RelabelCounter/_RelabelHistogram`（`stat_logger.py:135-143`）把上游硬编码的 `engine` label 重塑为 `stage`+`replica`，覆盖约 65 个上游 family（TTFT/ITL/TPOT/KV cache 用量等） |
| 日志向聚合（非 Prometheus） | `vllm_omni/metrics/stats.py:136` `OrchestratorAggregator` | 打印 per-request/per-stage/per-transfer 详细表格到 INFO 日志，供开发调试 |

节流：`OmniSchedulerMixin.make_stats()`（`vllm_omni/core/sched/omni_scheduler_mixin.py`，`metrics.md:145-149`）把每个 scheduler 的 stats 上报频率限制在 1Hz/replica，避免 ZMQ 序列化开销。全部指标受 `--log-stats` 门控（默认关闭，`metrics.md:131-143`）。

### 6. Sleep Mode — 空闲时权重卸载释放显存

`docs/features/sleep_mode.md` 描述该特性继承自 [vLLM 原生 Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html)，并扩展了 **Omni ACK 协议**以支持多阶段流水线和异构硬件（NVIDIA/AMD/Intel/Huawei）的协同睡眠/唤醒。

- **两级休眠**（`sleep_mode.md:14-20`）：Level 1「权重卸载到 Host RAM」（快速 DMA 恢复）；Level 2「完全解除映射」（VRAM scavenging，恢复较慢但可回收 95%+ 显存）。
- **AR/LLM worker 实现**：`vllm_omni/worker/base.py:191` `sleep(level)` 与 `:215` `wake_up(tags)`，基于 vLLM 的 `CuMemAllocator`（`vllm.device_allocator.cumem`）；`level==1` 只卸载 `("weights",)` tag，`level==2` 卸载全部（`base.py:200`）。`_maybe_get_memory_pool_context`（`base.py:172`）在 sleep mode 启用时用 `allocator.use_memory_pool(tag=tag)` 包裹权重/KV分配。
- **握手协议**：`handle_sleep_task(task: OmniSleepTask) -> OmniACK`（`worker/base.py:225`）计算 `total_freed`（跨 rank `all_reduce`，`base.py:239-245`）与 `rank_residual_gib`，装配 `OmniACK`（定义于 `vllm_omni/diffusion/data.py:1455`，字段 `task_id`/`status`/`stage_id`/`rank`/`freed_bytes`/`metadata`）。
- **DiT worker 实现**：`vllm_omni/diffusion/worker/diffusion_worker.py:522` `sleep(level)` / `:570` `wake_up(tags)` / `:595` `handle_sleep_task` / `:652` `handle_wake_task`（另有 stub 变体在 `:1183/1199/1216/1219` 用于非分布式简化路径），DiT stage 无论 TP 大小只在 rank 0 汇报单条 ACK（`vllm_omni/entrypoints/async_omni.py:924` 注释）。
- **顶层编排 API**：`vllm_omni/entrypoints/async_omni.py:913` `async def sleep(stage_ids, level, mode)` 通过 `collective_rpc(method="handle_sleep_task", ...)` 广播到目标 stage 的所有 worker，用 `event_resolver.watch_task(task_id, expected_count=total_workers)` 等待全部 ACK 齐集（`:919-942`）；`:950` `wake_up()` 目前**不支持 level=2 之后的唤醒**（权重已被丢弃，尚未实现从磁盘重载，`:953-959` 直接 `raise NotImplementedError`）。
- **HTTP 接口**：`POST /v1/omni/sleep {"stage_ids":[...], "level":2}`、`POST /v1/omni/wakeup {"stage_ids":[...]}`（`sleep_mode.md:106-155`）。
- **CLI 开关**：`vllm serve ... --omni --enable-sleep-mode`（`sleep_mode.md:74-92`）。

### 7. ComfyUI 集成

`docs/features/comfyui.md` 描述 ComfyUI 集成是构建在 Omni **在线服务 API** 之上的前端插件（自定义节点包），可指向本地或远程运行的 `vllm serve --omni` 服务，本身不引入新的推理路径。

- 安装方式：将 `apps/ComfyUI-vLLM-Omni` 拷贝到 ComfyUI 的 `custom_nodes` 子目录（`comfyui.md:19`）。
- 节点实现：`apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/nodes.py`（736 行），核心类：`_VLLMOmniGenerateBase`(`:24`)、`VLLMOmniGenerateImage`(`:42`，T2I/I2I)、`VLLMOmniGenerateVideo`(`:151`，T2V/I2V)、`VLLMOmniUnderstanding`(`:230`，多模态理解→文本/音频)、`VLLMOmniTTS`(`:344`)、`VLLMOmniVoiceClone`(`:406`)、`VLLMOmniARSampling`/`VLLMOmniDiffusionSampling`/`VLLMOmniSamplingParamsList`(`:475/519/603`，AR 与 Diffusion 采样参数节点可拼接成多阶段采样参数列表)、`VLLMOmniRemoteLoRA`(`:637`)、`VLLMOmniQwenTTSParams`/`VLLMOmniWanParams`(`:683/710`，模型专属参数节点)。
- 功能边界：节点 UI 严格对齐在线服务接口能力，"不能提供比接口更多的功能"（`comfyui.md:56-57`）；对于像 BAGEL 这类只暴露单一 stage 采样参数的模型，会被当作单阶段模型处理（`comfyui.md:66`）。
- 附带 example workflow JSON（图像/视频生成、多模态理解、TTS、服务链式调用），前端资源 `apps/ComfyUI-vLLM-Omni/web/main.js`。

### 8. Experimental 特性一览（`vllm_omni/experimental/`）

`vllm_omni/experimental/__init__.py:2-8` 明确声明该目录下模块"处于活跃开发中，API/配置面/默认值可能随时无预警变更"。

| 子模块 | file:line | 内容 |
|---|---|---|
| AR-Diffusion Engine | `vllm_omni/experimental/ar_diffusion/engine.py:16` `ARDiffusionEngine(DiffusionEngine)` | 面向 AR/分块因果 diffusion 模型（world model、AR-DiT，例如 DreamZero）的引擎级 KV cache 管理；复用 vLLM 的分页 KV 栈（`KVCacheManager`/`BlockPool`/`BlockTables`）作为库，而非在每个模型里手写；通过 `OmniDiffusionConfig.engine_backend = "ar_diffusion"` 按模型选择性启用（`engine.py:26`）。子目录 `kv_cache/`（`config.py`/`manager.py`/`paged_attention.py`/`paged.py`/`state.py`）与 `runner.py`（`ARDiffusionModelRunner`）。 |
| Full-duplex 交互框架 | `vllm_omni/experimental/fullduplex/README.md:1-20` | 模型无关的实时全双工（流式输入/流式输出）交互框架：`core/`（`DuplexRuntime` 事件循环 + epoch barge-in、`DuplexSession`、`DuplexAdapter` ABC、协议定义，仅 `core/` 跨模型共享）+ `joyvl/`（JoyVL 具体实现：`adapter.py`、`decision/`「speak/silence/delegate 策略与 prompt」、`memory/`「三级摘要记忆 `InteractionBrain`」、`serving/`「OpenAI 兼容 HTTP 编排」、`bridges/`「模型后端与委派」）。README 明确当前可运行的服务路径是 `joyvl/serving/` 直接驱动 `decision/`+`memory/`，`core/`+`adapter.py` 只是框架示范（面向如 MiniCPM-o 这类融合音频模型的未来接入）。 |


---

## §11. 瓶颈 / 风险总表 + 组件-文件主索引

> 本章对标 MMProcessor.md 的 §5「Bottleneck Summary」与末尾「Index」：把前面各章挖出的**半成品 / 约束 / 维护者需要盯的点**按优先级汇总，再给一张跨章的组件→file:line 速查表。作为 release 协调视角，这里的每一条都可直接转成 issue / release note 风险项。

### 11.1 维护者需要盯的「半接线 / 约束 / 文档漂移」总表

图例：🔴 会影响正确性/发布口径　🟡 能力缺口/体验　🟢 文档或次要

| 级别 | 事项 | 现状 | 证据 file:line | 建议动作 |
| --- | --- | --- | --- | --- |
| 🔴 | **Composable Parallel 只打通 5/12 轴** | `tp/dp/pp/ep/stage_replica` 已 wired；`sp_ulysses/sp_ring/cfg/vae_pp/hsdp/stage_pp/cp` 声明即抛 `AxisTranslationError` | `vllm_omni/config/composable_parallel/spec.py:35-55`、`translator.py:50` | strategy YAML 文档需显式标注"DiT 专属轴仍只能走 `StageDeployConfig` 字段"，避免用户误配 |
| 🔴 | **Ray 多机 stage 放置未接入主流程** | `create_placement_group`/`start_ray_actor` 原语已写好但仓库内无调用点；实际生效的只有 pinned-memory 兼容与 SHM 阈值联动 | `vllm_omni/distributed/ray_utils/utils.py:110-181`（无 caller）；`__init__.py:4-9` 未导出 | release note 不应宣称"Ray 多机放置可用"；`--worker-backend ray` 当前主要是环境适配层 |
| 🔴 | **`wake_up` 不支持 level=2 之后唤醒** | level-2 权重已丢弃，尚未实现从磁盘重载，直接 `raise NotImplementedError` | `vllm_omni/entrypoints/async_omni.py:953-959` | sleep-mode 文档/接口需标注 level-2 为单向；或补重载路径 |
| 🟡 | **Diffusion LoRA 无 CLI** | `lora_path/lora_scale/max_cpu_loras` 只在 `OmniDiffusionConfig`（Python API），未镜像到 `OmniDiffusionEngineArgs` | `vllm_omni/diffusion/data.py:642-644`；`arg_utils.py` 无 lora 字段 | `vllm serve` 用户只能靠 Python/配置文件；补 CLI flag |
| 🟡 | **量化多为 NPU-only** | MXFP8/MXFP4/MXFP4-dualscale/量化 KV-cache 目前主要落在 Ascend NPU；ROCm 仅 MXFP4(gfx950) | `vllm_omni/quantization/factory.py:135-143`；`quantized_kvcache.md`；`platforms/npu/quant/kv_quant_npu.py:26` | 硬件支持矩阵要在 release note 明确，避免 CUDA 用户误期待 |
| 🟡 | **Mori 连接器跨节点未支持** | AMD Mori TE 仅节点内 GPU-to-GPU（RDMA/XGMI），跨节点见 issue #1742 | `connectors/mori_transfer_engine_connector.py:110` | 多机 AMD 部署需回退 Mooncake TE |
| 🟡 | **Diffusion 连续批处理仍 experimental** | `step_execution=True` 路径不支持 `cache_backend`、KV-transfer 等 request-mode 特性；仅 QwenImage 实现四段式契约 | `docs/design/feature/diffusion_continuous_batching.md:3-5`；`diffusion_model_runner.py:681-682` | 默认仍走 request-level batching；连续批处理按实验特性标注 |
| 🟡 | **Omni 张量前缀缓存仅单 kv-cache group** | 多 group 会 warning 并只用第一组 | `vllm_omni/core/prefix_cache.py:690-694` | hybrid KV cache 模型上 prefix cache 收益受限 |
| 🟡 | **仅 D2H2D，无纯 D2D** | 所有连接器走 Device→Host→Device；NCCL/UCX/IPC 纯 D2D 在路线图 | `docs/design/feature/disaggregated_inference.md:108-112` | 跨 stage 大张量传输仍有 host 往返开销 |
| 🟢 | **dit_module.md 文档漂移** | 文档写 `diffusion/worker/gpu_worker.py`+`GPUWorker`，实际是 `diffusion_worker.py`+`DiffusionWorker` | `vllm_omni/diffusion/worker/diffusion_worker.py:182` | 更新设计文档 |
| 🟢 | **`distributed_executor_backend` 仅 `mp`** | diffusion executor 的 `ray`/`external_launcher` 未实现 | `vllm_omni/diffusion/executor/abstract.py:37-44` | 文档标注 |

### 11.2 CPU / GPU 时间轴心智模型（Qwen3-Omni 语音链路，对标 MMProcessor §5 timeline）

```
请求到达
  │
  ├─[CPU] Entrypoint 解析 + InputProcessor (stage-0)        §8 §3
  │        chat template / 多模态加载 / mm_processor
  ▼
  ├─[GPU] Thinker(AR prefill+decode)  ← KV cache 主压力      §3
  │        │ hidden/KV  ──OmniConnector(D2H2D)──▶            §7
  ▼        ▼
  ├─[GPU] Talker(AR, multi-codebook MTP)                     §3
  │        │ codec tokens ──connector──▶                     §7
  ▼        ▼
  ├─[GPU] Code2Wav(生成/扩散 vocoder)  ← 算力密集             §4 §5
  │        async_chunk 使三段重叠, TTFP ↓~92%                §5
  ▼
  └─[CPU] 输出装配 + 序列化 (base64音频 / SSE / WS 二进制帧)  §8
```

关键结论：AR 段是**访存密集 + KV 显存主压力**，DiT/vocoder 段是**算力密集**；二者混在同一进程会互相拖调度节拍，这正是 §2 编排解耦 + §6 per-stage 并行 + §7 E/P/D/G 分离部署三者存在的根因。`async_chunk`（§5）是把这条串行链路重叠起来的关键旋钮。

### 11.3 组件 → 文件主索引（跨章速查）

| 子系统 | 核心类 / 入口 | File:line | 章 |
| --- | --- | --- | --- |
| 引擎瘦代理 | `AsyncOmniEngine` | `vllm_omni/engine/async_omni_engine.py:191` | §2 |
| 多 stage 编排 | `Orchestrator` | `vllm_omni/engine/orchestrator.py:203` | §2 |
| stage 多副本池 | `StagePool` | `vllm_omni/engine/stage_pool.py:48` | §2 |
| CFG 伴随追踪 | `CfgCompanionTracker` | `vllm_omni/engine/cfg_companion_tracker.py:16` | §2 |
| AR 调度器 | `OmniARScheduler` | `vllm_omni/core/sched/omni_ar_scheduler.py:50` | §3 |
| 一步生成调度器 | `OmniGenerationScheduler` | `vllm_omni/core/sched/omni_generation_scheduler.py:42` | §3 |
| 张量前缀缓存 | `OmniTensorPrefixCache` | `vllm_omni/core/prefix_cache.py:33` | §3 |
| AR runner | `GPUARModelRunner` | `vllm_omni/worker/gpu_ar_model_runner.py:289` | §3 |
| 扩散引擎 | `DiffusionEngine` | `vllm_omni/diffusion/diffusion_engine.py:134` | §4 |
| 扩散调度器 | `RequestScheduler` / `StepScheduler` | `vllm_omni/diffusion/sched/{request_scheduler.py:19, step_scheduler.py:30}` | §4 |
| 扩散 runner | `DiffusionModelRunner` | `vllm_omni/diffusion/worker/diffusion_model_runner.py:95` | §4 |
| 步执行契约 | `SupportsStepExecution` | `vllm_omni/diffusion/models/interface.py:46` | §4 |
| 缓存后端选择 | `get_cache_backend` | `vllm_omni/diffusion/cache/selector.py:11` | §5 |
| Cache-DiT / TeaCache | `CacheDiTBackend` / `TeaCacheHook` | `cache/cache_dit_backend.py:1024` / `cache/teacache/hook.py:30` | §5 |
| 扩散 LoRA | `DiffusionLoRAManager` | `vllm_omni/diffusion/lora/manager.py:36` | §5 |
| 扩散模型注册表 | `_DIFFUSION_MODELS` | `vllm_omni/diffusion/registry.py:22` | §5 |
| 可组合并行翻译 | `translate_strategy_stack` | `vllm_omni/config/composable_parallel/translator.py:312` | §6 |
| DiT 并行组 | `initialize_model_parallel` | `vllm_omni/diffusion/distributed/parallel_state.py:791` | §6 |
| CFG 并行 mixin | `CFGParallelMixin` | `vllm_omni/diffusion/distributed/cfg_parallel.py:57` | §6 |
| VAE patch 并行 | `VaePatchParallelism` | `vllm_omni/diffusion/distributed/vae_patch_parallel.py:348` | §6 |
| 连接器基类 | `OmniConnectorBase` | `vllm_omni/distributed/omni_connectors/connectors/base.py:12` | §7 |
| 连接器工厂 | `OmniConnectorFactory` | `vllm_omni/distributed/omni_connectors/factory.py:24` | §7 |
| 成员协调器 | `OmniCoordinator` | `vllm_omni/distributed/omni_coordinator/omni_coordinator.py:19` | §7 |
| 负载均衡 | `LoadBalancingPolicy` | `vllm_omni/distributed/omni_coordinator/load_balancer.py:27` | §7 |
| OpenAI 服务组装 | `omni_run_server_worker` | `vllm_omni/entrypoints/openai/api_server.py:460` | §8 |
| 统一在线入口 | `AsyncOmni.generate` | `vllm_omni/entrypoints/async_omni.py:259` | §8 |
| Chat serving | `OmniOpenAIServingChat` | `vllm_omni/entrypoints/openai/serving_chat.py:287` | §8 |
| Pipeline 注册表 | `OMNI_PIPELINES` | `vllm_omni/config/pipeline_registry.py:96` | §9 |
| 配置装配厂 | `StageConfigFactory` | `vllm_omni/config/config_factory.py:47` | §9 |
| stage 部署配置 | `StageDeployConfig` | `vllm_omni/config/stage_config.py:302` | §9 |
| upstream 补丁 | `vllm_omni/patch.py` | `vllm_omni/patch.py:1`（486 行） | §9 |
| 平台抽象 | `OmniPlatform` | `vllm_omni/platforms/interface.py:31` | §10 |
| 量化工厂 | `build_quant_config` | `vllm_omni/quantization/factory.py:350` | §10 |
| Omni Prometheus 指标 | `OmniPrometheusMetrics` | `vllm_omni/metrics/prometheus.py:42` | §10 |
| Sleep 编排 | `AsyncOmni.sleep` | `vllm_omni/entrypoints/async_omni.py:913` | §10 |

### 11.4 与本板块已有深潜笔记的对照（本页是 umbrella，细节下钻）

| 本页章 | 已有深潜笔记（docs/vllm-omni/） |
| --- | --- |
| §1 全局架构 | [架构地图](architecture-map.md)、[多模态运行时综述](multimodal-runtime-overview.md)、[组件与请求流转](components-request-flow.md) |
| §2 编排 | [Orchestrator：多 stage 编排核心](engine-orchestrator.md)、[请求完整生命周期](request-lifecycle-end-to-end.md)、[Thinker→Talker 交接](thinker-talker-handoff.md) |
| §3 AR / 前缀缓存 | [worker 类层次](worker-class-hierarchy.md)、[NPU 前缀缓存缺兜底案例](npu-prefix-cache-missing.md)、[talker_mtp 图安全](talker-mtp-graph-safety.md) |
| §4/§5 扩散 | [Diffusion pipeline 内部](diffusion-pipeline-internals.md)、[Diffusion 注意力后端](diffusion-attention-backend.md)、[BAGEL 模型解剖](bagel-model-anatomy.md) |
| §7 连接器 | [跨 stage 数据面：连接器与 KV 传输](distributed-connectors-kv.md) |
| §8 服务 | [TTS 服务链路](tts-serving-path.md)、[Qwen3-TTS 端到端](qwen3-tts-end-to-end.md) |
| §10 平台/量化 | [platforms/npu 架构导读](npu-platform-architecture.md)、[平台解耦](platform-decoupling.md)、[量化全景](quantization-overview.md)、[图模式在 runner 里的实现](npu-gpu-graph-in-runner.md) |

> 分析方法：tag `v0.25.0rc1`（commit `d3c47ef`）在独立 git worktree 只读检出，9 个特性簇并行深潜、逐条回到 file:line 核实后汇总。file:line 为该快照，跨版本会漂，引用前请复核对应符号是否仍存在。
