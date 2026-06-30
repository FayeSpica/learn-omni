---
tags:
  - vllm
  - vllm-omni
  - vllm-ascend
  - 调试
  - 断点
  - NPU
  - GPU
  - 请求流转
---

# 断点点位地图：启动期 + 请求期（GPU/NPU × vllm / vllm-omni / vllm-ascend）

> 一个问题：**调试 omni 时该在哪些 `file:line` 下断点？每个点跑在哪个进程？GPU 和 NPU 在哪分叉？**
>
> 本文是「**断在哪**」的速查地图，覆盖**启动期**（serve 拉起到 ready）和**请求期**（curl 到返回）两条线。配套的「**怎么 attach**」（多进程 debugpy 远程附着、VSCode 配置、NPU 注意事项）见 [在 VSCode 里远程调试 Ascend 容器](debug-ascend-remote.md)。流程全貌见 [从 curl 到返回的请求生命周期](request-lifecycle-end-to-end.md)。行号对照 `/Users/fayespica/git/vllm_omni/{vllm,vllm-omni,vllm-ascend}`，随版本漂移。

## 第一原则：先确认进程，再下断点

断点抓不到，**九成是断在了错误的进程**。omni 是多进程嵌套结构，下断点前先想清楚目标代码跑在谁身上：

```
omni main/API 进程
├─ 启动期 A–C（CLI / build / 平台解析）
├─ 请求期 A/B（serving / 编排入口）+ F 尾段（detok / 响应）
└─ Orchestrator 线程（主进程内的后台 asyncio 线程；启动期拉 stage，请求期路由）
     └─ stage 子进程  = vLLM EngineCoreProc（每个 stage / replica 一个）
          ├─ 启动期 F–H（EngineCore 初始化 / 握手 / busy loop）
          ├─ 请求期 C/D（scheduler）+（TP=1 时）E forward/sample
          └─ Worker_TP* 子进程（仅 TP>1）
               └─ 启动期 G（init_device/load_model/capture）+ 请求期 E（forward/attention/sample）
```

| 你要断的东西 | attach 哪个进程 |
|---|---|
| CLI / serving / 编排路由 / 输出回流 | **主进程** |
| stage 调度、stage 核握手 | **目标 stage 子进程** |
| `init_device` / `load_model` / `forward` / `attention` / `capture` / `sample` | TP=1 → **stage 子进程**；TP>1 → **Worker_TP0 子进程** |

> 对照本仓库 `qwen3_omni_moe.yaml`：thinker(stage0) **TP=2** → forward 在 `Worker_TP0`；talker(stage1)/code2wav(stage2) TP=1 → forward 在 stage 核本体。附着机制（按 stage/rank 门控的 `debugpy.listen`）见 [debug-ascend-remote §4](debug-ascend-remote.md)。

---

## 一、启动期点位（serve → ready）

!!! note "omni 不走 `vllm serve`"
    omni 有自己的 CLI（`vllm-omni`，`OmniServeCommand`，console script 注册）。但**每个 stage 子进程内部跑的就是 vLLM 原生 `EngineCoreProc` 启动路径**——所以「vllm serve 的启动」= omni 每个 stage 子进程里发生的事（下表 F–H）。

### A · CLI / serve（主进程）

| 点位 | file:line | owner |
|---|---|---|
| omni CLI 入口 | omni `entrypoints/cli/main.py:9` `main` | omni |
| serve 派发 | omni `entrypoints/cli/serve.py:86` `OmniServeCommand.cmd` | omni |
| 平台兜底替换 | omni `entrypoints/cli/serve.py:46` `_ensure_vllm_platform` | omni |

### B · 引擎构建 + stage 配置（主进程）

| 点位 | file:line | owner |
|---|---|---|
| server worker | omni `entrypoints/openai/api_server.py:459` `omni_run_server_worker` | omni |
| **读 pipeline/stage 拓扑** | omni `api_server.py:629` `build_async_omni_from_stage_config` | omni |
| 建 AsyncOmni → AsyncOmniEngine | omni `api_server.py:687` → `omni_base.py:173` | omni |

### C · 平台解析（懒加载，首次访问触发）

| 点位 | file:line | GPU / NPU |
|---|---|---|
| 解析 omni 平台类 | omni `platforms/__init__.py:133` `resolve_current_omni_platform_cls_qualname` → `:181` 实例化 | `CudaOmniPlatform` / `NPUOmniPlatform`（后者继承 vllm-ascend `NPUPlatform`） |
| (NPU)ascend 插件注册 | vllm-ascend `__init__.py:40` `register`（import 期经 `vllm.platform_plugins` entry-point，`setup.py:541`） | 仅 NPU |

### D · Orchestrator 线程（主 → 编排线程）

| 点位 | file:line | owner |
|---|---|---|
| spawn 后台线程 | omni `async_omni_engine.py:304/313` `threading.Thread(_bootstrap_orchestrator)` | omni |
| 建 event loop + 初始化 stages | omni `async_omni_engine.py:400` `_bootstrap_orchestrator` / `:411` `_initialize_stages` | omni |

### E · 拉起 stage 子进程（编排线程 → stage 子进程）

| 点位 | file:line | owner |
|---|---|---|
| stage 启动编排 | omni `engine/stage_runtime.py:223` `StageRuntime.initialize` / `:528` `_initialize_local_llm_replica` / `:562` `launch_stage_replica` | omni |
| **spawn 子进程** | omni `engine/stage_engine_core_proc_manager.py:159` `proc.start()`（target=`run_stage_core`） | omni |

### F · stage 子进程内 = vLLM 启动主体（★"vllm serve"★）

| 点位 | file:line | owner | 进程 |
|---|---|---|---|
| stage 核入口 | omni `engine/stage_engine_core_proc.py:55` `run_stage_core` / `:127` `StageEngineCoreProc(...)` | omni 壳 | stage |
| **EngineCore 初始化** | vllm `v1/engine/core.py:900` `EngineCoreProc.__init__` → `:99` `EngineCore.__init__` → `:123` 建 Executor | **vllm** | stage |
| (TP>1)spawn worker | vllm `v1/executor/multiproc_executor.py:182/702` `make_worker_process`/`proc.start` → `worker_main:807` | vllm | →worker |

### G · 设备 / 权重 / KV / 图（GPU·NPU 分叉最深）

| 想看什么 | GPU | NPU（vllm-ascend） | 进程 |
|---|---|---|---|
| **设备初始化**（最早分叉） | vllm `gpu_worker.py:250` `init_device` / `:322` 分布式环境 | vllm-ascend `worker/worker.py:450` `init_device`（`:393 _init_device`） | stage/worker |
| **权重加载**（每 stage 只加载自己那份） | vllm `gpu_model_runner.py:5155` `load_model` / `:5176` `model_loader.load_model` | vllm-ascend `worker/model_runner_v1.py:253`（`_torch_cuda_wrapper` 抹平 CUDA/NPU） | stage/worker |
| KV profiling/分配 | vllm `core.py:239` `_initialize_kv_caches` / `:283` `determine_available_memory` / `gpu_worker.py:606` `initialize_kv_cache` | 同结构，昇腾显存 API | stage+worker |
| **图捕获** | vllm `gpu_worker.py:649` `_dummy_run` / `:658` `capture_model`（CUDAGraph） | vllm-ascend `compilation/acl_graph.py` `ACLGraphWrapper`（`torch.npu.NPUGraph`） | worker |

### H · ready 握手（stage → 编排 → 主）

| 点位 | file:line | 进程 |
|---|---|---|
| stage 发 READY | vllm `core.py:1100` `_perform_handshake`；omni `stage_engine_core_proc.py:139` 完成 / `:169` `run_busy_loop` | stage |
| 编排确认全 stage 起来 | omni `async_omni_engine.py:314` `_wait_for_orchestrator_init` / `:326` `"Orchestrator ready with N stages"` | 主 |
| **HTTP 监听 = ready** | omni `api_server.py:557` `serve_http` | 主 |

---

## 二、请求期点位（curl → 返回）

阶段号对齐 [请求生命周期文档](request-lifecycle-end-to-end.md) 的 A–F。

### A/B · 入口与编排（主进程，设备无关）

| 想看什么 | 点位 |
|---|---|
| 请求解析/多模态抽取 | omni `entrypoints/openai/serving_chat.py` `create_chat_completion` |
| omni 被当 vllm 引擎调 | vllm `…/chat_completion/serving.py:358` `engine_client.generate` |
| tokenize / `additional_information` 注入 | omni `engine/stage_init_utils.py:53` `build_stage0_input_processor` |
| **请求进编排 / stage 路由** | omni `engine/orchestrator.py:424` `_handle_add_request` |

### C · 进 stage 的 EngineCore（ZMQ 边界，vllm 原生）

| 想看什么 | 点位 | 进程 |
|---|---|---|
| ZMQ 发送（请求出主进程） | vllm `v1/engine/core_client.py:1121` `add_request_async` | 主 |
| 请求落到 stage 核 | vllm `v1/engine/core.py:372` `EngineCore.add_request` | stage |
| **每一拍迭代入口** | vllm `v1/engine/core.py:479` `EngineCore.step` | stage |
| ZMQ 收回（输出回主） | vllm `v1/engine/core_client.py:1005` `process_outputs_socket` | 主 |

### D · 调度 → worker（omni 扩展）

| 想看什么 | 点位 | 进程 |
|---|---|---|
| 批次构成 / prefill·decode / KV 反压 | omni `core/sched/omni_ar_scheduler.py` `schedule` | stage |
| worker 工厂选谁 | omni `engine/stage_init_utils.py:150` `resolve_worker_cls` | 主(启动期) |
| **执行入口** | GPU omni `worker/gpu_ar_worker.py` `execute_model` / NPU omni `platforms/npu/worker/npu_ar_worker.py` | stage/worker |

### E · forward / attention / 图 / sample（GPU·NPU 分叉）

| 想看什么 | GPU | NPU（vllm-ascend） | 进程 |
|---|---|---|---|
| **模型 forward** | vllm `gpu_model_runner.py:3793` `self.model(...)` | vllm-ascend `worker/model_runner_v1.py:253` 区 | worker |
| 注意力 kernel | vllm FA/SDPA backend `forward` | vllm-ascend `attention/attention_v1.py:1095/1115` `npu_fused_infer_attention_score[_v2]` | worker |
| 采样 | vllm `gpu_model_runner.py:4435` `sample_tokens`/`_sample` | vllm-ascend `AscendSampler` | worker |
| 图（捕获期勿单步！） | CUDAGraph wrapper | vllm-ascend `compilation/acl_graph.py` replay | worker |

### F · 输出组装 → 跨 stage → 响应（主进程，设备无关）

| 想看什么 | 点位 | 进程 |
|---|---|---|
| runner 产出（hidden/codes 走 wire） | omni `worker/gpu_ar_model_runner.py` `OmniModelRunnerOutput` 组装 | worker/stage |
| **stage 输出回编排 / 转下一站** | omni `orchestrator.py:632` `_orchestration_output_handler` / `:1158` `_forward_to_next_stage` | 主 |
| 跨 stage 打包 OmniPayload | omni `model_executor/stage_input_processors/qwen3_omni.py` `thinker2talker_*` | 主/连接器 |
| 多模态累积 | omni `engine/output_processor.py` `MultimodalOutputProcessor` | 主 |
| 文本回流 / detokenize | vllm `v1/engine/async_llm.py:656` `output_handler` → `detokenizer.py` | 主 |
| 最终 yield | omni `entrypoints/async_omni.py:249` `generate` | 主 |

---

## 三、GPU / NPU 分叉速查

| 维度 | GPU | NPU | 启动期/请求期分叉点 |
|---|---|---|---|
| 平台类 | `CudaOmniPlatform` | `NPUOmniPlatform`(←vllm-ascend `NPUPlatform`) | 启动 C |
| Worker 基座 | vllm `GPUWorker` | vllm-ascend `NPUWorker` | 启动 G / 请求 D |
| 设备初始化 | `gpu_worker.py:250` | vllm-ascend `worker.py:450` | **启动 G 第一行就分家** |
| 权重加载 | `gpu_model_runner.py:5155` | vllm-ascend `model_runner_v1.py:253` | 启动 G |
| 注意力 kernel | FA/SDPA | `npu_fused_infer_attention_score` | 请求 E |
| 图捕获 | CUDAGraph | `ACLGraphWrapper` | 启动 G / 请求 E |
| 采样 | vllm Sampler | `AscendSampler` | 请求 E |
| 同步/显存/通信 | `torch.cuda.*` / NCCL | `torch.npu.*` / HCCL(`NPUCommunicator`) | 启动 G / 请求 E |

> 规律一致：**CLI / 编排 / EngineCore 框架 / 输出组装全是设备无关的同一份代码；分叉只在 worker 基座以下的设备叶子。** 启动期分叉比请求期更早——`init_device` 一行就分家。

---

## 四、两条最小走查

### 启动一次，8 个断点

1. omni `serve.py:86` `OmniServeCommand.cmd` — 确认走 omni CLI ✅主
2. omni `api_server.py:629` `build_async_omni_from_stage_config` — stage 拓扑对不对 ✅主
3. omni `stage_engine_core_proc_manager.py:159` `proc.start` — 每个 stage 子进程拉起 ✅编排
4. omni `stage_engine_core_proc.py:127` `StageEngineCoreProc(...)` — 进 stage 核 ✅stage
5. GPU `gpu_worker.py:250` `init_device` / NPU vllm-ascend `worker.py:450` — 设备初始化分叉 ✅stage/worker
6. `load_model`（表 G） — 权重加载（每 stage 只加载自己那份）✅stage/worker
7. `capture_model`（表 G，NPU 走 ACLGraph） — 图捕获 ✅worker
8. omni `api_server.py:557` `serve_http` — ready ✅主

### 一个请求，6 个断点（GPU 心智）

1. omni `orchestrator.py:424` `_handle_add_request` — 请求进编排 ✅主
2. vllm `core.py:479` `EngineCore.step` — 进 stage 这一拍 ✅stage
3. omni `omni_ar_scheduler.py` `schedule` — 这拍批了谁 ✅stage
4. vllm `gpu_model_runner.py:3793` `self.model(...)` — forward 现场 ✅worker
5. vllm `gpu_model_runner.py:4435` `sample_tokens` — 出 token ✅worker
6. omni `orchestrator.py:632` `_orchestration_output_handler` — 输出回流/转下一站 ✅主

NPU 把 4/5 换成 vllm-ascend 的 `model_runner_v1` / `attention_v1` / `AscendSampler`，其余 1/2/3/6 一字不差。

---

## 五、NPU / 多进程调试注意（要点摘录）

- **`debugpy` 启动期要卡在断点之前**：调 `load_model`/`capture` 要让 `wait_for_client()` 在 stage/worker 进程**早于这些调用**执行（见 [debug-ascend-remote §4](debug-ascend-remote.md) 的按 stage/rank 门控注入）。
- **aclgraph 捕获期不要单步**：捕获中触发 host 同步会破坏捕获甚至崩。要断 host 逻辑先 `enforce_eager`，或只断到 `_dummy_run`/replay。
- **多进程一次只点一个**：TP 多 rank 都 `wait_for_client` 会卡死集合通信；只调 rank0，其余 rank 门控放行。
- **看真实 device 值要同步**：NPU 张量取值触发 device→host，优先 `.gpu[:n].cpu()`。
- **超时**：被断住的进程会让心跳/watchdog 超时杀子进程；调试期调大相关 timeout，或接受「断一次就重启」。

---

## 六、关键文件索引

| 仓库 | 启动期 | 请求期 |
|---|---|---|
| **vllm** | `v1/engine/core.py:900/99/123/239` · `executor/multiproc_executor.py:182/702/807` · `v1/worker/gpu_worker.py:250/606/649/658` · `gpu_model_runner.py:5155` | `v1/engine/core.py:372/479` · `core_client.py:1121/1005` · `gpu_model_runner.py:3793/4435` · `async_llm.py:656` |
| **vllm-omni** | `entrypoints/cli/{main,serve}.py` · `entrypoints/openai/api_server.py:459/629/557` · `async_omni_engine.py:304/411` · `stage_runtime.py:223/562` · `stage_engine_core_proc{,_manager}.py` | `orchestrator.py:424/632/1158` · `stage_init_utils.py:53/150` · `core/sched/omni_ar_scheduler.py` · `worker/gpu_ar_*` · `output_processor.py` · `entrypoints/async_omni.py:249` |
| **vllm-ascend** | `__init__.py:40` · `platform.py:155/819/896` · `worker/worker.py:393/450` · `worker/model_runner_v1.py:253` · `compilation/acl_graph.py` | `attention/attention_v1.py:1095/1115` · `worker/model_runner_v1.py` · `AscendSampler` · `npu_communicator.py:23` |

---

!!! info "说明"
    本文综合三仓库源码 + 多路并行 trace；类名/调用关系可靠，行号随版本漂移，以实际仓库为准。**配套阅读**：[在 VSCode 里远程调试 Ascend 容器](debug-ascend-remote.md)（怎么 attach 子进程）、[从 curl 到返回的请求生命周期](request-lifecycle-end-to-end.md)（流程全貌）、[三处 worker 继承关系](worker-class-hierarchy.md)（执行层继承网）。
