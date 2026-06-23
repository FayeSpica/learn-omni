# 在 VSCode 里远程调试 Ascend 容器内的 vLLM-Omni

> 场景：模型跑在远程昇腾机器 **Ascend01** 的 **容器**里，我们要在本机 VSCode 里断点调试当前这个
> **NPU 前缀缓存 hidden 还原失败（talker assistant 段为空 → `6 vs 9` 崩溃）** 的问题。

本文给出可直接照做的步骤，并标注**这个具体 bug 该在哪些进程、哪些 `file:line` 下断点**。

---

## 0. 先认清难点：vLLM-Omni 是多进程的

断点抓不到，十有八九是**断在了错误的进程**。一个 Omni 部署同时有这些进程：

```
APIServer / Orchestrator (主进程)
└─ 每个 stage 一个引擎核：StageEngineCoreProc_stage{N}_replica0
   └─ 若该 stage TP>1：再 spawn  Worker_TP0 / Worker_TP1 ...（模型 forward 在这里跑）
      若该 stage TP==1：模型 forward 直接在 StageEngineCoreProc 里跑（uniproc executor）
```

对照本仓库 `qwen3_omni_moe.yaml`：

| stage | 角色 | TP | 模型 forward 实际在哪个进程 |
|---|---|---|---|
| 0 | thinker | **2** | `Worker_TP0` / `Worker_TP1` |
| 1 | talker | 1 | `StageEngineCoreProc_stage1_replica0` 本体 |
| 2 | code2wav | 1 | `StageEngineCoreProc_stage2_replica0` 本体 |

**当前 bug 的断点分布**（先记住，后面要用）：

| 关注点 | 代码位置 | 跑在哪个进程 |
|---|---|---|
| 前缀缓存合并是否触发 | `platforms/npu/worker/npu_ar_model_runner.py::_maybe_get_combined_prefix_cache_tensors` | **stage0 的 Worker_TP0** |
| 前缀缓存写入 slot_mapping | `platforms/npu/worker/npu_ar_model_runner.py::_maybe_update_prefix_cache` | stage0 Worker_TP0 |
| 命中登记开关 | `core/prefix_cache.py::_get_merged_tensors`（`if req_id in self._new_req_cache_hit_ids`） | stage0 Worker_TP0 |
| 命中登记是否发生 | `worker/gpu_model_runner.py::_update_states`（`add_prefix_cached_new_req_id`，看 `num_computed_tokens>0`） | stage0 Worker_TP0 |
| thinker→talker 投递累积 | `model_executor/stage_input_processors/qwen3_omni.py::thinker2talker_async_chunk` | Orchestrator / 连接器进程 |
| 最终崩溃点 | `model_executor/models/qwen3_omni/qwen3_omni.py::_get_talker_assistant_parts` | **stage1 引擎核本体** |

> 结论：**多进程下 IDE 的「按 F5 启动」基本没用**，要用 **debugpy 远程 attach** 精确连到目标子进程。

---

## 1. 总体方案

```
本机 VSCode
  └─(Remote-SSH)→ Ascend01 主机
       └─(Dev Containers: Attach to Running Container)→ 容器内 VSCode backend
            └─(debugpy attach, localhost:5678)→ 目标子进程（被 env 门控 listen）
```

推荐 **Remote-SSH → Attach 容器**，这样 VSCode backend 直接跑在容器里，**文件系统/路径一致，不用配 pathMappings**。

---

## 2. 步骤一：Remote-SSH 连到 Ascend01

本机装扩展 **Remote - SSH**。`~/.ssh/config` 加：

```sshconfig
Host ascend01
    HostName <Ascend01 的 IP>
    User root
    # 如需跳板：ProxyJump bastion
```

VSCode 命令面板 → `Remote-SSH: Connect to Host` → `ascend01`。

---

## 3. 步骤二：进入容器

先在 Ascend01 上看容器：

```bash
docker ps            # 找到跑 vllm-omni 的容器名，假设叫 omni
```

VSCode 装 **Dev Containers** 扩展 → 命令面板 → `Dev Containers: Attach to Running Container` → 选 `omni`。
附着成功后 `File: Open Folder` 打开容器里的仓库，例如（按崩溃栈里的真实路径）：

```
/root/lwm_omni/git/vllm-omni
```

> 这样 VSCode 的源码视图就是容器里那份**可编辑安装（pip install -e）**的代码，断点行号天然对齐。

容器里装调试器：

```bash
pip install debugpy
```

---

## 4. 步骤三：在目标子进程注入 debugpy（关键）

multiprocessing spawn 出来的子进程，**IDE 不会自动接管**，必须让目标进程自己 `listen` 并等 attach。
做法：放一个**按 env 门控、按身份选端口**的 hook，在进程早期 import 一次。

在容器仓库根新建 `omni_debug_hook.py`：

```python
# omni_debug_hook.py —— 仅调试用，勿提交
import os

def maybe_start_debugpy() -> None:
    spec = os.environ.get("OMNI_DEBUG", "")     # 例: "stage0:tp0"
    if not spec:
        return

    # 进程身份：stage 来自 run_stage_core 设的 env；tp rank 来自 LOCAL_RANK
    stage = os.environ.get("VLLM_OMNI_STAGE_ID", os.environ.get("OMNI_STAGE_ID", "?"))
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))

    want_stage, _, want_tp = spec.partition(":")
    want_stage = want_stage.replace("stage", "")
    want_tp = want_tp.replace("tp", "") or "0"
    if str(stage) != str(want_stage) or str(rank) != str(want_tp):
        return

    import debugpy
    # 端口按身份错开，避免多进程抢占
    port = 5678 + int(stage) * 10 + int(rank)
    debugpy.listen(("0.0.0.0", port))
    print(f"[debugpy] stage={stage} tp={rank} listening on :{port}, waiting for attach ...", flush=True)
    debugpy.wait_for_client()   # 阻塞直到 VSCode attach；只想后台监听就删掉这行
```

**注入点**（二选一，按你要断的进程）：

- **要断 stage0 的 TP worker**（前缀缓存合并/写入/命中登记都在这）：
  在 `NPUARModelRunner.__init__`（`platforms/npu/worker/npu_ar_model_runner.py`）最前面加：
  ```python
  from omni_debug_hook import maybe_start_debugpy
  maybe_start_debugpy()
  ```
  并确保 worker 能看到 stage id——`run_stage_core` 已 `os.environ["VLLM_OMNI_REPLICA_ID"]`；若没有 `VLLM_OMNI_STAGE_ID`，就在 `run_stage_core`（`engine/stage_engine_core_proc.py`）里补一行
  `os.environ["VLLM_OMNI_STAGE_ID"] = str(omni_stage_id)`，让 TP 子进程继承。

- **要断 stage1 引擎核本体**（最终崩溃点 `_get_talker_assistant_parts`）：
  在 `run_stage_core` 设完 env 之后插：
  ```python
  os.environ["VLLM_OMNI_STAGE_ID"] = str(omni_stage_id)
  from omni_debug_hook import maybe_start_debugpy
  maybe_start_debugpy()
  ```

启动服务时带上目标：

```bash
OMNI_DEBUG=stage0:tp0 <你平时的启动命令>      # 断 thinker 的 TP0
# 或
OMNI_DEBUG=stage1:tp0 <你平时的启动命令>      # 断 talker 引擎核
```

进程会停在 `wait_for_client()`，打印 `listening on :5688`（stage0+tp0 → 5678+0*10+0=5678；注意按公式算端口）。

---

## 5. 步骤四：VSCode attach 配置

容器里仓库 `.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "omni: attach stage0 tp0",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "127.0.0.1", "port": 5678 },
      "justMyCode": false
    },
    {
      "name": "omni: attach stage1",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "127.0.0.1", "port": 5688 },
      "justMyCode": false
    }
  ]
}
```

- `justMyCode: false`：要能进 `vllm` / `vllm_ascend` / `transformers` 的库代码。
- 因为 VSCode backend 在**容器内**，`127.0.0.1` 就是容器本机，**无需 pathMappings**。
- 若你没用「Attach 容器」而是直连主机调容器内进程，则需 `ssh -L 5678:localhost:5678 ascend01` 端口转发，并在 attach 里加
  `"pathMappings": [{ "localRoot": "${workspaceFolder}", "remoteRoot": "/root/lwm_omni/git/vllm-omni" }]`。

按 F5 选对应配置 → 进程从 `wait_for_client()` 继续 → 命中断点。

---

## 6. 步骤五：针对本 bug 的断点与观察

先打这几个断点（对应我们日志里 `[diag]` 的位置），一次请求即可定位：

1. `worker/gpu_model_runner.py` → `_update_states` 里
   `if self.omni_prefix_cache is not None and new_req_data.num_computed_tokens > 0:`
   **看 `new_req_data.num_computed_tokens` 是不是 >0**。==0 → 调度侧没报前缀命中 → 登记不发生，后面 merge 必然直通。

2. `core/prefix_cache.py` → `_get_merged_tensors`
   `if req_id in self._new_req_cache_hit_ids:`
   **看进没进这个分支**；进了再看 `block_ids = self._get_cached_block_ids(...)` 和 `cached_hs = cache[block_ids]` 的 shape 是否非空。

3. `platforms/npu/worker/npu_ar_model_runner.py` → `_maybe_get_combined_prefix_cache_tensors`
   看 `combined_multimodal_outputs` 里 `hidden_states.layers.0` 每个 req 的行数：**是「前缀+新增」全长，还是只有新增**。

4. `model_executor/stage_input_processors/qwen3_omni.py` → `thinker2talker_async_chunk`
   看 `thinker_emb`/累积后的 `prefill` 行数有没有追上 `prompt_len`。

5. `model_executor/models/qwen3_omni/qwen3_omni.py` → `_get_talker_assistant_parts`
   崩前现场：`thinker_embed.shape[0]` vs `im_start_index`，确认 assistant 段是否空。

> 这些点我们已经用 `logger.warning("[diag]...")` 埋过；断点是为了能**逐帧看变量/调用栈**，比日志更细。

**Watch / Debug Console 常用表达式**：

```python
self.omni_prefix_cache.has_prefix_cached_new_req_ids()
sorted(self.omni_prefix_cache._new_req_cache_hit_ids)
self.omni_prefix_cache.mm_cache_keys
self.input_batch.block_table[0].slot_mapping.gpu[:num_tokens_padded]   # 对比 .cpu 是否过期
{k: {r: t.shape for r,t in v.items()} for k,v in (combined_multimodal_outputs or {}).items() if isinstance(v,dict)}
```

---

## 7. 昇腾 / NPU 专属注意事项

- **不要在 aclgraph 捕获期单步**：捕获中触发 host 同步会破坏捕获甚至崩。要调前缀缓存这种 host 侧逻辑没问题；若涉及 graph capture，先 `enforce_eager` 或只在 replay 阶段断。
- **看真实 device 值要同步**：NPU 张量打印/取值会触发 device→host。`slot_mapping.gpu` vs `.cpu` 的过期问题正是这类——调试时优先看 `.gpu[:n].cpu()`。
- **多进程不要全开**：`OMNI_DEBUG` 一次只点一个进程；TP 两个 rank 都 `wait_for_client` 会卡住集合通信。只调 rank0，其余 rank 不匹配直接放行。
- **超时**：被断住的进程会让 watchdog/心跳超时杀子进程。调试期把相关 timeout 调大，或接受「断一次就重启」。
- **编辑即生效**：容器里是 `pip install -e`，改源码（含上面注入）保存即对下次启动生效，无需重装。

---

## 8. 更轻量的替代（不想配 IDE 时）

- **一行注入断点**：在目标代码处临时写
  ```python
  import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
  ```
- **远程 pdb**：`from remote_pdb import RemotePdb; RemotePdb('0.0.0.0', 4444).set_trace()`，再 `nc 127.0.0.1 4444`。
- **就用 `[diag]` 日志**：我们已埋的 `[diag][prefix_merge]` / `[diag][t2t_*]` / `[diag][assistant_parts]` 足以区分「命中没登记 / 写没落盘 / 读错块」，多数情况下不必上断点。

---

## 9. 一页速查

```text
1) Remote-SSH 连 ascend01
2) Dev Containers: Attach 到容器 → 打开 /root/lwm_omni/git/vllm-omni → pip install debugpy
3) 放 omni_debug_hook.py + 在 NPUARModelRunner.__init__(或 run_stage_core) 注入 maybe_start_debugpy()
4) OMNI_DEBUG=stage0:tp0 启动服务（断 thinker TP0）
5) .vscode/launch.json 配 attach 127.0.0.1:<端口>，justMyCode=false
6) 断点：gpu_model_runner._update_states(num_computed_tokens) → prefix_cache._get_merged_tensors(命中分支)
        → npu_ar_model_runner._maybe_get_combined(合并行数) → qwen3_omni._get_talker_assistant_parts(崩前)
7) 发一个含图/音输入的请求触发，逐帧看 _new_req_cache_hit_ids 与 merge 行数
```
