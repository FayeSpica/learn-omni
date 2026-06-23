# 案例倒推：NPU 上 talker 因前缀缓存缺兜底而崩（`a(6) vs b(9)`）

> 现象：Qwen3-Omni 全模态（带图/音输入）用例，**GPU 正常、NPU 崩**。崩在 talker 预处理：
> `assistant_text_hidden(6) + assistant_codec_hidden(9)` 形状不匹配。
>
> 本文从**崩点倒推到根因**，把每一环标清 GPU/NPU 是「同」还是「分叉」，结论是：
> **真正的分叉只有一个——`omni_prefix_cache` 在 NPU 从未实例化**。

---

## 0. 崩溃栈

```
npu_ar_model_runner.execute_model → _preprocess
  → qwen3_omni.talker_preprocess_prefill → _thinker_to_talker_prefill
    → _get_talker_assistant_parts
       input_embeds = assistant_text_hidden + assistant_codec_hidden
RuntimeError: The size of tensor a (6) must match the size of tensor b (9) at non-singleton dimension 0
```

`assistant_text_hidden` 的行数 = `min(3, A) + 4(pad) + 1(bos) + 1(text)`，其中 `A = assistant_hidden.shape[0]`。
`a=6` ⟺ `A=0`（`0+4+1+1`），即 **assistant 段为空**；`assistant_codec_hidden` 恒为 `3+6=9`。

---

## 1. 关键诊断数据（埋 `[diag]` 抓到的）

对比一条**通过**和一条**崩溃**的请求：

| | 通过 | 崩溃 |
|---|---|---|
| `target_len`（prompt+生成） | 176 | **274** |
| 生成 token 数 | ~2 | ~100 |
| `im_start_indexes` | `[0,171,176]` | `[0,171,274]` |
| `mm_token_count` | 64 | 64（相同！） |
| `thinker_embed` 行数 | **175**（≈全） | **145**（截断） |
| `assistant_hidden` | (4,1024) ✓ | **(0,1024)** ✗ |

两条关键结论：

1. **不是多模态输入本身的问题**——通过/崩溃两条 `mm_token_count` 都是 64。
2. `thinker2talker` 累积日志里 **`all_len - prefill_shape = 129` 恒定**：hidden 从一开始就比 token 流落后 129 行。
3. **`[diag][prefix_merge]` 从不打印**——它在 `if self.omni_prefix_cache is not None:` 里面 → 说明 **NPU 上 `omni_prefix_cache` 是 `None`**。

---

## 2. 从崩点倒推

```
① 崩点  _get_talker_assistant_parts   [stage1 talker]
   a(6) vs b(9)
   ▸ 共享代码。直接原因：thinker_embed[171:274] 为空 → assistant_hidden 行数=0
   ───────────────────────────────────────────────
② thinker_embed 只有 145 行（应 274）   [stage1 talker 预处理]
   _thinker_to_talker_prefill 用 274 的 mask/分段索引 145 的 embed
   ▸ 共享代码。错不在这，错在喂进来的 embed 就短
   ───────────────────────────────────────────────
③ embed.prefill=145 怎么投递来的   [thinker→talker 投递]
   ┌─ 分叉 A（传输层，非病根）
   │   GPU：thinker2talker_full_payload ← OmniConnectorModelRunnerMixin.accumulate_full_payload_output（concat 全序列）
   │   NPU：thinker2talker_async_chunk  ← OmniKVTransferManager（逐 chunk）
   │   ∵ GPUARModelRunner(..., OmniConnectorModelRunnerMixin)；NPUARModelRunner 不继承 mixin
   └─ 但 async_chunk 只是忠实搬运「上游已经短了的 145」，不是病根
   ───────────────────────────────────────────────
④ thinker 产出的 hidden 本身就是 145   [stage0 thinker]   ★真正的分叉★
   KV 前缀缓存命中 129 → thinker 只算新增 token → 捕获 145 行（即「prefill 少数据」）
   本应由 omni_prefix_cache.get_merged_* 把那 129 行补回来
   ┌─ 分叉 B（根因）
   │   GPU：omni_prefix_cache 存在 → 补回 129 → 全序列 274
   │   NPU：omni_prefix_cache = None（从没建）→ 补不回 → 永远 145
   └─ ∵ 实例化挂在 OmniGPUModelRunner.initialize_metadata_builders（GPU 专有方法），
        ascend 初始化流程根本不调它
```

**结论：真正的分叉只有 ④。** ①②③ 全是共享代码或忠实搬运。

---

## 3. 两个 cache 别混（这是"prefill 少数据"的本质）

| | 是什么 | 作用 |
|---|---|---|
| **KV 前缀缓存**（`enable_prefix_caching`，vLLM 标准） | 缓存前缀 token 的 attention KV | 命中时**让 thinker prefill 只算新增 token**（`num_computed_tokens=129` 被跳过） |
| **omni_prefix_cache**（`OmniTensorPrefixCache`） | 缓存前缀 token 的 **hidden / mm 层** | 把被跳过的 129 行 hidden **补回来**给 talker（merge） |

因果链：

```
enable_prefix_caching=true（stage0）
 → KV 前缀命中（复用 129 token）
  → thinker 只算新增，捕获 hidden 少 129 行   ← 「prefill 少数据」
   → 本应 omni_prefix_cache.merge 补回
    → NPU omni_prefix_cache=None → 补不回 → 崩
```

「prefill 少数据」**不是 bug，是前缀缓存命中的设计后果**；omni_prefix_cache 才是兜底。NPU 兜底缺失，缺口裸露。

证据对得上：**通过的请求是全新 prompt（没命中前缀，偏移≈1）**；**崩的请求命中了 129 token 前缀（偏移=129）**。这也解释了「会话里第一条正常、后面带共享上下文的崩」。

---

## 4. 根因定位：`omni_prefix_cache` 为何在 NPU 是 None

实例化只出现在一处：

```python
# vllm_omni/worker/gpu_model_runner.py
def initialize_metadata_builders(self, kv_cache_config, kernel_block_sizes):
    super().initialize_metadata_builders(...)
    ...
    if self.cache_config.enable_prefix_caching:           # ← 仅此一处建 cache
        self.omni_prefix_cache = OmniTensorPrefixCache(
            num_blocks=kv_cache_config.num_blocks,
            block_size=self.cache_config.block_size,
            hidden_size=self.model_config.get_hidden_size(),
            hs_dtype=self.dtype,
        )
```

- `initialize_metadata_builders` 是 **vLLM-GPU 专有方法**；`grep` 全仓只在 `gpu_model_runner.py` 出现。
- NPU 的 `NPUARModelRunner(OmniNPUModelRunner)` 走 vllm-ascend 自己的 KV/metadata 初始化，**根本不调这个方法** → 实例化代码在 NPU 上是死代码 → `omni_prefix_cache` 停在默认 `None`。

于是 NPU 上：KV 前缀缓存照常缩短 prefill（129 偏移），但兜底的 hidden 缓存**压根不存在** → 所有依赖它的写/合并逻辑（包括后面 C/D/E）都被 `if omni_prefix_cache is not None` 挡成死代码。

---

## 5. 分叉 B 往里还有 4 个 NPU 缺口

只有当你想"在 NPU 真正开这份 cache"时才需要补：

| | GPU | NPU | 说明 |
|---|---|---|---|
| **C 写槽位** | `slot_mapping.gpu[:n].cpu()` 同步 | stale `.cpu`（kernel 只写 `.gpu`） | 槽位错 → 即便写了也读不回 |
| **D 写路径** | 异步 `schedule_async_write`+`drain_ready_async_writes`（**CUDA 门控**） | 只有同步 `update_omni_tensor_prefix_cache` | NPU 用不了异步路，CUDA 门控早退 |
| **E CPU 暂存** | stage `hidden_states_cpu`（同步 D2H 视图） | 无；hidden 路径未 coerce 到 CPU | 同步 index_copy 用 device 源写 CPU cache 会不匹配 |
| **F 命中登记** | `add_prefix_cached_new_req_id`（`_update_states`，`num_computed_tokens>0`） | 继承同方法，但**未确认触发** | merge 只有 `req_id ∈ _new_req_cache_hit_ids` 才拼前缀 |
| **建 cache 本身** | initialize_metadata_builders | 试加 `initialize_kv_cache` → **init 失败（疑 OOM）** | 整 KV 大小的 CPU 镜像太大 |

> `omni_prefix_cache` 是**面向 GPU 的设计**：整份 KV block 大小的 **CPU 镜像** + **CUDA 异步拷贝流水线**。搬到 NPU 是重活，第一步建 cache 就疑似 OOM。

---

## 6. 两条修法

- **路线 1（绕开缺口，推荐先做）**：thinker `enable_prefix_caching: false`
  ```yaml
  stages:
    - stage_id: 0
      enable_prefix_caching: false   # 每次整段 prefill，不依赖 omni cache 兜底
  ```
  → 不命中前缀 → 不缩短 prefill → thinker 每次捕获全序列 274 → 根本不需要 omni_prefix_cache → 链路自洽，崩溃消失。
  代价：共享前缀重算（TTFT↑、共享前缀场景吞吐↓）。**低风险、立刻解锁**，且能反证整条倒推。

- **路线 2（保前缀缓存性能）**：在 NPU 真正落地 `omni_prefix_cache`
  1. 先解决建 cache 的 **OOM**（缩小镜像/惰性分配/换 dtype，而非整 KV 全量镜像）；
  2. 补 **C**（`.gpu→.cpu` 同步槽位，已验证写法）、**E**（hidden coerce 到 CPU 再 index_copy）；
  3. 确认 **F**（`num_computed_tokens>0` 是否在 ascend 调度侧报命中 → 触发登记）；
  4. **D** 在 NPU 只能走同步路，接受其代价。
  **高成本、需逐项验证**。

---

## 7. 排查方法论小结（可复用）

1. **从崩点倒推，给每一环贴「共享 / 分叉」标签**——避免在共享代码或下游症状上空转。
2. **多进程下先确认断点/日志在对的进程**：thinker(TP>1) 在 `Worker_TP*`，talker(TP=1) 在 `StageEngineCoreProc` 本体。
3. **`[diag]` 日志比断点更快定位数据流缺口**：
   - `[diag][prefix_merge]` 不打印 = `omni_prefix_cache is None`（一锤定音）。
   - `all_len - prefill` 恒定偏移 = 前缀缓存命中长度。
4. **分清"制造缺口"与"填补缺口"的配对**：KV 前缀缓存制造（缩短 prefill），omni_prefix_cache 填补（merge 还原）。GPU 两者齐全，NPU 缺填补。
5. **平台漂移的高发区**：功能被挂在某平台专有方法/Mixin 上（这里是 `initialize_metadata_builders` / `OmniConnectorModelRunnerMixin`），另一平台不走那条路就静默丢功能。

---

## 相关阅读

- [platforms/npu 架构导读](npu-platform-architecture.md)
- [三处 worker 的职责与继承关系梳理](worker-class-hierarchy.md)
- [在 VSCode 里远程调试 Ascend 容器内的 vLLM-Omni](debug-ascend-remote.md)
