---
tags:
  - vllm
  - vllm-omni
  - vllm-ascend
  - worker
  - ModelRunner
  - 继承关系
  - NPU
---

# 三处 worker 的职责、关联与继承关系梳理

> 一个问题：**`vllm_omni/platforms/npu/worker`、`vllm_omni/worker`、`vllm-ascend/vllm_ascend/worker` 这三处都叫 worker,各管什么?它们是怎么串起来、谁继承谁的?**
>
> 三处不是平级的三套实现,而是**沿两根轴叠出来的一张网**:一根是「硬件」(GPU ↔ 昇腾 NPU),一根是「角色」(自回归 AR ↔ 生成 Generation)。本文按"地基 → 落硬件 → 落 omni"的顺序铺开,签名均对照源码核对。平台解耦的机制背景见 [Omni 平台无关/相关解耦](platform-decoupling.md)。

## 一、结论速览:三处的定位

| 目录 | 角色 | 一句话定位 |
|---|---|---|
| `vllm-ascend/vllm_ascend/worker` | **硬件适配层** | 把 vLLM 上游的 GPU worker/runner **翻译到昇腾 NPU**(NPUWorker / NPUModelRunner) |
| `vllm_omni/worker` | **omni 的 GPU 实现** | 在 GPU 上把 worker/runner 扩成 **omni 三段式**(AR / Generation)+ 连接器 I/O |
| `vllm_omni/platforms/npu/worker` | **omni × 昇腾的交汇点** | 把上面两者**缝合**:既要 omni 的三段式特性,又要昇腾的设备实现 |

关键认知:**第三处目录本身几乎不写新逻辑,它的价值在"继承谁"** —— 通过菱形继承,同时把"omni 特性"和"昇腾特性"拉到一起(§六)。

## 二、地基:上游 vLLM 的 worker 与 model runner

一切的根在 `vllm/v1/worker/`,而且要先分清**两条并行的类**:

| 类 | 职责 | 类比 |
|---|---|---|
| **Worker** | **进程/设备层**:`init_device()`、显存测算、KV cache、调度 `execute_model()` | 工头 |
| **ModelRunner** | **计算层**:建 batch、跑 forward、采样、图捕获 | 干活的 |

**一个 Worker 持有一个 ModelRunner** —— Worker 在 `init_device()` 里实例化对应的 runner,自己负责设备/内存,把真正的前向委托给 runner。两条链各自独立继承,后面会反复看到这个分工。

上游基类:

```python
# vllm/v1/worker/worker_base.py
class WorkerBase:           # 纯接口:init_device/load_model/execute_model/get_kv_cache_spec... 全是 raise
class WorkerWrapperBase:    # 懒初始化包装

# vllm/v1/worker/gpu_worker.py
class Worker(WorkerBase):   # GPU 上的真实现,override 上述全部 + sleep()/wake_up()

# vllm/v1/worker/gpu_model_runner.py
class GPUModelRunner:       # GPU 上的计算层基类
```

## 三、vllm-ascend/worker:把地基落到昇腾

vllm-ascend 做的是**硬件适配**——直接继承上游基类,换成昇腾的设备/算子/图捕获:

```python
# vllm_ascend/worker/worker.py
class NPUWorker(WorkerBase):               # 注意:直接继承 WorkerBase,不走 GPU 的 Worker
    # init_device/load_model/execute_model/determine_available_memory/sleep/wake_up 全部昇腾实现

# vllm_ascend/worker/model_runner_v1.py
class NPUModelRunner(GPUModelRunner):       # runner 则复用 GPU 的,叠昇腾 attention/ACLGraph/kernel

# vllm_ascend/_310p/worker_310p.py
class NPUWorker310(NPUWorker):              # 310P 变体:覆盖设备初始化/显存/分片
class NPUModelRunner310(NPUModelRunner):    # 310P 的 ACLGraph 与 FRACTAL_NZ 格式
```

值得注意的**不对称**:

- `NPUWorker` **直接继承 `WorkerBase`**(而非 GPU 的 `Worker`)—— worker 层昇腾要从接口重写,复用不了 CUDA 细节。
- `NPUModelRunner` **继承 GPU 的 `GPUModelRunner`** —— 计算层骨架(建 batch、采样流程)可大量复用,只换 attention 后端、图捕获和 kernel。

这正是 vllm-ascend 作为"第二类后端"的典型姿态:**runner 层尽量复用上游,worker 层另起炉灶**。

## 四、vllm_omni/worker:GPU 上的 omni 三段式

omni 在 GPU 上把 worker/runner 扩成三段式流水线(Thinker→Talker→Code2Wav,见 [Qwen3-Omni 在 NPU 上是怎么跑起来的](qwen3-omni-npu.md))。这里出现了**角色分化**(AR vs Generation)和**两个 mixin**:

```python
# vllm_omni/worker/base.py
class OmniGPUWorkerBase(GPUWorker):         # GPUWorker = vllm 的 Worker
    # override determine_available_memory():改用 pynvml 按进程(PID)算显存
    # 换 OmniTorchProfilerWrapper;新增 sleep/wake/handle_sleep_task/handle_wake_task

# vllm_omni/worker/mixins.py
class OmniWorkerMixin:                       # 跨切面:确保 worker 子进程里 load_omni_general_plugins()

# vllm_omni/worker/gpu_ar_worker.py
class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):          # 自回归段(Thinker/Talker)
    # init_device() 里实例化 GPUARModelRunner
# vllm_omni/worker/gpu_generation_worker.py
class GPUGenerationWorker(OmniWorkerMixin, OmniGPUWorkerBase):  # 生成段(Code2Wav,非自回归)
    # init_device() 里实例化 GPUGenerationModelRunner
```

对应的 model runner 这边也分化,并叠上**连接器 mixin**(跨 stage 数据搬运,见 [组件与请求流转](components-request-flow.md)):

```python
# vllm_omni/worker/gpu_model_runner.py
class OmniGPUModelRunner(GPUModelRunner):     # omni 特性:prefix cache、中间张量缓冲、连接器集成
# vllm_omni/worker/omni_connector_model_runner_mixin.py
class OmniConnectorModelRunnerMixin:          # 统一 connector.put()/get(),管后台 I/O 线程
# gpu_ar_model_runner.py / gpu_generation_model_runner.py
class GPUARModelRunner(OmniGPUModelRunner, OmniConnectorModelRunnerMixin): ...
class GPUGenerationModelRunner(OmniGPUModelRunner, OmniConnectorModelRunnerMixin): ...
```

要点:

- **角色分化(AR / Generation)发生在叶子层**:worker 和 runner 都各分两个叶子,分别对应"自回归"与"非自回归生成"两类 stage。
- **两个 mixin 各管一件横切事**:`OmniWorkerMixin` 管插件加载(worker 侧),`OmniConnectorModelRunnerMixin` 管跨 stage 通信(runner 侧)。mixin 放在 MRO **最前**,保证它的 `__init__` 先跑。

## 五、vllm_omni/platforms/npu/worker:omni × 昇腾的交汇

这是问题里那个最绕的目录。它要的是:**既有 §四 的 omni 三段式能力,又有 §三 的昇腾设备实现**。做法是分别在 worker 链和 runner 链上"换基类"。

### 5.1 Worker 链:条件继承昇腾 worker

```python
# vllm_omni/platforms/npu/worker/base.py
from vllm_omni.platforms.npu._310p import is_310p
if is_310p():
    from vllm_ascend._310p.worker_310p import NPUWorker310 as NPUWorker
else:
    from vllm_ascend.worker.worker import NPUWorker      # ← 直接拿 vllm-ascend 的 NPUWorker

class OmniNPUWorkerBase(NPUWorker):          # 把 §四 的 OmniGPUWorkerBase 换成昇腾基座
    # 换 OmniTorchProfilerWrapper,override profile()

# npu_ar_worker.py / npu_generation_worker.py —— 和 GPU 侧完全对称
class NPUARWorker(OmniWorkerMixin, OmniNPUWorkerBase): ...        # init_device → NPUARModelRunner
class NPUGenerationWorker(OmniWorkerMixin, OmniNPUWorkerBase): ...# init_device → NPUGenerationModelRunner
```

**条件继承**:用 `is_310p()` 在 import 期决定基类是 `NPUWorker` 还是 `NPUWorker310`,把 310P 变体**透明地塞进继承链中段**,叶子类(NPUARWorker 等)无感。

### 5.2 Runner 链:菱形继承,同时拿两边特性

最关键的一处:

```python
# vllm_omni/platforms/npu/worker/npu_model_runner.py
class OmniNPUModelRunner(OmniGPUModelRunner, NPUModelRunner):   # ★菱形★
    # 同时继承:omni 的 GPU runner(omni 特性) + vllm-ascend 的 NPU runner(昇腾特性)
    # load_model() 里打平台相关 patch

# npu_ar_model_runner.py / npu_generation_model_runner.py
class NPUARModelRunner(OmniNPUModelRunner): ...        # 自回归 + 异步 ACLGraph 捕获
class NPUGenerationModelRunner(OmniNPUModelRunner): ...# code2wav 非自回归波形合成
```

`OmniNPUModelRunner` 的两个父类 `OmniGPUModelRunner` 和 `NPUModelRunner` **都继承自 `GPUModelRunner`** —— 这就是一个标准菱形,汇于 `GPUModelRunner`。MRO 让它先吃到 omni 的扩展,再吃到昇腾的扩展,二者最终落回同一个 GPU 基座,不会撕裂。这正是 [平台解耦](platform-decoupling.md) 里讲的"菱形继承"模式的实例。

## 六、两根轴叠成一张网

把所有叶子 worker 放进「硬件 × 角色」矩阵,三处目录的关系一目了然:

| | **AR(自回归:Thinker/Talker)** | **Generation(非自回归:Code2Wav)** |
|---|---|---|
| **GPU**(`vllm_omni/worker`) | `GPUARWorker` | `GPUGenerationWorker` |
| **NPU**(`vllm_omni/platforms/npu/worker`) | `NPUARWorker` | `NPUGenerationWorker` |

四个叶子两两对称:**同一列共享"角色"逻辑**(实例化哪个 runner、自回归还是生成),**同一行共享"硬件"基座**(GPU 的 `OmniGPUWorkerBase` / NPU 的 `OmniNPUWorkerBase`)。`vllm-ascend/worker` 不在这张表里——它是 NPU 那一行的**地基供应商**。

## 七、两条平行继承链(mermaid)

### Worker 链

```mermaid
flowchart TB
  WB["WorkerBase<br/>vllm.v1.worker.worker_base<br/>纯接口"]
  WB --> GW["Worker<br/>vllm gpu_worker"]
  WB --> NW["NPUWorker<br/>vllm_ascend.worker.worker"]
  NW -. "is_310p()" .-> NW310["NPUWorker310<br/>vllm_ascend._310p"]
  GW --> OGB["OmniGPUWorkerBase<br/>vllm_omni/worker/base"]
  NW --> ONB["OmniNPUWorkerBase<br/>vllm_omni/platforms/npu/worker/base<br/>(条件继承 NPUWorker/310)"]
  OGB --> GAR["GPUARWorker"]
  OGB --> GGEN["GPUGenerationWorker"]
  ONB --> NAR["NPUARWorker"]
  ONB --> NGEN["NPUGenerationWorker"]
  MIX["OmniWorkerMixin<br/>(插件加载)"] -. "mix-in 最前" .- GAR
  MIX -. .- NAR
```

### ModelRunner 链(注意 NPU 处的菱形)

```mermaid
flowchart TB
  GMR["GPUModelRunner<br/>vllm"]
  GMR --> OGMR["OmniGPUModelRunner<br/>vllm_omni/worker"]
  GMR --> NMR["NPUModelRunner<br/>vllm_ascend"]
  OGMR --> GARM["GPUARModelRunner"]
  OGMR --> GGENM["GPUGenerationModelRunner"]
  OGMR --> ONMR["OmniNPUModelRunner<br/>★菱形:同时继承<br/>OmniGPUModelRunner + NPUModelRunner★"]
  NMR --> ONMR
  ONMR --> NARM["NPUARModelRunner"]
  ONMR --> NGENM["NPUGenerationModelRunner"]
  CMIX["OmniConnectorModelRunnerMixin<br/>(跨 stage I/O)"] -. .- GARM
  CMIX -. .- GGENM
```

## 八、关联关系:谁持有谁、谁 import 谁

- **持有关系**:每个 `*Worker` 在 `init_device()` 里实例化对应的 `*ModelRunner`(GPUARWorker→GPUARModelRunner,NPUARWorker→NPUARModelRunner,以此类推)。worker 管设备,runner 管计算。
- **跨树 import(继承边)**:
    - omni GPU → vllm:`OmniGPUWorkerBase` ← `vllm.Worker`;`OmniGPUModelRunner` ← `vllm.GPUModelRunner`
    - omni NPU → vllm-ascend:`OmniNPUWorkerBase` ← `vllm_ascend NPUWorker`(条件 310);`OmniNPUModelRunner` ← `vllm_ascend NPUModelRunner`;并 import `graph_capture` / `ascend_forward_context`
    - vllm-ascend → vllm:`NPUWorker` ← `WorkerBase`;`NPUModelRunner` ← `GPUModelRunner`
- **方向恒定**:依赖永远是 **omni → ascend → vllm**,反向没有 import。这保证 vllm-ascend 不需要知道 omni 的存在,vllm 不需要知道任何下游 —— 解耦的底线。

## 九、三个设计点小结

1. **Worker / ModelRunner 双链分工**:worker 管进程与设备(显存、KV、调度),runner 管计算(forward、采样、图)。三处目录里两条链各自独立继承,不要混看。
2. **条件继承(310P)**:`OmniNPUWorkerBase` 用 `is_310p()` 在 import 期换基类,把硬件变体藏进链条中段,叶子无感。
3. **菱形继承(omni × 昇腾)**:`OmniNPUModelRunner` 同时继承 omni-GPU runner 与 ascend-NPU runner(共同祖先 `GPUModelRunner`),这是 NPU 上既要 omni 特性、又要昇腾实现的**唯一缝合点**,也是整套结构里最该看懂的一行。

## 十、数据流对齐:隐式的格式契约,什么时候会 break

类能继承到一起、能实例化,**不等于数据能对上**。worker/runner 之间、stage 之间的数据流,还压着一批**不被类型系统约束、纯靠约定维系**的格式契约。它们的共同特点:**违反时多半不是干净报错,而是静默错算或 hang**。下面按"在哪对齐"分五层,行号对照源码。

!!! warning "先记住 break 的三档严重度"
    - **① 干净报错**(有断言/raise):好排查 —— 如 KV 必须 1-D uint8、device 不能混。
    - **② 静默错算**(无断言):最危险 —— 如 codes 布局错、Thinker 维度不匹配,**不 crash,出错误结果**。
    - **③ 死锁 hang**:集合通信不对齐。
    本节大多数契约属 **②**。

### 10.1 菱形的 `load_model` 路由(继承层对齐)

先破一个常见误解:`OmniNPUModelRunner` **没有定义 `__init__`**,所以实例化时走 MRO 第一位 `OmniGPUModelRunner.__init__`,它调 `super().__init__(*args, **kwargs)` —— 在菱形 MRO(`OmniNPUModelRunner → OmniGPUModelRunner → NPUModelRunner → GPUModelRunner`)里,这是**协作式 super**,会依次链过两个父类。**属性初始化是完整的,没有"父类 init 被跳过"的问题。**

真正的对齐点在 `load_model`(`npu_model_runner.py:37-53`):

```python
class OmniNPUModelRunner(OmniGPUModelRunner, NPUModelRunner):
    def load_model(self, *args, **kwargs) -> None:
        ...
        NPUModelRunner.load_model(self, *args, **kwargs)   # ← 显式点名父类,不是 super()
        enable_sp(self.vllm_config)                        # ↓ 手动补 omni-GPU 那套加载期 setup
        self._maybe_enable_output_token_ids_for_model_sampler()
        self._init_talker_mtp()
```

注意它**显式调 `NPUModelRunner.load_model(self, ...)`,而非 `super().load_model()`** —— 这是**故意绕开** `OmniGPUModelRunner.load_model`(后者会 `super().load_model()`,见 `gpu_model_runner.py:178`)。代价:omni-GPU 在 `load_model` 里做的加载期 setup **不会被 NPU 自动继承**,得像上面 line 48-53 这样**手抄一遍**。

- **break 条件**:将来 `OmniGPUModelRunner.load_model` 新增一步 setup,而 NPU 版忘了同步补 → **NPU 上静默缺这步**,不在 load 时报错,而是运行期某行为莫名缺失。这是 NPU 路径最容易腐化的地方。

### 10.2 Thinker → Talker:层选择 + dtype + 裁剪 + 维度(stage 对齐)

`stage_input_processors/qwen3_omni.py` 里藏着四个隐式契约:

| 契约 | 位置 | 内容 | break |
|---|---|---|---|
| **层魔数** | `:37-38, 216-217` | 固定取 layer `"0"`(embed)与 `"24"`(hidden) | 绑死特定模型层数,换模型/改层数即取错层 |
| **dtype** | `:216-217` | 统一 `.to(dtype=torch.float)` | 下游若按 bf16 假设算 → 精度/对齐错 |
| **stop 裁剪** | `:611-612` | `thinker_emb[:-1]` 砍掉最后一行(stop token) | 隐含"每步累一行、末行是 stop";若只有 1 行会把唯一行删空 |
| **维度匹配** | 投影(ResizeMLP) | Thinker `hidden_size` 必须 == Talker 投影输入维 | **无断言**,不匹配 → 投影处崩或静默错 |

```python
# qwen3_omni.py:216 —— 层魔数 + dtype,均无校验
p_emb = p_layers[int(_EMBED_LAYER_KEY)].detach().to(device=device, dtype=torch.float)   # "0"
p_hid = p_layers[int(_HIDDEN_LAYER_KEY)].detach().to(device=device, dtype=torch.float)  # "24"
```

### 10.3 codes 的 `[8, seq]` RVQ 布局(最隐蔽的静默错)

Talker → Code2Wav 这一跳,对码本张量的形状有**严格但无校验**的假设(`qwen3_omni.py:929`):

```python
codes = torch.cat(..., dim=0).transpose(0, 1).reshape(-1)
#        假定输入 [8, seq] ──┘  转 [seq, 8]  压平成交错序列喂 code2wav
```

`8` = RVQ 量化层数。这里假设输入是 **`[8, seq]`(层在前)**,转置后压成 `[rv0_t0, rv1_t0, …, rv7_t0, rv0_t1, …]` 的交错序列。

- **break 条件**:若 Talker 哪天输出成 `[seq, 8]`(时间在前),`transpose(0,1).reshape(-1)` **照样成功执行**,但语义彻底错位 → **合出错误音频,全程不报错**。这是②类里最阴险的:形状对、数值错。

### 10.4 跨 connector 边界:device / dtype / 连续性(数据面对齐)

张量过连接器要被序列化,这里有一组字节级契约:

```python
# distributed/omni_connectors/utils/serialization.py:90-92
if not t.is_contiguous():
    t = t.contiguous()          # 非连续 → 静默拷贝(性能损耗,不报错)
t = t.view(torch.uint8)         # 按字节重解释,要求 contiguous + 元素对齐
# dtype 以字符串存(如 "bfloat16"),解码端按 _SAFE_TORCH_DTYPES 还原;不支持的 dtype → raise
```

KV 打包则有**显式断言**(`kv_transfer_manager.py:177`):

```python
raise ValueError("Packed device KV payload must be a 1-D uint8 tensor")
```

- 隐式:非连续张量被**静默 contiguous 拷贝**(只是性能,无语义 break)。
- 显式①:KV 载荷必须 **1-D uint8**;且 device 必须**同构**(全 NPU 或全 GPU),混设备 → `.to(device=...)` 时 `RuntimeError`。

### 10.5 `execute_model` 的时序对齐(执行层)

`npu_ar_model_runner.py` 里有一条**仅靠注释维系**的顺序契约(`:344`):

```python
# [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(...)  # :369
...
deferred_state_corrections_fn = self._update_states(scheduler_output)  # :411
```

- **必须先 KV transfer,后 `_update_states()`**:因为 `_update_states` 会从 `input_batch` 移除 finished 请求;顺序反了 → req/block 索引错乱、**KV 污染**。**无断言,纯注释**(②/可能 crash)。
- **集合通信对齐**:有 KV transfer group 时,所有 rank 必须同步进入 `handle_preemptions`;漏一个 rank → 集合操作 **hang 死锁**(③)。

### 小结:契约 × 检查 × 症状

| 对齐点 | 位置 | 有检查? | 违反症状 | 档 |
|---|---|---|---|---|
| `load_model` 绕开 omni-GPU 父类 | `npu_model_runner.py:47` | 无 | NPU 静默缺加载期 setup | ② |
| Thinker 层魔数 0/24 | `qwen3_omni.py:37-38` | 无 | 取错层 | ② |
| Thinker↔Talker 维度匹配 | 投影层 | 无 | 投影崩 / 静默错 | ② |
| stop-token 裁剪 `[:-1]` | `qwen3_omni.py:612` | 无 | 单行被删空 | ② |
| codes `[8, seq]` 布局 | `qwen3_omni.py:929` | 无 | **错误音频,不报错** | ② |
| 序列化连续性 | `serialization.py:90` | 自动修 | 静默拷贝(性能) | — |
| KV 必须 1-D uint8 | `kv_transfer_manager.py:177` | **断言** | `ValueError` | ① |
| KV device 同构 | `.to(device)` | 隐式 | `RuntimeError` | ① |
| KV transfer 早于 `_update_states` | `npu_ar_model_runner.py:344` | 注释 | 索引错乱 / KV 污染 | ② |
| 多 rank `handle_preemptions` 同步 | KV transfer group | 无 | **死锁** | ③ |

**一句话**:这套数据流的脆弱点不在"类继承对不对",而在这些**没有类型/断言兜底的格式约定**——尤其 codes 布局、stage 间维度、`load_model` 路由这三处,错了都**不 crash**,只默默给出错误结果。改 Thinker 层数、换 Talker 投影维、调 RVQ 层数、或给 omni-GPU 的 `load_model` 加步骤时,**必须手动核对 NPU 侧是否对齐**。

## 关键文件 / 延伸阅读

- worker 基类:`vllm/v1/worker/worker_base.py` · `gpu_worker.py` · `gpu_model_runner.py`
- 昇腾适配:`vllm_ascend/worker/worker.py` · `model_runner_v1.py` · `_310p/`
- omni GPU:`vllm_omni/worker/{base,mixins,gpu_ar_worker,gpu_generation_worker,gpu_model_runner,omni_connector_model_runner_mixin}.py`
- omni NPU:`vllm_omni/platforms/npu/worker/{base,npu_ar_worker,npu_generation_worker,npu_model_runner,npu_ar_model_runner,npu_generation_model_runner}.py`
- [Omni 平台无关/相关解耦：现状与演进](platform-decoupling.md) —— 菱形继承 / 工厂 / hook 的机制背景
- [Qwen3-Omni 在 NPU 上是怎么跑起来的](qwen3-omni-npu.md) —— AR / Generation 三段式的来历
- [npu_model_runner 上游适配困境与解耦](snippets/npu-runner-decoupling.md) —— runner 层适配的具体痛点
