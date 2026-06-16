---
tags:
  - vllm
  - vllm-omni
  - vLLM-IR
  - CustomOp
  - torch.compile
  - 架构
---

# vLLM IR 是什么：从 CustomOp 的困境说起

> 两个问题：**vLLM IR 是什么？以及 vllm-omni 关于 CustomOp 的那场讨论（[issue #1030](https://github.com/vllm-project/vllm-omni/issues/1030)）在争什么？**
>
> 这篇承接 [Omni 平台无关/相关解耦：现状与演进](../vllm-omni/platform-decoupling.md)——那篇的结论是「算子层该从 `if is_npu()` 收敛到注册表」。本文给出更完整的答案：**上游 vLLM 不止做注册表，而是进一步做 vLLM IR；vllm-omni 的 diffusion CustomOp 讨论，最终也指向「与 vLLM IR 对齐」。** 内容基于两份 RFC（vllm-omni #1030、vllm [#32358](https://github.com/vllm-project/vllm/issues/32358)）与 `vllm/ir/` 源码。

## 一句话

**vLLM IR 是一套「函数式中间表示」：把一个算子的 *语义*（semantics）、*实现*（implementation）、*分发*（dispatching）三者彻底拆开。** 算子作为 torch FX 图里的一个「方言（dialect）」节点存在——编译期可以对它做融合/改写而不关心选了哪个实现，编译后再 *lowering* 到选中的具体 kernel。它要替代的，正是现在 `CustomOp` 那套「`if is_cuda()/is_rocm()` 选 `forward_*`」的笨重分发。

---

## 二、先看痛点：CustomOp 现在怎么分发

vLLM（和 vllm-omni 照搬的）`CustomOp` 用「平台分派」选实现：

```python
class CustomOp(nn.Module):
    def __init__(self):
        self._forward_method = self.dispatch_forward()

    def dispatch_forward(self):
        if current_platform.is_rocm():  return self.forward_hip
        elif current_platform.is_cuda(): return self.forward_cuda
        elif current_platform.is_xpu():  return self.forward_xpu
        # ...
        else:                            return self.forward_native
```

它能跑，但 RFC 列了几条硬伤：

1. **编译融合难**：`RMSNorm`、`Quant` 这类算子在 `torch.compile` 里要么 decompose 成一串脆弱的 torch 小算子，要么是各式各样的自定义 kernel——融合 pass 得为每种情况单独处理，分发逻辑「复杂、笨重、可见性低」。
2. **跨平台性能调优冲突**（omni #1030 的问题 1）：同一个算子（如 Triton 实现的 rope）在不同硬件上最优策略不同。比如 ROCm 想把 `rope(x); rope(y)` 融成 `rope(joint_xy)`，但这在 CUDA 上可能反而变慢。硬塞进 CustomOp 会让代码迅速膨胀。
3. **加新算子要侵入改模型代码**（omni #1030 的问题 2）：现在换一个算子实现得直接改模型源码。
4. **上游算子 API 与 diffusion 不匹配**（omni #1030 的问题 3）：像 `fused_qk_norm_rope` 是为自回归生成设计的，签名里带 `position_ids`；diffusion 没有 chunk-prefill / token 生成，根本不需要它，却被迫层层加胶水代码把这个冗余参数传下去。

---

## 三、vLLM IR 怎么解：语义 / 实现 / 分发三分离

### 1. 注册一个 op = 写它的「语义」（同时就是 native 默认实现）

```python
# vllm/ir/ops/layernorm.py
@register_op
def rms_norm(x, weight, epsilon, variance_size=None):
    """Weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return (x.to(weight.dtype) * weight).to(orig_dtype)
```

这个纯 torch 函数**既是规格说明，也是 `native` 兜底实现**。注册时（`IrOp.__init__`）它会：用 `infer_schema` 推出算子 schema、把自己注册进 `IrOp.registry`、并通过 `torch.library` 在 `torch.ops.vllm_ir` 命名空间下 `define` 一个真正的 torch 自定义 op（`dispatch_key="CompositeExplicitAutograd"`，保证 AOTAutograd 里不被拆解）。

### 2. 注册「实现」= 给某个 provider 挂一份 kernel

```python
# vllm/custom_kernels/custom.py
@ir.ops.rms_norm.register_impl(provider="vllm_c")
def rms_norm(x, weight, epsilon):
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
```

- `register_impl(provider, *, supported, supports_args, inplace)`：
  - `supported`：**静态**平台/库可用性检查（如 `torch.cuda.is_available()`）——只判平台能力，不许塞环境变量等业务开关。
  - `supports_args`：**动态**判断「这组 shape/dtype 我这份实现能不能吃」，签名必须和 native 完全一致（连参数名、默认值都校验），为的是热路径分发够快。
  - 实现的 schema **必须**和 native 完全相同——`IrOpImpl.__init__` 里强校验，从根上杜绝「不同平台 API 漂移」。

### 3. 「分发」= 按优先级列表选，而不是 if 链

不再 `if is_cuda()`，而是每个 op 有一份 **priority list**（来自 `VllmConfig`，平台给默认值，用户可覆盖）：

```python
def dispatch(self, *args, **kwargs) -> IrOpImpl:
    if not self._priority_impls:
        return self.impls["native"]          # 没设优先级 → native 兜底
    for impl in self._priority_impls:
        if impl.supports_args(*args, **kwargs):
            return impl                       # 第一个吃得下这组参数的赢
    # 优先级列表最后一个必须能吃所有参数（native 可做兜底）
```

配置方式直观到命令行就能调 kernel：

```bash
vllm serve Qwen/Qwen-0.6B --kernel-config.ir_op_priority.rms_norm=vllm_c
```

还有 `set_priority()`（上下文管理器，作用域内临时覆盖）和 `set_default()`（进程级，worker 启动时一次性设定）。

### 4. 模型代码：只写语义，不碰平台

```python
# 模型 forward 里直接调，平台/kernel 全透明
vllm.ir.ops.rms_norm(x, self.weight, self.epsilon)
```

模型作者从此**不需要知道有几种实现、当前在什么硬件上**——这正解了 omni #1030 的「加算子要改模型代码」。

### 5. 编译期：先当不透明节点做融合，再 lowering

- 开 `torch.compile` 时（`_ENABLE_TORCH_WRAP=True`，默认），IR op 在 FX 图里就是一个 `torch.ops.vllm_ir.*` 节点。融合 pass 可以基于「语义节点」做跨层改写（match `rms_norm` + `quant` → 融合），**完全不关心底下选了哪个 provider**。
- 编译流水线后段再把节点 *lowering* 到选中的实现，吃满 Inductor 优化。
- 不用 Inductor 的平台（如 NPU）可以 `enable_torch_wrap(False)`，跳过 torch op 包装层、直接分发到实现，省掉 dispatch 开销、也不需要 lowering。

### 6. 顺带白拿的能力

- **autotuning**：`register_input_generator` + `override_tolerance`（见 `layernorm.py`）让框架能自动生成输入、按数值容差对不同实现做正确性校验与自动调优。
- **inplace 支持**：`allow_inplace=True` 生成 `maybe_inplace` 重载；默认重载保持函数式（`func_impl_fn` 会 clone 活化输入），兼顾「编译要纯函数」与「运行要省显存」。
- **编译缓存**：`IrOpImpl.uuid` 对实现源码做哈希，驱动 vllm-compile / Inductor 的缓存失效。

---

## 四、IR vs CustomOp：一张对照表

| 维度 | CustomOp（现状） | vLLM IR |
|---|---|---|
| 分发 | `if is_cuda()` 选 `forward_*`，散在基类 | 单一事实源：per-op priority list |
| 语义 vs 实现 | 耦合在同一个类的方法里 | 彻底分离（native 即语义） |
| 加实现 | 改类、可能改模型代码 | `@op.register_impl(provider=...)`，in-tree / OOT 都行 |
| 编译融合 | decompose 成脆弱 torch 序列，难匹配 | FX 里是稳定的语义节点，先融合后 lowering |
| 选 kernel | 代码里写死 | 配置/命令行 `ir_op_priority.<op>=<provider>` |
| 自动调优 | 无 | input_generator + tolerance 内建 |
| API 漂移 | 各平台 forward 可能签名不一 | schema 强校验，所有实现必须与 native 一致 |

> RFC 强调：IR **可非侵入式接入**——不改模型定义、对 OOT 的 `CustomOp` 注册提供平滑迁移路径，作为 FX 的一个 dialect 与普通 torch op / 现有 custom op 完全互操作，支持**逐步迁移**。

---

## 五、回到 vllm-omni #1030：这场讨论在争什么

issue #1030 是一份 **《[RFC]: Custom Op for diffusion》**，提出 diffusion 自定义算子越来越多、现有 CustomOp 平台分派机制「不够用」（上面四条痛点）。它抛出三个讨论方向：

1. 怎么设计既支持平台分派、又支持跨平台性能调优的算子抽象？
2. diffusion 要不要上 `torch.compile` custom pass？
3. 怎么让外部算子（如 vLLM 的）自然融入 diffusion，避免 API 不匹配和改模型文件？

讨论里的关键走向（评论区）：

- 有人建议短期先用 **「模型 init 时做 module replacement」** 作为务实方案（不需要 FX 专业知识、好调试），等 `torch.compile` 对 diffusion 成熟了再迁到 **FX passes**（`subgraph_rewriter` 自动在编译期替换算子、模型代码零改动）。
- Intel 的人直接点出：**「vLLM 正要把 CustomOp 迁到 vLLM IR 方案（#32358），能不能对齐？」**
- 维护者回应：**「等我们要支持 torch.compile custom pass 时就会对齐」**，并表示 **「更倾向与 vLLM 上游对齐」**，团队已在和做 vLLM IR 的人（Luka / ProExpertProg）讨论；vLLM IR 作者也表态「乐意讨论 IR 如何支持 vLLM-omni 的分发」。

**所以结论很清楚**：omni 的 diffusion CustomOp 问题，社区共识不是自己再造一套分发，而是**向上游 vLLM IR 收敛**。

---

## 六、和「平台解耦」那篇怎么衔接

前一篇我总结 omni 现状是「worker/模型已工厂化、算子/量化还在 if 化」，演进方向是「收敛成 hook+注册表，向 `CustomOp.register_oot` 靠拢」。

这篇补上了**更远的那一站**：

```
omni 现状           →   上游 CustomOp.register_oot   →   vLLM IR（终点）
if is_npu() 散落         单点注册表 + forward_oot         语义/实现/分发三分离
                                                         + FX 融合 + 自动调优
```

- `diffusion/attention/backends/abstract.py`、`quantization/*_config.py`、`diffusion/layers/custom_op.py` 里那些 `if is_npu()/is_cuda()` 分支，是 IR priority list 的天然替换对象。
- omni #1030 问题 3（`fused_qk_norm_rope` 的 `position_ids` 胶水）正是 IR「schema 强校验 + 语义分离」要消灭的东西。
- 落地节奏取决于 omni 何时支持 `torch.compile` custom pass——维护者已明确把「对齐 IR」挂在这个里程碑上。

---

## 七、关键引用

| 内容 | 出处 |
|---|---|
| vLLM IR RFC（动机、设计、迁移） | vllm-project/vllm [#32358](https://github.com/vllm-project/vllm/issues/32358) |
| Omni diffusion CustomOp RFC | vllm-project/vllm-omni [#1030](https://github.com/vllm-project/vllm-omni/issues/1030) |
| IR 核心实现（IrOp / register_op / register_impl / dispatch / priority / lowering） | `vllm/vllm/ir/op.py` |
| IR op 实例（rms_norm / fused_add_rms_norm + input_generator + tolerance） | `vllm/vllm/ir/ops/layernorm.py` |
| 现有 CustomOp 分派（被替代对象） | `vllm/vllm/model_executor/custom_op.py`；`vllm-omni/vllm_omni/diffusion/layers/custom_op.py` |
| 相关笔记 | [Omni 平台无关/相关解耦：现状与演进](../vllm-omni/platform-decoupling.md) |

---

!!! info "说明"
    本文基于两份 RFC 与当时 `vllm/ir/` 源码快照整理；vLLM IR 仍在演进（RFC 反馈期 1/14–1/25），接口与落地细节可能变化，以上游最新为准。详细设计另见 RFC 内链接的 Google Doc / Slides。
