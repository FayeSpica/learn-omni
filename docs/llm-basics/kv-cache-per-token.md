---
tags:
  - LLM基础
  - KV-Cache
  - 显存
  - 推理
  - H100
---

# 给定 config.json 与 H100，算「每生成一个 Token」的 KV Cache 显存

> 一道入门必做题：**拿到一个模型的 `config.json`、一张 H100，怎么算出生成阶段每多吐一个 token、KV Cache 要多吃多少显存?**
>
> 本文从「KV Cache 是什么 → 公式推导 → 从 config.json 取参数 → 实算 Llama-3.1-8B → GQA 省了多少 → 放到 H100 上能存多少」一条线走完，并标出 MLA/explicit head_dim 等常见坑。

## 结论先行

**每生成一个 token，单条序列新增的 KV Cache 显存：**

$$
\text{bytes/token} = 2 \times L \times n_{kv} \times d_{head} \times b
$$

| 符号 | 含义 | config.json 字段 |
|---|---|---|
| `2` | K 和 V 两份 | —（常数） |
| $L$ | 层数 | `num_hidden_layers` |
| $n_{kv}$ | KV 头数（GQA/MQA 下 < 注意力头数） | `num_key_value_heads` |
| $d_{head}$ | 每个头的维度 | `head_dim`，或 `hidden_size / num_attention_heads` |
| $b$ | 每个元素字节数 | bf16/fp16 = 2，fp8 = 1 |

**以 Llama-3.1-8B、bf16 为例 = `128 KiB/token`。** 推导见下。

## 一、KV Cache 是什么，为什么每个 token 都要存

自回归生成时，第 $t$ 个 token 的注意力要和**前面所有 token** 的 Key、Value 做计算。如果每步都重算前文的 K/V，复杂度是 $O(t^2)$ 的重复劳动。

**KV Cache 就是把每个 token 在每一层算出的 K、V 向量存下来**，下一步直接复用。代价是显存：缓存随序列长度**线性增长**，每新增一个 token，就要为它在**每一层**存一份 K 和一份 V。

> 所以「每 token 显存」= 一个 token 在所有层的 K + V 占用。这正是上面公式里没有「序列长度」的原因——它算的是**增量**，乘上序列长度才是总量。

## 二、公式推导

单个 token、单层、单个 KV 头，存一个 K 向量需要 $d_{head}$ 个元素，V 同理。于是：

```
一个 token 一层一个 KV 头 :  K(d_head) + V(d_head) = 2 · d_head 个元素
一个 token 一层所有 KV 头 :  2 · d_head · n_kv
一个 token 所有层        :  2 · d_head · n_kv · L
换算成字节              :  2 · d_head · n_kv · L · b
```

注意是 $n_{kv}$（KV 头数）**不是** $n_{q}$（注意力头数）。MHA 时两者相等；**GQA/MQA 时 $n_{kv}$ 小很多**，这是省显存的关键。

## 三、从 config.json 取参数

以 Llama-3.1-8B 的 `config.json`（节选）：

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "torch_dtype": "bfloat16"
}
```

- $L$ = `num_hidden_layers` = **32**
- $n_{kv}$ = `num_key_value_heads` = **8**（GQA：32 个 Q 头共享 8 个 KV 头，每 4 个 Q 头共用一组 KV）
- $d_{head}$ = 没有 `head_dim` 字段，按 `hidden_size / num_attention_heads` = 4096 / 32 = **128**
- $b$ = bf16 = **2 字节**

## 四、实算 Llama-3.1-8B

$$
2 \times 32 \times 8 \times 128 \times 2 = 131072 \text{ 字节} = \mathbf{128\ KiB/token}
$$

逐步：

```
2 × 32      = 64
64 × 8      = 512
512 × 128   = 65536
65536 × 2   = 131072 字节 = 128 KiB
```

直觉换算：

| 量 | 大小 |
|---|---|
| 每 token | 128 KiB |
| 一条 8K 上下文序列 | 128 KiB × 8192 ≈ **1 GiB** |
| 一条 128K 上下文序列 | 128 KiB × 131072 = **16 GiB** |

> 一条满 128K 的序列，光 KV Cache 就吃掉 16 GiB——和 8B 模型权重本身一样大。这就是长上下文显存压力的来源。

## 五、GQA 省了多少

如果 Llama-3.1-8B 用传统 MHA（$n_{kv} = n_q = 32$）：

$$
2 \times 32 \times 32 \times 128 \times 2 = 524288 = 512\text{ KiB/token}
$$

GQA（$n_{kv}=8$）相比 MHA（$n_{kv}=32$）**KV Cache 直接降到 1/4**（512 → 128 KiB）。MQA（$n_{kv}=1$）更极端，降到 1/32。这就是现代模型几乎都用 GQA 的核心动机——**用极小的精度损失换 KV Cache 显存与解码带宽**。

## 六、放到 H100 上：80 GB 能存多少 token

H100 SXM 关键参数：

| 参数 | 值 | 跟这题的关系 |
|---|---|---|
| HBM3 容量 | **80 GB** | 决定能存多少 KV（容量上限） |
| HBM 带宽 | ~3.35 TB/s | 决定解码速度（下一题：解码是访存受限） |

显存预算粗算（实际推理框架如 vLLM 用 `gpu_memory_utilization` 控制）：

```
总预算 (利用率 0.9)   ≈ 80 GB × 0.9      = 72 GiB
减模型权重 (8B × 2B)  ≈ 16 GiB
减激活/碎片等开销     ≈ 几 GiB
留给 KV Cache        ≈ 56 GiB（保守估）
```

能缓存的 token 总数：

$$
\frac{56\ \text{GiB}}{128\ \text{KiB/token}} = \frac{56 \times 1024 \times 1024\ \text{KiB}}{128\ \text{KiB}} \approx 4.6\times10^{5}\ \text{tokens}
$$

**约 45 万 token 的 KV 预算**。这个数字直接决定并发能力：

- 64 条并发、各 7K 上下文 ≈ 45 万 token → 刚好吃满；
- 想跑更长上下文或更高并发，就得 KV 量化（fp8 砍半）、更激进的 GQA、PagedAttention 减碎片，或换 MLA。

## 七、常见坑

1. **explicit `head_dim`**：别默认 $d_{head} = hidden/heads$。Qwen3 等模型 `config.json` 里**显式给 `head_dim`**，且可能不等于 `hidden_size / num_attention_heads`。有就以字段为准。
2. **MLA 模型不适用本公式**：DeepSeek-V2/V3 用 **MLA（多头潜在注意力）**，每 token 每层只存一份**压缩潜向量** $c_{KV}$（`kv_lora_rank`）+ 一小段 RoPE（`qk_rope_head_dim`），不按头数展开：
   $$
   \text{MLA bytes/token} \approx L \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}) \times b
   $$
   DeepSeek-V3（$L$=61, lora=512, rope=64, bf16）≈ 61 × 576 × 2 ≈ **68 KiB/token**——一个 671B 的模型，每 token KV 竟比 8B 的 MHA 还小。这是 MLA 的杀手锏。
3. **fp8 KV Cache**：`kv_cache_dtype=fp8` 时 $b=1$，KV 直接砍半，但可能掉点，需评估。
4. **MoE 不影响 KV**：KV Cache 只和注意力相关，专家数/激活专家数**不进公式**（MoE 省的是权重计算，不是 KV）。
5. **滑动窗口 / 局部注意力**：Gemma、部分 Mistral 变体用 sliding window，KV 上限被窗口截断，不随序列无限增长。
6. **MTP / 投机解码**：会临时多存草稿 token 的 KV，属额外开销，不在基础公式内。

## 八、多模型对比（bf16，每 token KV）

| 模型 | $L$ | $n_{kv}$ | $d_{head}$ | 每 token KV | 注意力 |
|---|--:|--:|--:|--:|---|
| Qwen2.5-7B | 28 | 4 | 128 | **56 KiB** | GQA |
| Llama-3.1-8B | 32 | 8 | 128 | **128 KiB** | GQA |
| Mistral-7B | 32 | 8 | 128 | **128 KiB** | GQA |
| Llama-3.1-70B | 80 | 8 | 128 | **320 KiB** | GQA |
| Qwen2.5-72B | 80 | 8 | 128 | **320 KiB** | GQA |
| DeepSeek-V3 (671B) | 61 | — | — | **~68 KiB** | MLA（另一公式） |

> 注意：KV Cache 大小由 $L \times n_{kv} \times d_{head}$ 决定，**和总参数量不强相关**。72B 的 KV（320 KiB）只是 8B（128 KiB）的 2.5 倍；而 671B 的 DeepSeek-V3 靠 MLA 反而最省。

## 小结

- 公式就一行：$2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot b$，记住「2 份、按 KV 头数、所有层、增量」。
- 从 config.json 取 `num_hidden_layers` / `num_key_value_heads` / `head_dim`(或 hidden/heads) / dtype 四个量即可。
- H100 的 80 GB 把「每 token KV」放大成「能存多少 token / 支持多少并发」的容量问题。
- 例外要会判：explicit head_dim、MLA、fp8 KV、sliding window。

!!! info "下一题预告"
    有了「每 token KV」，自然接着问：**一次请求的总 KV（prefill + decode）怎么算?H100 上能同时跑多少条并发?为什么解码速度被 HBM 带宽卡住而不是算力?** —— 留待本系列后续。
