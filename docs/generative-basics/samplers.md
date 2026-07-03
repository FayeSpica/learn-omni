---
tags:
  - 生成模型基础
  - 采样器
  - scheduler
  - 扩散模型
  - flow-matching
  - Euler
  - UniPC
---

# 采样器：从 DDPM 到 Flow Matching（UniPC / Euler）

> 两个问题：**采样器(scheduler)到底管什么?为什么少步蒸馏模型非用 Euler 不可?**
>
> 本文从定义出发，串起 DDPM → DDIM → DPM-Solver/UniPC → Flow Matching/Euler 这条演化线，落点到 vllm-omni diffusion 引擎里 Wan2.2 的 `sample_solver` 开关。接 [DiT 笔记](dit.md) 的第五节「现代变体」往下钻。

## 一句话

**采样器 = 扩散去噪循环里的「步进规则」**——模型只负责在每一步「看着带噪样本、预测噪声/速度」，而**怎么用这个预测把 $z_t$ 推进到更干净的 $z_{t-1}$、一共走几步、每步跨多大**，全由采样器决定。

关键认知:**采样器与模型解耦**。同一个 DiT 权重，换不同采样器就能在「50 步高画质」和「4 步快出」之间切换——这正是 PR #2134 给 Wan2.2 加 `sample_solver=euler` 开关的前提。

## 一、它在流程里的位置

```mermaid
graph LR
    Z["带噪 latent z_t"] --> M["DiT forward<br/>预测 噪声ε / 速度v"]
    T["时间步 t / 条件 c"] --> M
    M --> S["采样器 step()<br/>据预测推进一步"]
    S -->|z_{t-1}| Z
    S -->|循环 N 步后| Z0["干净 latent z_0"]
```

**模型每步只回答「这里有多少噪声/往哪走」;采样器回答「那我该迈多大一步」。** 循环 N 次，N 就是 `num_inference_steps`。

## 二、演化线：一堵墙逼出下一个采样器

每一代采样器都是被上一代的「步数太多 / 太慢」逼出来的。

| 采样器 | 核心思想 | 典型步数 | 被什么逼出来 |
|---|---|---|---|
| **DDPM** (2020) | 反向**随机**马尔可夫链，每步去一点噪 + 注入随机 | ~1000 | 原始定义，慢到不能用 |
| **DDIM** (2021) | 改成**确定性**、非马尔可夫，可跳步 | 20~50 | DDPM 太慢，要能跳步 |
| **DPM-Solver / ++** (2022) | 把反向过程看成 **ODE**，用**高阶数值解法** | 10~20 | DDIM 一阶，低步数精度不够 |
| **UniPC** (2023) | 统一的**预测-校正**高阶多步解法，低步收敛更稳 | 8~20 | 追求更少步 + 更稳 |
| **Euler (flow matching)** | 学一条噪声→数据的**直线速度场**，最简一阶步进 | **1~8** | 换掉整个「加噪/去噪」范式，为极少步铺路 |

前四个是「**在扩散框架内**优化解 ODE 的方法」;最后一个 Flow Matching 是**换了框架**——下一节单讲。

## 三、Flow Matching 与 Euler（本 PR 的落点）

**Flow Matching / Rectified Flow**(SD3、Flux、Wan 用的就是这套)换了个思路:

- 传统扩散想「一步步把噪声剥掉」;flow matching 想「在噪声 $z_1$ 和数据 $z_0$ 之间画一条**尽量直的路径**，让模型学这条路径上每一点的**速度(velocity) $v$**」。
- 采样 = 沿速度场解一个 ODE:$\dfrac{dz}{d\sigma}=v$。$\sigma$ 是噪声水平(1→0)。

**Euler** 就是解这个 ODE 最朴素的一阶方法。看 PR #2134 新增的 `scheduling_wan_euler.py`，`step()` 的核心就一行:

```python
prev_sample = sample + (sigma_next - sigma) * model_output
#   下一步   =  当前  + (下一噪声水平 - 当前)  ×  模型预测的速度
```

一步一个直线外推，没有高阶项、没有历史缓冲——**极其简单，正是它能在 4 步稳定工作的原因**。

### flow_shift：把步数花在刀刃上

路径虽近似直线，但噪声区间对画质的贡献不均。`flow_shift` 把 $\sigma$ 调度「扭一下」，让更多步落在关键区间:

```python
sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
```

经验值:**720p 用 5.0，480p 用 12.0**(PR 里到处出现这两个数就是这个原因)。

## 四、为什么少步蒸馏「必须」换 Euler

这是最容易被忽略、却最关键的因果点，单独讲清:

1. **蒸馏(distill)时的假设**:LightX2V 这类 4 步蒸馏 LoRA，是**配着简单 Euler 步进训练的**——学生被训练成「用 4 个 Euler 直线步复现老师 50 步的结果」。
2. **UniPC 的假设**:高阶多步解法假设相邻步之间足够平滑、可以用历史步外推高阶项。这在 50 步时成立。
3. **冲突**:在 **4 步** + 蒸馏后的速度场上，UniPC 的高阶假设**不成立**，外推会放大误差 → 画质崩。
4. **结论**:必须让**推理采样器与蒸馏时的假设一致** → 用 Euler。

> 一句话记住:**采样器要和模型「训练时假定的解法」对齐**。默认权重配 UniPC，蒸馏权重配 Euler，错配就崩。这就是 PR #2134 存在的核心动机。

配套参数也都服务于这条链(详见 [DiT 笔记](dit.md) 与后续 CFG 篇):

- `num_inference_steps=4` —— 蒸馏换来的少步;
- `guidance_scale=1.0` —— 关掉 CFG(蒸馏已把引导烘焙进去，再叠加会坏);
- `boundary_ratio=0.875` —— Wan2.2 双 transformer(high/low-noise)的切换点，与采样器正交但同在一次去噪里。

## 五、在 vllm-omni 里怎么切

采样器在 pipeline 里可**按请求热重建**:请求把 `sample_solver` / `flow_shift` 放进 `sampling_params.extra_args`(在线经 `extra_params` JSON 合并)，`forward()` 里解析，若与当前不同就重建 scheduler，不必重启服务。

```python
# pipeline_wan2_2*.py 的思路
sample_solver = resolve_wan_sample_solver(req, default="unipc")
flow_shift    = resolve_wan_flow_shift(req, od_config)
if sample_solver != self._sample_solver or flow_shift != self._flow_shift:
    self.scheduler = build_wan_scheduler(sample_solver, flow_shift)  # unipc / euler
```

## 小结

| 维度 | 采样器(scheduler) |
|---|---|
| 管什么 | 去噪循环的步进规则:用几步、每步迈多大 |
| 不管什么 | 预测噪声/速度(那是模型/DiT 的事) |
| 与模型关系 | **解耦**;但必须与「训练时假定的解法」对齐 |
| 主流演化 | DDPM → DDIM → DPM-Solver/UniPC → Flow Matching(Euler) |
| 默认权重 | UniPC(高阶多步，20~50 步高画质) |
| 蒸馏权重 | **Euler**(一阶直线，4 步;错配 UniPC 会崩) |
| 关键旋钮 | 步数、`flow_shift`、(CFG 的)guidance_scale |

!!! info "与其它笔记的关系"
    本文讲「采样器本身」;骨干模型见 [DiT](dit.md)，承载它的算子分发见 [vLLM IR：CustomOp](../vllm/vllm-ir-and-customop.md)。VAE(latent 空间)、CFG(引导原理与算力代价)是本页的两个近邻，仍在 [板块目录](index.md) 的待补清单里——建议接着补 CFG。
