---
tags:
  - 昇腾
  - vllm-ascend
  - 量化
  - 速查
---

# 昇腾(vllm-ascend)量化特性支持速查

> 速查:vllm-ascend 上 **W8A8 / W4A8 / W4A4 / W4A16 / MXFP8 / MXFP4 / FP8 / KV C8** 等量化方案各自的支持情况(按 Linear / MoE 层区分)。基于 `vllm_ascend/quantization/` 源码整理,可能随版本变化,以实际仓库为准。

## 一、三类量化配置入口

平台对外声明的 `supported_quantization`(`platform.py`)有四个入口:

| 量化方法名 | 配置类 | 来源/格式 |
|---|---|---|
| `ascend` | `AscendModelSlimConfig` | 华为 **ModelSlim** 工具量化出的权重(昇腾主路径) |
| `compressed-tensors` | `AscendCompressedTensorsConfig` | 社区 compressed-tensors 格式 |
| `fp8` | `AscendFp8Config` | FP8 |
| `deepseek_v4_fp8` | — | DeepSeek-V4 FP8 专路 |

大多数昇腾量化权重走 **ModelSlim(`ascend`)** 路径:checkpoint 里带量化描述,`quant_parser` 解析出每层的 `quant_type`,再到 scheme 注册表 `get_scheme_class(quant_type, layer_type)` 找具体实现。

## 二、支持矩阵(scheme × 层类型)

下表来自 `vllm_ascend/quantization/methods/` 里所有 `@register_scheme(quant_type, layer_type)` 注册项——这是**真实支持的组合**:

| 量化方案 | Linear | MoE | 说明 |
|---|:--:|:--:|---|
| **W8A8**(静态) | ✓ | — | 权重+激活 int8 静态量化;MoE 走 W8A8_DYNAMIC |
| **W8A8_DYNAMIC** | ✓ | ✓ | int8 动态(激活在线量化),Linear+MoE 全支持 |
| **W8A8_MXFP8** | ✓ | ✓ | MX 格式 FP8(块级 scale) |
| **W8A8_MIX**(pdmix) | ✓ | ✓ | PD 分离场景的混合精度 |
| **W8A16** | ✓ | — | weight-only int8(仅权重量化,激活 fp16) |
| **W4A16** | — | ✓ | weight-only int4,**仅 MoE**(典型用于大 MoE 省显存) |
| **W4A8_DYNAMIC** | ✓ | ✓ | 权重 int4 / 激活 int8 动态 |
| **W4A8_MXFP**(W4A8MXFP) | ✓ | ✓ | W4A8 的 MX 浮点变体 |
| **W4A4_DYNAMIC**(laos) | ✓ | — | 权重+激活 int4 动态,**仅 Linear** |
| **W4A4_FLATQUANT_DYNAMIC** | ✓ | — | W4A4 + FlatQuant(等价变换降量化误差),仅 Linear |
| **W4A4_MXFP4** | ✓ | ✓ | MX 格式 FP4,Linear+MoE |
| **W4A4_MXFP4_FLATQUANT** | ✓ | — | MXFP4 + FlatQuant,仅 Linear |
| **FP8** | ✓(ds_linear) | ✓(w4a8_moe) | DeepSeek Linear + W4A8 MoE 形态 |
| **KV C8**(FAKQuant / INT8_DYNAMIC) | — | — | attention 层:**KV cache int8 量化** |

> 读法:`✓` 表示该 `(quant_type, layer_type)` 有注册实现。`—` 表示该层类型没注册对应 scheme(通常意味着该方案不覆盖这类层,或由另一变体承担,如 W8A8 静态的 MoE 由 W8A8_DYNAMIC 处理)。

### 关键结论

- **W8A8**:支持最全(静态/动态/MXFP8/混合),Linear 与 MoE 都覆盖,是昇腾上最成熟的方案。
- **W4A4**:**Linear 覆盖好,MoE 仅 MXFP4 一种**(`W4A4_MXFP4`)。纯 int4 的 W4A4_DYNAMIC / FlatQuant 变体目前**只支持 Linear**。
- **W4A16 / W8A16**:weight-only;W4A16 反而**只给 MoE**,W8A16 只给 Linear——两者互补。
- **MX 格式(MXFP8 / MXFP4)**:需要硬件/库支持 MX block-scaled 浮点,使用前确认 SOC 版本。
- **KV cache** 可单独 int8 量化(C8),与权重/激活量化正交,通过 `kv_cache_dtype` 触发。

## 三、怎么启用

1. **权重侧**:用华为 **ModelSlim** 离线量化产出 checkpoint(权重已是 int4/int8 + scale/offset,带量化描述文件)。
2. **加载侧**:vllm-ascend 自动识别 `quantization="ascend"`(或显式传 `--quantization ascend`),`quant_parser` 按层解析 quant_type 并匹配 scheme。
3. **KV cache**:`--kv-cache-dtype int8`(C8)单独开启。
4. MoE 还涉及 `dynamic_eplb`、专家并行等,量化实现里有对应的融合 GEMM(如 `quant_apply_mlp` + SwiGLU 融合)。

> 量化算子最终落到 `torch_npu` / 自定义 kernel:int4 权重在 NPU 上以 int32 打包(`pack_to_int32` / `unpack_from_int32`),激活动态量化用 `npu_dynamic_quant`,内部张量格式可能 ND→FRACTAL_NZ。

## 四、文件索引

| 内容 | 文件 |
|---|---|
| scheme 注册表 | `vllm_ascend/quantization/methods/registry.py` |
| 各 scheme 实现 | `vllm_ascend/quantization/methods/{w8a8_static,w8a8_dynamic,w8a8_mxfp8,w8a8_pdmix,w8a16,w4a16,w4a8,w4a8_mxfp4,w4a4_laos_dynamic,w4a4_flatquant,w4a4_mxfp4,w4a4_mxfp4_flatquant,fp8,kv_c8}.py` |
| MoE 量化枚举 | `vllm_ascend/quantization/quant_type.py`(`QuantType`) |
| 配置类 | `quantization/{modelslim_config,compressed_tensors_config,fp8_config}.py` |
| 解析器 | `vllm_ascend/quantization/quant_parser.py` |
| 平台声明 | `vllm_ascend/platform.py`(`supported_quantization`) |

---

!!! info "说明"
    支持矩阵来自源码 `@register_scheme` 注册项快照,随版本更新可能增减(尤其 W4A4 的 MoE 支持在持续补齐)。落地前建议在目标版本仓库重新 `grep -rn "@register_scheme" vllm_ascend/quantization/methods/` 核对。

    **硬件视角**:本篇是**软件量化方案**;这些方案能否拿到原生算力取决于硬件代次,参见[昇腾代次与原生低精度格式(A2/A3/A5·950)](ascend-generations-low-precision.md)。相关:[Qwen3-Omni 在 NPU 上是怎么跑起来的](../../vllm-omni/qwen3-omni-npu.md)。
