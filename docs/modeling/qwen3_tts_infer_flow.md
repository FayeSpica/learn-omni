# Qwen3-TTS 在 vLLM-Omni 中的推理流程详解

## 0. HF checkpoint 如何加载成可执行的模型？

本文从 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 的 HF checkpoint 开始，依次说明文件内容、模型创建、参数加载，以及文本到音频的完整计算流程。

### 0.1 HF 仓库中有哪些文件？

以下目录取自模型 revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`，列出推理相关文件：

```text
Qwen3-TTS-12Hz-1.7B-CustomVoice/
|-- config.json
|-- generation_config.json
|-- model.safetensors
|-- tokenizer_config.json
|-- vocab.json
|-- merges.txt
|-- preprocessor_config.json
`-- speech_tokenizer/
    |-- config.json
    |-- configuration.json
    |-- model.safetensors
    `-- preprocessor_config.json
```

| 文件 | 保存的内容 | 推理时的用途 |
|---|---|---|
| 根目录 `config.json` | 模型类型、Talker 与 CodePredictor 的层数和维度、特殊 ID、speaker ID | 创建 Stage 0 模型模块 |
| 根目录 `model.safetensors` | `talker.*` 参数张量 | 加载 Talker、文本 embedding/projection、codec head 和 CodePredictor |
| `tokenizer_config.json`、`vocab.json`、`merges.txt` | 文本 tokenizer 配置、词表与合并规则 | 将文本模板编码为文本 ID |
| `generation_config.json` | checkpoint 随附的生成参数 | 提供模型生成配置；本文的实际采样参数沿 deploy 配置和请求参数说明 |
| `preprocessor_config.json` | 根目录预处理配置 | 随模型保存的预处理元数据 |
| `speech_tokenizer/config.json` | 音频 encoder/decoder 结构、采样率、上采样率和码本数 | 创建 Speech Tokenizer encoder 与 decoder |
| `speech_tokenizer/model.safetensors` | `encoder.*`、`decoder.*` 张量 | 加载参考音频编码器与波形解码器 |
| `speech_tokenizer/preprocessor_config.json` | 音频特征提取配置 | 创建音频 feature extractor |
| `speech_tokenizer/configuration.json` | 随音频 tokenizer 发布的附加配置 | 保存音频 tokenizer 的附加元数据 |

本次读取 safetensors 头部核对到：主文件包含 404 个张量条目，参数前缀为 `talker`；音频 tokenizer 文件包含 496 个张量条目，前缀为 `encoder` 和 `decoder`。主例使用 CustomVoice；Base 的 speaker encoder 权重由其对应 checkpoint 提供。

来源：[HF 文件目录](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/0c0e3051f131929182e2c023b9537f8b1c68adfe)、[主权重](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/model.safetensors)、[音频 tokenizer 权重](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/speech_tokenizer/model.safetensors)。

### 0.2 config.json 如何确定模型结构？

根配置包含以下关键字段：

```text
model_type = "qwen3_tts"
architectures = ["Qwen3TTSForConditionalGeneration"]
|
`-- talker_config
    |-- hidden_size = 2048
    |-- num_hidden_layers = 28
    |-- vocab_size = 3072
    |-- text_vocab_size = 151936
    |-- text_hidden_size = 2048
    |-- num_code_groups = 16
    `-- code_predictor_config
        |-- hidden_size = 1024
        |-- num_hidden_layers = 5
        `-- vocab_size = 2048
```

vLLM-Omni 根据 `qwen3_tts` 选择 `QWEN3_TTS_PIPELINE`，从 pipeline 与 deploy 配置获得两个 Stage 的模型类和执行配置：[vllm_omni/model_executor/models/qwen3_tts/pipeline.py:32](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/pipeline.py#L32)、[vllm_omni/deploy/qwen3_tts.yaml:24](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L24)

```text
HF model directory + revision
             |
       read config.json
             |
       QWEN3_TTS_PIPELINE
             |
       +-----+--------------------------------+
       |                                      |
       v                                      v
Stage 0                                  Stage 1
Qwen3TTSTalkerForConditionalGeneration    Qwen3TTSCode2Wav
       |                                      |
       +-- Qwen3Model                         +-- Speech Tokenizer decoder
       +-- text_embedding                         created from
       +-- text_projection                        speech_tokenizer/config.json
       +-- codec LM head
       +-- CodePredictor
       +-- reference audio encoder
```

创建模块时，配置决定参数张量的形状。例如 `nn.Embedding(151936,2048)` 创建 `[151936,2048]` 权重，`Linear(2048,3072)` 创建 `[3072,2048]` 权重。随后 loader 将 checkpoint 中的数值写入这些参数。

### 0.3 safetensors 中的参数如何对应到 Stage 0？

safetensors 头部记录每个张量的名称、shape、dtype 和数据偏移；loader 读取张量后，按名称映射交给模型的 `load_weights()`。

`Qwen3TTSTalkerForConditionalGeneration.hf_to_vllm_mapper` 定义前缀映射。以下参数名和 shape 已通过主权重头部核对，表中目标名相对于 Stage 0 模型对象：[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:348](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L348)

| HF 参数名 | checkpoint shape | Stage 0 参数名 | 用途 |
|---|---|---|---|
| `talker.model.codec_embedding.weight` | `[3072,2048]` | `model.embed_tokens.weight` | 主码本及控制 ID 查表 |
| `talker.model.text_embedding.weight` | `[151936,2048]` | `text_embedding.weight` | 文本 ID 查表 |
| `talker.text_projection.linear_fc1.weight` | `[2048,2048]` | `text_projection.linear_fc1.weight` | 文本投影第一层 |
| `talker.text_projection.linear_fc2.weight` | `[2048,2048]` | `text_projection.linear_fc2.weight` | 文本投影第二层 |
| `talker.codec_head.weight` | `[3072,2048]` | `lm_head.weight` | hidden 到主码本 logits |
| `talker.code_predictor.small_to_mtp_projection.weight` | `[1024,2048]` | `code_predictor.small_to_mtp_projection.weight` | Talker 空间到 predictor 空间 |
| `talker.code_predictor.model.codec_embedding.0.weight` | `[2048,2048]` | `code_predictor.model.codec_embedding.0.weight` | 第一个残差码本的 embedding |
| `talker.code_predictor.lm_head.0.weight` | `[2048,1024]` | `code_predictor.lm_head.0.weight` | 第一个残差码本的预测头 |

这些条目的 checkpoint dtype 为 BF16。`Linear` 的权重布局为 `[out_features,in_features]`，因此 `[1024,2048]` 的投影权重执行 `2048 -> 1024` 映射。

Transformer 层按前缀映射：

```text
HF:       talker.model.layers.0.*
Stage 0:  model.layers.0.*

HF:       talker.model.norm.*
Stage 0:  model.norm.*

HF:       talker.code_predictor.*
Stage 0:  code_predictor.*
```

Stage 0 的 `load_weights()` 先加载 `talker.*`，再加载 checkpoint 中提供的 `speaker_encoder.*`（[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1334](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1334)）。随后从 `speech_tokenizer/` 加载 `encoder.*`（[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1347](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1347)），并单独复制 encoder 的 VQ buffers，包括 `embed`、`embed_sum`、`cluster_usage` 和 `initialized`（[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1363](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1363)）。

### 0.4 Q/K/V 与 gate/up 权重如何合并？

CodePredictor 的 HF 权重分别保存 Q、K、V 投影，运行模块使用融合的 `qkv_proj`。`CodePredictorBaseModel.load_weights()` 收集同层张量，沿第 0 维按 Q、K、V 顺序拼接。[vllm_omni/model_executor/models/common/qwen3_code_predictor.py:470](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L470)

以 predictor 第 0 层为例，以下 shape 来自权重头部：

```text
HF self_attn.q_proj.weight   [2048,1024] --+
HF self_attn.k_proj.weight   [1024,1024] --+--> cat(dim=0)
HF self_attn.v_proj.weight   [1024,1024] --+        |
                                                 v
runtime self_attn.qkv_proj.weight            [4096,1024]
```

输入 `[B,S,1024]` 经过一次投影得到 `[B,S,4096]`，forward 再按 `[2048,1024,1024]` 拆出 Q、K、V。

MLP 的 gate/up 按相同方式合并：

```text
HF mlp.gate_proj.weight      [3072,1024] --+
HF mlp.up_proj.weight        [3072,1024] --+--> cat(dim=0)
                                                 |
                                                 v
runtime mlp.gate_up_proj.weight              [6144,1024]
```

forward 得到 6144 维输出后拆成两个 3072 维向量，执行 `SiLU(gate) * up`，再通过 `down_proj` 返回 1024 维。Talker 主干通过 vLLM 的 Qwen3 权重加载路径，将对应分量加载到 `QKVParallelLinear` 和融合 gate/up 参数中；本文按 TP=1 展开完整形状。[vllm_omni/model_executor/models/common/qwen3_code_predictor.py:325](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L325)、[vllm/model_executor/models/qwen3.py:65](https://github.com/vllm-project/vllm/blob/3dc7a68ce45ce1b98c2879139465833611f117cb/vllm/model_executor/models/qwen3.py#L65)

### 0.5 音频 tokenizer 权重如何分配到两个 Stage？

两个 Stage 分别从 `speech_tokenizer/model.safetensors` 加载各自所需的张量：[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1347](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1347)、[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:558](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L558)

```text
speech_tokenizer/model.safetensors
                  |
        +---------+---------+
        |                   |
   encoder.*            decoder.*
        |                   |
        v                   v
Stage 0.encoder       Stage 1.decoder
        |                   |
reference waveform    generated codec IDs
        |                   |
reference codec       output waveform
```

Stage 0 使用 `DefaultModelLoader.Source(..., subfolder="speech_tokenizer")` 创建权重迭代器，加载 `encoder.*`。Stage 1 的 `load_weights()` 消费主权重迭代器后，为该子目录创建单独的迭代器，将 `decoder.*` 加载到 `self.decoder`。

例如，HF 张量 `decoder.pre_transformer.input_proj.weight` 的 shape 为 `[512,1024]`，加载到 Stage 1 同名参数，执行 codec 特征的 `1024 -> 512` 投影。该张量在 checkpoint 中保存为 F32；Stage 1 加载后按 `model_config.dtype` 转换 decoder，本文默认 deploy 为 BF16。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:581](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L581)、[vllm_omni/deploy/qwen3_tts.yaml:117](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L117)

### 0.6 权重加载后会准备哪些运行数据？

| 模块 | 初始化内容 | 后续使用位置 |
|---|---|---|
| Talker | `_tts_pad_embed = text_projection(text_embedding(tts_pad_id))`，shape `[1,2048]` | prefill 条件拼接与 decode 文本条件 |
| Talker | 将 15 张 residual embedding 权重堆叠为 `_stacked_codec_embed`，shape `[15,2048,2048]` | MTP 完成后批量 gather 并求和 |
| Talker | 主码本 logits mask，以及按配置准备的 silence mask | `compute_logits()` |
| Code2Wav | SnakeBeta 的指数与倒数缓存 | 波形 decoder 的激活计算 |
| 执行后端 | 按配置和执行路径准备编译、warmup 与 Graph shape buckets | 模型 forward 与图回放 |

模型完成加载后，请求进入以下流程：

```text
HF files
 -> config creates modules
 -> named tensors populate parameters and buffers
 -> runtime buffers / execution preparation
 -> request preprocessing
 -> Talker + CodePredictor
 -> Code2Wav
 -> audio
```

本文主例为单卡 TP=1。配置和权重头部于 2026-09-24 核对；vLLM-Omni 源码链接固定到 `3d43571b94f1683412023c45bd886f9ce30f76bd`，Qwen3 主干源码链接固定到 vLLM `3dc7a68ce45ce1b98c2879139465833611f117cb`。后文示例 token/codec ID 为教学假设；形状与加载流程依据配置、权重头部及源码推导。

## 1. Talker、CodePredictor 和 Code2Wav 如何分布在两个 Stage 中？

Qwen3-TTS 的主要生成路径是：文本条件驱动 Talker，Talker 每步生成一个主码本 ID；CodePredictor 补全同一帧的其余码本；Speech Tokenizer 的 decoder 把多帧 codec 解码成波形。

```text
Text + language + speaker / instruction / reference audio
                         |
                         v
             PromptEmbedsBuilder
                         |
                         v
+------------------------ Stage 0 ------------------------+
| Talker: Qwen3Model, 28 layers                           |
|   prefill / temporal autoregressive decode              |
|   hidden -> codec_head -> sample c[t,0]                 |
|                         |                               |
| CodePredictor: 5 layers, 15 autoregressive substeps      |
|   (hidden, c[t,0]) -> c[t,1] -> ... -> c[t,15]           |
|                         |                               |
| complete frame C[t] = [c[t,0], ..., c[t,15]]             |
|   sum codec embeddings + next text vector              |
|                         +----> next Talker input         |
+-------------------------|-------------------------------+
                          | codec frames; chunk transport
                          v
+------------------------ Stage 1 ------------------------+
| Code2Wav / Speech Tokenizer decoder                     |
|   RVQ lookup -> Conv -> Transformer -> Upsampling       |
|   -> waveform [N], sample rate = 24000 Hz                |
+--------------------------------------------------------+
                          |
                          v
                 audio response / chunks
```

CodePredictor 在 Stage 0 内被调用。Stage 1 通过 `LLM_GENERATION` 执行类型接入引擎，内部执行 codec-to-waveform 网络，直接输出波形，`compute_logits()` 返回 `None`。[vllm_omni/model_executor/models/qwen3_tts/pipeline.py:1](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/pipeline.py#L1)、[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1495](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1495)、[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:135](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L135)

### 1.1 prompt 位置数、codec 帧数、码本数和波形采样点数分别是什么？

| 记号 | 含义 | 例子 |
|---|---|---|
| `P` | Talker prompt embedding 的位置数 | 包含文本、角色、语言、音色等条件的位置 |
| `T` | 音频 codec 帧数 | 25 帧对应 2 秒音频内容 |
| `G` | 每帧码本数 | 本模型为 16；一帧有 16 个 ID |
| `S` | CodePredictor 一次 forward 的序列长度 | 默认固定 buffer 长度为 `G+1=17` |
| `N` | 波形采样点数 | `N=T×1920`，单声道 |

```text
100 codec frames
   = 100 × 16 = 1600 codec IDs
   = 100 × 1920 = 192000 waveform samples
   = 192000 / 24000 = 8 seconds of audio content
```

生成音频的时长为 8 秒。

## 2. Talker、CodePredictor 和 Codec decoder 的配置分别是什么？

以下参数来自 1.7B CustomVoice checkpoint 的配置。

| 参数 | Talker | CodePredictor | Codec decoder 的 Transformer |
|---|---:|---:|---:|
| 层数 | 28 | 5 | 8 |
| hidden size | 2048 | 1024 | 512 |
| intermediate size | 6144 | 3072 | 1024 |
| Q heads | 16 | 16 | 16 |
| KV heads | 8 | 8 | 16 |
| head dim | 128 | 128 | 64 |
| Q 投影输出宽度 | 2048 | 2048 | 1024 |
| K/V 各自输出宽度 | 1024 | 1024 | 1024 |
| Attention 范围 | 因果全注意力 | 帧内短序列因果注意力 | sliding window=72 |
| 状态保留 | 引擎管理 Talker KV | 当前实现 re-prefill，无持久 KV | 流式路径按请求保存状态 |

其他关键参数：

| 参数 | 值 | 用途 |
|---|---:|---|
| `text_vocab_size` | 151936 | 文本 embedding 词表 |
| `text_hidden_size` | 2048 | 文本查表后的宽度 |
| Talker `vocab_size` | 3072 | 包含实际 codec ID 和控制、语言、speaker 等 ID |
| CodePredictor `vocab_size` | 2048 | 每个残差码本预测头的类别数 |
| `num_code_groups` | 16 | 主码本 1 个，残差码本 15 个 |
| `codec_pad_id / codec_bos_id / codec_eos_token_id` | 2148 / 2149 / 2150 | codec 条件与停止标记 |
| `tts_pad / tts_bos / tts_eos` | 151671 / 151672 / 151673 | 文本侧 TTS 特殊 ID |
| `output_sample_rate` | 24000 | 输出波形采样率 |
| `decode_upsample_rate` | 1920 | 每 codec 帧的波形采样点数 |

注意力投影宽度与音频帧率的计算如下：

1. CodePredictor 的 `hidden_size=1024`，但 `16×128=2048`；它的 Q 投影扩展到 2048 维，O 投影再回到 1024 维。
2. 模型名称写 “12Hz”，按配置计算的帧率是 `24000/1920=12.5 Hz`，即每帧 80 ms。音频长度由 tokenizer 的采样率和上采样率计算。

## 3. 阶段一：请求如何变成 Talker 的输入

### 3.1 HTTP 请求中的文本如何传给 Talker？

以两个 CustomVoice 请求为例：

```text
A: text="你好",   language="Chinese", speaker="Vivian"
B: text="你是谁", language="Chinese", speaker="Vivian"
instruct: empty
non_streaming_mode: True  (CustomVoice default)
```

`Qwen3TTSAdapter` 构造 `tts_params`，把文本、task_type、speaker、language 等放在 `additional_information`。它估算完整 prefill 长度，创建相同长度的占位 token 列表：

```python
# 简化表示，保留源码中的关键契约
prompt = tokens_input(prompt_token_ids=[1] * ph_len)
prompt["additional_information"] = tts_params
```

这些 `1` 用于引擎的长度与调度记账。模型 `preprocess()` 根据实际条件构造 `inputs_embeds`，prefill 的记账 ID 则改为合法的 `codec_pad_id`。Talker 通过 `inputs_embeds` 接收条件向量。 [vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:868](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L868)、[vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py:465](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py#L465)

```text
request text / voice / language
       |
       +--> estimate prompt length --> placeholder input_ids [P]
       |
       +--> additional_information
                    |
                    v
              preprocess()
                    |
           prompt embeddings [P,2048]
                    |
                    v
           Qwen3Model(inputs_embeds=...)
```

### 3.2 文本 ID 和 codec ID 分别如何转换为 embedding？

定义：

```text
E_text(id):  text_embedding(id)             -> [2048]
T(id):       text_projection(E_text(id))    -> [2048]
E_0(id):     Talker codec embedding(id)     -> [2048]
E_g(id):     residual codec embedding g     -> [2048], g=1..15
```

`text_projection` 是两层带 bias 的 MLP：

```text
text ID
  -> Embedding(151936,2048)
  -> Linear(2048,2048)
  -> SiLU
  -> Linear(2048,2048)
  -> text vector [2048]
```

`text_projection` 使用学习到的权重，将 2048 维文本 embedding 映射到 2048 维 Talker 条件空间。codec ID 通过独立的 codec embedding 表查表。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:433](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L433)

### 3.3 CustomVoice prompt 的每个位置包含什么？

Builder 先套用 assistant 文本模板，然后取前三个角色 token、正文 token，并单独构造 TTS 特殊向量。[vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py:1021](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py#L1021)

为便于推演，假设正文 A 编成两个 ID `[101,102]`，B 编成三个 ID `[201,202,203]`。这些教学用 ID 分别表示长度为 2 和 3 的正文序列。

选择 Chinese、Vivian 时，codec 条件 ID 为：

```text
[codec_think, think_bos, Chinese, think_eos, Vivian, codec_pad, codec_bos]
[2154,        2156,      2055,    2157,      3065,   2148,      2149]
```

`non_streaming_mode=True`、无 instruct 时，A 的 prompt 可逐位置展开：

| 位置 | 输入向量 | 来源 |
|---:|---|---|
| 0..2 | `T(role[0..2])` | assistant 角色模板的三个 token |
| 3 | `T(tts_pad) + E_0(2154)` | codec_think |
| 4 | `T(tts_pad) + E_0(2156)` | think_bos |
| 5 | `T(tts_pad) + E_0(2055)` | Chinese |
| 6 | `T(tts_pad) + E_0(2157)` | think_eos |
| 7 | `T(tts_pad) + E_0(3065)` | Vivian |
| 8 | `T(tts_bos) + E_0(2148)` | 文本 BOS 与 codec PAD 对齐 |
| 9 | `T(101) + E_0(2148)` | 第一个正文 token |
| 10 | `T(102) + E_0(2148)` | 第二个正文 token |
| 11 | `T(tts_eos) + E_0(2148)` | 正文结束 |
| 12 | `T(tts_pad) + E_0(2149)` | codec BOS，启动语音生成 |

因此：

```text
P_A = 3 role + 6 codec prefix + (2 text + 1 eos) + 1 codec_bos = 13
P_B = 3 role + 6 codec prefix + (3 text + 1 eos) + 1 codec_bos = 14
```

这组长度对应 Chinese 分支、内置 speaker、无 instruct 和完整文本 prefill 的条件组合。Auto 语言、VoiceDesign、Base ICL 都会改变 prompt 结构。

### 3.4 non_streaming_mode、async_chunk 和音频流式返回分别控制什么？

| 控制项 | 作用 |
|---|---|
| `non_streaming_mode=True` | 正文和 tts_eos 全放进 prefill；后续文本条件主要为 tts_pad |
| `non_streaming_mode=False` | prefill 放首个正文 token；余下文本向量进入 `trailing_text_hidden`，在 decode 时逐步使用 |
| deploy `async_chunk=True` | Stage 0 按 chunk 把 codec 送到 Stage 1 |
| HTTP / WebSocket 输出模式 | 控制音频数据返回客户端的方式 |

Builder 默认 CustomVoice / VoiceDesign 为 True，Base 为 False。False 模式下，decode 每步从文本队列取一个向量；队列耗尽后继续使用 `T(tts_pad)`。[vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py:1010](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py#L1010)

## 4. 阶段二：Talker prefill 与单层 Transformer

### 4.1 两个请求的 prompt 如何展平成一个输入张量？

为讲清形状，先假定 A、B 的 prefill 在同一次 forward 完整执行；实际引擎可拆成 chunked prefill。

```text
A embeddings: [13,2048], positions 0..12
B embeddings: [14,2048], positions 0..13
               |
               v
packed embeddings: [27,2048]
logical positions: [0,1,...,12, 0,1,...,13]
request boundaries: [0,13,27]
```

请求边界由 attention metadata 等调度信息表达，每个请求的 attention 范围限定在自身序列内。逻辑位置在各请求内重新开始；上图展示逻辑位置，物理 position tensor 布局由 RoPE 和 runner 实现决定。

`[27,2048]` 也可能来自单请求 27 个位置，但两者的 attention 边界、KV 映射和待采样末位不同。

### 4.2 一层 Talker 的数据流

Talker 使用 dense `Qwen3Model`，没有 Qwen3MoE 的 router、top-k experts 或专家分发。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:420](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L420)、[vllm/model_executor/models/qwen3.py:65](https://github.com/vllm-project/vllm/blob/3dc7a68ce45ce1b98c2879139465833611f117cb/vllm/model_executor/models/qwen3.py#L65)

```text
x [27,2048]
 |
 +---------------- residual ------------------+
 |                                            |
 RMSNorm                                      |
 |                                            |
 QKV projection [27,4096]                      |
 |                                            |
 +-- Q [27,2048] -> [27,16,128] -> Q norm --+   |
 +-- K [27,1024] -> [27, 8,128] -> K norm --+   |
 +-- V [27,1024] -> [27, 8,128]            |   |
                                          |   |
                      RoPE(Q,K)           |   |
                          |               |   |
                    causal GQA            |   |
                          |               |   |
                  [27,16,128]             |   |
                          |               |   |
                 O projection             |   |
                    [27,2048]             |   |
                          +-------------------+
                          |
                         add
                          |
 +---------------- residual ------------------+
 |                                            |
 RMSNorm                                      |
 |                                            |
 gate/up projection [27,12288]                 |
 |                                            |
 split gate [27,6144], up [27,6144]             |
 |                                            |
 SiLU(gate) * up [27,6144]                     |
 |                                            |
 down projection [27,2048]                    |
 +--------------------------------------------+
                          |
                         add
                          |
                    [27,2048]
```

图中是数学上的残差连接。vLLM 实现会把 residual 与 RMSNorm 融合，并在层间传递 residual，具体 kernel 边界由后端融合实现决定。

#### RMSNorm

对每个位置的向量独立计算：

```text
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

Talker 层内归一化宽度是 2048，Q/K 的 per-head norm 宽度是 128。RMSNorm 按均方根缩放输入，再乘以可学习权重。

#### GQA 与因果边界

```text
16 query heads / 8 KV heads = 2
Q heads 0,1 share KV head 0
Q heads 2,3 share KV head 1
...
```

每个 head 的逻辑注意力为：

```text
scores = Q @ K^T / sqrt(128) + causal / request-boundary mask
output = softmax(scores) @ V
```

prefill 的当前位置读取同请求此前及当前位置的 K/V。实际后端通过 attention kernel 完成分数计算、归一化和加权求和；中间矩阵的存储方式由具体 kernel 实现决定。

#### SwiGLU MLP

```text
MLP(x) = W_down( SiLU(W_gate x) * (W_up x) )
SiLU(z) = z * sigmoid(z)
```

Talker 每层 intermediate size 为 6144，28 层之后经最终 RMSNorm 得到输出 hidden states。[vllm/model_executor/models/qwen3.py:174](https://github.com/vllm-project/vllm/blob/3dc7a68ce45ce1b98c2879139465833611f117cb/vllm/model_executor/models/qwen3.py#L174)

### 4.3 如何从 prefill 的末位 hidden 采样第 0 码本 ID？

取 A、B 各自 prompt 的最后一行：

```text
hidden [27,2048]
  -> select rows 12 and 26
  -> [2,2048]
  -> codec LM head (2048 -> 3072)
  -> logits [2,3072]
  -> mask + sampling
  -> first main-codebook IDs [2]
```

本版本 `compute_logits()` 允许主码本 ID `1..2047` 及 `codec_eos_token_id=2150`，屏蔽其他 ID；**主码本 ID=0 也被该 mask 禁止采样**。残差码本预测头输出 2048 类 logits，使用各自的采样路径。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:476](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L476)

默认部署 Talker 采样参数包括 `temperature=0.9`、`top_k=50`、`repetition_penalty=1.05`、`min_tokens=2`。请求可通过支持的参数覆盖默认采样配置。[vllm_omni/deploy/qwen3_tts.yaml:94](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L94)

例如假设采样得到：

```text
A: c[0,0] = 120
B: c[0,0] = 350
```

此时每个请求得到第一帧的第 0 个码本 ID，接下来由 CodePredictor 补齐其余 15 个 ID。模型同时保存末位 hidden 到 `hidden_states.last`，供后续 CodePredictor 使用。

## 5. 阶段三：CodePredictor 在一帧内补齐 15 个码本

### 5.1 CodePredictor 在哪次 decode 中补齐一帧？

数学上可以说“Talker 生成主码，CodePredictor 补全”，但 runner 的调用边界更细：前一次 Talker 采样得到的 ID，在**下一次 decode preprocess** 中被补齐。

```text
Prefill iteration:
  prompt -> Talker -> h[0] -> sample c[0,0]
                         store hidden_states.last

Decode iteration 1:
  previous ID c[0,0] + stored h[0]
       -> CodePredictor -> C[0] = [c[0,0],...,c[0,15]]
       -> sum all 16 embeddings + text_step
       -> Talker -> h[1] -> sample c[1,0]
       -> publish C[0] for the codec stream

Decode iteration 2:
  previous ID c[1,0] + stored h[1]
       -> CodePredictor -> C[1]
       -> sum embeddings + text_step
       -> Talker -> h[2] -> sample c[2,0]
       -> publish C[1]
```

这里 `h[t]` 定义为**用于预测 `c[t,0]` 的 hidden**。采用这个定义，CodePredictor 的输入恰好是 `h[t]` 与 `c[t,0]`。

V1 runner 的 `_talker_mtp_forward()`、MRV2 model state 的 `_run_batched_mtp()` 都把这一过程批量化，并将结果写回本次 Talker `inputs_embeds` 和请求的 `codes.audio`。模型接口名为 `talker_mtp` / `mtp`，负责生成当前音频帧的残差码本，并准备下一次 Talker 输入。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1495](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1495)、[vllm_omni/worker/gpu_model_runner.py:1935](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/worker/gpu_model_runner.py#L1935)、[vllm_omni/worker_v2/model_states/omni_model_state.py:811](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/worker_v2/model_states/omni_model_state.py#L811)

### 5.2 CodePredictor 的输入与投影

对两个请求，输入是：

```text
layer0_code:       [2,1]       = [[120],[350]]
layer0_embed:      [2,1,2048]  = E_0(layer0_code)
last_talker_hidden:[2,1,2048]  = h[0]
```

1.7B Talker 是 2048 维、CodePredictor 是 1024 维，因此存在带 bias 的 `small_to_mtp_projection: 2048 -> 1024`：

```text
proj_buf [B_pad,17,1024]

position 0: projection(h[t])
position 1: projection(E_0(c[t,0]))
position 2: initially 0; later projection(E_1(c[t,1]))
...
position 16: spare / padded position in the default fixed buffer
```

`B_pad` 是执行桶的 batch 大小，可大于真实请求数。最终输出只取真实的 B 行。[vllm_omni/model_executor/models/common/qwen3_code_predictor.py:1108](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L1108)

### 5.3 15 个残差码本如何逐个预测？

`for step in range(1,16)` 每次调用同一组 5 层 Transformer，但用不同码本的 LM head：

| 子步骤 | 已知条件 | 取哪一行 hidden | 预测头 | 新产生 |
|---:|---|---:|---|---|
| 1 | `h[t], E_0(c[t,0])` | 1 | `lm_heads[0]` | `c[t,1]` |
| 2 | 再加 `E_1(c[t,1])` | 2 | `lm_heads[1]` | `c[t,2]` |
| 3 | 再加 `E_2(c[t,2])` | 3 | `lm_heads[2]` | `c[t,3]` |
| ... | ... | ... | ... | ... |
| 15 | `h[t], E_0(c[t,0]), ..., E_14(c[t,14])` | 15 | `lm_heads[14]` | `c[t,15]` |

每个头将 `[2,1024]` 映射成 `[2,2048]` logits，再采样一个残差 ID。

默认固定长度路径每次都对 `[B_pad,17,1024]` 执行 forward。因果 attention 将当前所取行的可见范围限制在已填充前缀内。采样出 `c[t,15]` 后，这一帧的循环结束。

源码支持可选的 prefix re-prefill 长度桶；启用后可以缩短物理 forward 长度。本版本 NPU 使用固定长度 re-prefill 路径。有效前缀长度为 `step+1`，实际 forward 的 `seq_len` 由执行长度桶决定。 [vllm_omni/model_executor/models/common/qwen3_code_predictor.py:690](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L690)

### 5.4 CodePredictor 每个子步骤如何重新计算 K/V？

`CodePredictorBaseModel` 用 SDPA/NPU attention，对短序列重复执行 prefill。每个子步骤重新计算传入序列的 K/V。

```text
step 1: recompute positions [0..16], read output at 1
step 2: recompute positions [0..16], read output at 2
...
step15: recompute positions [0..16], read output at15
```

单层 predictor 的结构仍是 RMSNorm、QK norm、RoPE、GQA、SwiGLU，但维度不同：

```text
[B_pad,17,1024]
  -> QKV linear [B_pad,17,4096]
  -> Q [B_pad,16,17,128]
     K [B_pad, 8,17,128]
     V [B_pad, 8,17,128]
  -> causal attention
  -> O: 2048 -> 1024
  -> residual + RMSNorm
  -> gate/up: 1024 -> 6144
  -> split into two 3072-wide tensors
  -> SiLU(gate) * up
  -> down: 3072 -> 1024
```

CUDA 等非 NPU 路径使用 `scaled_dot_product_attention(..., is_causal=True, enable_gqa=...)`；NPU 走 `_forward_npu_attention()` 等平台算子路径。[vllm_omni/model_executor/models/common/qwen3_code_predictor.py:278](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L278)

### 5.5 CodePredictor 如何采样，输出形状是什么？

Qwen3-TTS wrapper 使用 `sampling_mode="per_call"`。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py:38](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py#L38) 默认 residual sampling 是 temperature 0.9、top-k 50、top-p 1.0；有采样时使用 Gumbel-max，关闭采样或温度不大于 0 时取 argmax。本路径启用采样时要求 `top_p=1.0`，其他值会触发参数错误。[vllm_omni/model_executor/models/common/qwen3_code_predictor.py:1130](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L1130)

示意输出：

```text
A: C[0] = [120, 51, 902, ..., 7]      # 16 IDs
B: C[0] = [350, 89, 613, ..., 201]    # 16 IDs

audio_codes shape = [2,16]
```

15 个码本子步骤串行依赖；同一子步骤的两个请求可以批量计算。

## 6. 阶段四：完整 codec 帧如何成为下一次 Talker 输入

`talker_mtp()` 将一帧全部码本查表、相加，再叠加当前文本条件：

```text
codec_sum[t] = E_0(c[t,0]) + E_1(c[t,1]) + ... + E_15(c[t,15])
x_next       = codec_sum[t] + text_step

[2,16] IDs
  -> 16 embeddings, each [2,2048]
  -> sum across codebooks [2,2048]
  -> add text_step [2,2048]
  -> next Talker inputs_embeds [2,2048]
```

按码本求和后的向量宽度为 2048。源码把 15 张 residual embedding 表堆叠后做一次 gather，减少逐表查找的调用；数学含义仍是按码本查表后求和。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1550](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1550)

文本条件的选择：

```text
if trailing_text_hidden still has rows:
    text_step = next row
else:
    text_step = T(tts_pad)
```

主例完整文本已在 prefill 中提供，decode 使用 pad 文本条件。若 `non_streaming_mode=False`，文本队列则会逐步提供余下正文与 tts_eos。

### 6.1 Talker decode 与 prefill 的 shape 差异

第一次 decode，每个请求只新增一个 Talker 位置：

| 项目 | 完整 prefill 示例 | 第一次 decode 示例 |
|---|---|---|
| 当前计算的位置数 | `13+14=27` | `1+1=2` |
| 输入 hidden | `[27,2048]` | `[2,2048]` |
| 每请求逻辑新位置 | A 0..12；B 0..13 | A 13；B 14 |
| attention 可用 KV 总长度 | 分别增至 13、14 | 分别增至 14、15 |
| 待采样 hidden | `[2,2048]` | `[2,2048]` |
| 主码本 logits | `[2,3072]` | `[2,3072]` |

本次 decode 计算两个新位置的 Query，A、B 分别读取长度为 14、15 的 K/V。引擎依请求独立管理 KV block 与位置映射。

`max_num_seqs=64` 表示调度配置的请求数上限。请求抵达、结束、prefill/decode 混排都会改变实际 batch。

### 6.2 采样到 EOS 后，最后一帧和剩余 chunk 如何处理？

Pipeline 设置 `stop_token_ids=[2150]`，主 Talker 采样 codec EOS 后结束该请求的主码流。EOS 用于标记主码流结束。[vllm_omni/model_executor/models/qwen3_tts/pipeline.py:47](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/pipeline.py#L47)

本实现还有防御逻辑：`talker_mtp()` 对非法主码本行置零，输出/转接路径通过 `codec_frame_valid` 等标志过滤无效行，包括 prefill 的占位零行。结束时还要 flush 已产生但不足一个整 chunk 的帧。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:1546](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L1546)、[vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:81](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py#L81)

主码采样之后，下一轮 decode 完成 residual completion，形成完整音频帧。生成可因采样到 codec EOS 或达到 `max_tokens` 而结束；adapter 会检查结束原因与生成结果。

## 7. 阶段五：Stage 0 的帧怎样交给 Stage 1

### 7.1 codec 如何从 [T,16] 展平，再恢复为 [16,T]？

Stage 0 每帧保存一行，因此累积逻辑形状为 `[T,16]`。Stage 1 decoder 需要 `[B,16,T]`。转接函数采用 **codebook-major** flatten。[vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:395](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py#L395)

先用两个码本、三帧的玩具数组说明顺序（真实模型是 16 个码本）：

```text
frames [T,G]:
  t0: [10,20]
  t1: [11,21]
  t2: [12,22]

transpose -> [G,T]:
  g0: [10,11,12]
  g1: [20,21,22]

flat: [10,11,12,20,21,22]
       ------g0------ ------g1------

Stage 1: flat.reshape(G,T) -> [G,T]
```

真实模型 25 帧传 `16×25=400` 个整数 ID，随后恢复成 `[16,25]`。

### 7.2 默认异步模式每个 chunk 发送多少个新帧？

本文版本默认 deploy 配置为：

```yaml
async_chunk: true
codec_chunk_frames: 25
codec_left_context_frames: 72
initial_codec_chunk_frames: 1
decode_batch_max_size: 4
```

使用默认固定 chunk 配置时，假设最终生成 60 个完整帧：

```text
complete frames accumulated: 1      26       51       60 + finished
                             |       |        |        |
new frames sent:             1      25       25        9
audio content duration:    80ms   2000ms   2000ms    720ms
```

源码还支持 fixed ramp、adaptive chunk 和请求级 initial chunk 覆盖，因此具体边界以生效配置为准。

**当前异步转接实现后续只发送新完成的帧。** Code2Wav 自己保留 quantizer/conv/Transformer 上下文。

对 Base ICL，第一块可额外包含参考 codec 前缀；默认取参考码末尾最多 72 帧用于 decoder 初始化。后续块复用同请求的 decoder 状态。Code2Wav 的参考前缀长度由转接配置控制，Talker 的 ICL prompt 由 prompt builder 构造。[vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:356](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py#L356)、[vllm_omni/deploy/qwen3_tts.yaml:34](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L34)

### 7.3 async_chunk=False 时如何传递完整 codec 序列？

`async_chunk=False` 时，pipeline 使用 `talker2code2wav_full_payload()` 发送完整 codec payload，并用 `talker2code2wav_token_only()` 产生调度所需的长度占位。[vllm_omni/model_executor/models/qwen3_tts/pipeline.py:61](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/pipeline.py#L61)、[vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:576](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py#L576)

另外，decoder 的非流式长序列也可以在内部按 `chunk_size=300`、`left_context_size=25` 等参数分块计算。`chunk_size` 与 `left_context_size` 控制 decoder 内部计算；`codec_chunk_frames` 控制 Stage 间发送频率。

## 8. 阶段六：Code2Wav 把离散码转换成波形

### 8.1 Stage 1 如何组成 batch

`Qwen3TTSCode2Wav.forward()` 按请求拆分输入，验证长度可被 16 整除，恢复 `[16,T_i]`，再按最大长度组装：

```text
A: [16,25]
B: [16, 9]
      |
      v
request_codes [2,16,25]
request_lengths = [25,9]
```

B 的有效长度为 9 帧，补齐后的物理长度为 25 帧。有效长度独立传给 `batched_chunked_decode()`；它按状态与形状分组、受 batch 限制执行，并按实际长度还原输出。Stage 0 的 Talker batch 与 Stage 1 的 decoder batch 由不同调度和分组逻辑形成。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:429](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L429)、[vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:1615](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L1615)

### 8.2 RVQ 如何将 codec ID 解码为连续向量？

`SplitResidualVectorQuantizer` 将第一码本与剩下 15 个码本分开解码：

```text
codes [B,16,T]
   |
   +--> first codebook [B,1,T]
   |       lookup -> [B,256,T] -> output projection -> [B,512,T]
   |
   +--> remaining codebooks [B,15,T]
           15 lookups and sum -> [B,256,T]
           -> output projection -> [B,512,T]
   |
   v
sum first + remaining -> quantized [B,512,T]
```

这里实际构造 `dimension=codebook_dim//2=256`、`output_dimension=512`。底层 `EuclideanCodebook.decode()` 使用归一化后的 codebook embedding 查表。

这里使用 Speech Tokenizer decoder 的 codebook embedding，将 codec ID 映射到波形解码所需的连续特征空间。第 6 节的 Talker embedding 则将 codec ID 映射到 2048 维 Talker 输入空间。[vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:826](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L826)

### 8.3 Code2Wav 各层的张量形状如何变化？

下面描述 `_forward_exact()` 的数学形状，假设输入 `[2,16,25]` 且两行都有 25 个有效帧。流式实现通过保存的状态处理新增帧。

```text
codes                          [2,16,25]
  |
  RVQ decode
  v
quantized                      [2,512,25]
  |
  causal pre_conv: 512 -> 1024
  v
                               [2,1024,25]
  |
  transpose
  v
                               [2,25,1024]
  |
  pre_transformer.input_proj: 1024 -> 512
  v
                               [2,25,512]
  |
  8 Transformer layers + final norm
  |   Q/K/V heads = 16; head_dim = 64
  |   sliding attention window = 72
  v
                               [2,25,512]
  |
  pre_transformer.output_proj: 512 -> 1024
  v
                               [2,25,1024]
  |
  transpose + two ConvTranspose/ConvNeXt upsampling blocks
  |   ×2, ×2
  v
                               [2,1024,100]
  |
  causal conv: 1024 -> 1536
  v
                               [2,1536,100]
  |
  decoder blocks: ×8, ×5, ×4, ×3
  |   channels: 1536 -> 768 -> 384 -> 192 -> 96
  v
                               [2,96,48000]
  |
  SnakeBeta + final causal conv 96 -> 1 + clamp[-1,1]
  v
waveform                       [2,1,48000]
```

上采样倍率：

```text
2 × 2 × 8 × 5 × 4 × 3 = 1920
25 × 1920 = 48000 samples
48000 / 24000 = 2 seconds
```

codec decoder 的 Transformer 带自己的 layer scale、sliding attention 与残差结构。[vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:446](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L446)

### 8.4 Code2Wav 为每个请求保存哪些流式状态？

Stage 1 维护 `_decoder_state_cache[request_id]`。根据是否带 ICL reference，首块走相应初始化路径；后续走 `decode_suffix()` 等增量路径。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:372](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L372)、[vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:1282](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L1282)

```text
request_id A
  |
  +-- prefix_frames / decoder_prefix_frames
  +-- reference / prefix hidden and conv context
  +-- suffix_quantized / suffix_conv
  +-- past_key_values (codec decoder Transformer)
  +-- suffix_frames
```

它同时保存 Transformer KV 和卷积/下游网络需要的边界上下文。不同请求分开保存，正常完成或分段结束后清理对应状态。

ICL 首块用参考前缀建立状态，返回目标新生成部分的 waveform。

Stage 1 最终返回 `OmniOutput` 中的音频张量列表和采样率。decoder 按配置的 dtype 计算，模型输出音频转换为 FP32，后续响应编码按 API 格式处理。[vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:511](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L511)

## 9. Base、CustomVoice、VoiceDesign 的差异发生在哪

| 任务 | 主要条件 | prompt 构造区别 | Code2Wav 参考前缀 |
|---|---|---|---|
| CustomVoice | 内置 speaker ID，可带 instruct | speaker 在 codec embedding 表查表 | 通常无 |
| VoiceDesign | 描述声音的 instruct | instruction embeddings 前置，没有内置 speaker token | 通常无 |
| Base x-vector-only | 参考音频或预计算 speaker embedding | ECAPA-TDNN 提取的 speaker 条件 | 无参考 codec 前缀 |
| Base ICL | 参考音频、参考文本及音色条件 | 参考 codec 和参考/目标文本按 `_generate_icl_prompt()` 组合 | 首块带有界参考 codec 前缀 |

Base 通过两条支路提取参考音频的条件：

```text
reference waveform
       |
       +--> Mel features -> ECAPA-TDNN -> speaker embedding
       |                                   |
       |                                   +--> voice identity conditioning
       |
       +--> Speech Tokenizer encoder -> reference codec [T_ref,16]
                                           |
                                           +--> ICL prompt / decoder prefix
```

x-vector-only 使用 speaker embedding 提供音色条件；预计算 speaker/ref_code、缓存命中也会跳过部分编码。实际是否需 ref_text、ref_audio，应以当前 adapter 的 task_type 校验和 builder 分支为准。各模型变体的层数、维度和输入参数由对应 checkpoint 配置决定。[vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py:117](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py#L117)、[vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py:1121](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py#L1121)

### 9.1 Mel 是什么，怎样从波形得到 Mel 特征？

**Mel 是一种描述声音频率的感知尺度。Mel 频谱将一小段声音在各个 Mel 频带上的强度表示为一组数字。** 将连续时间窗口的这些数字排列起来，就得到一个“时间 × 频带”的矩阵。低频区域的频带划分较细，高频区域较宽，用于表达声音随时间变化的频谱特征。

在这里，Mel 特征作为 ECAPA-TDNN speaker encoder 的输入。speaker encoder 从参考语音的频谱变化中提取音色相关特征，汇总为 speaker embedding，供 Talker 使用。

以 **1 秒、24 kHz、单声道的参考音频**为例，波形有 24000 个采样值：

```text
waveform [1,24000]
   |
   | 每个窗口 1024 个采样点，窗口移动步长 256
   | 两侧各 reflect padding 384 个采样点
   | 每个窗口乘 Hann window，再执行 STFT
   v
complex spectrum [1,513,93]
   |
   | 每个复数频率系数取幅值
   v
magnitude spectrum [1,513,93]
   |
   | Mel filter bank [128,513]
   | 将 513 个频率位置按权重汇总到 128 个 Mel 频带
   v
Mel spectrum [1,128,93]
   |
   | log(clamp(value, min=1e-5))
   | transpose 时间轴与频带轴
   v
log-Mel features [1,93,128]
   |
   v
ECAPA-TDNN -> speaker embedding
```

三个维度分别表示 batch、时间窗口数、Mel 频带数。`[1,93,128]` 表示一条参考音频被划分为 93 个时间窗口，每个窗口用 128 个频带强度特征描述。

| 参数 | 当前实现的值 | 含义 |
|---|---:|---|
| `sampling_rate` | 24000 Hz | 每秒 24000 个波形采样值 |
| `win_size` / `n_fft` | 1024 | 每个窗口覆盖约 `1024/24000=42.67 ms` |
| `hop_size` | 256 | 相邻窗口的起点相隔约 `256/24000=10.67 ms` |
| 单边频谱宽度 | `1024/2+1=513` | STFT 保留从 0 Hz 到 12000 Hz 的频率位置 |
| `num_mels` | 128 | 每个时间窗口输出 128 个 Mel 频带特征 |
| `fmin` / `fmax` | 0 / 12000 Hz | Mel 滤波器覆盖的频率范围 |

窗口数由 padding、窗口长度和步长共同决定。上述例子中：

```text
padding = (1024 - 256) / 2 = 384
padded waveform length = 24000 + 384 + 384 = 24768
number of windows = floor((24768 - 1024) / 256) + 1 = 93
```

STFT 对每个窗口执行傅里叶变换，得到各频率位置的复数系数。源码计算 `sqrt(real² + imag² + 1e-9)` 得到幅值，再乘 Mel 滤波器矩阵。对第 `m` 个频带、第 `t` 个窗口：

```text
mel[m,t] = sum(filter[m,k] * magnitude[k,t], k=0..512)
log_mel[m,t] = log(max(mel[m,t], 1e-5))
```

每个 Mel 滤波器对相邻频率位置加权求和，当前实现使用 Slaney Mel 刻度和归一化。取对数压缩幅值的动态范围，使强弱声音的特征更适合送入后续网络。

实现位置：[vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py:187](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py#L187)；参数设置与 speaker encoder 调用：[vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py:699](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py#L699)；Mel 滤波器构造：[vllm_omni/utils/audio.py:14](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/utils/audio.py#L14)。

## 10. 推理过程中有哪些缓存，Graph 如何影响执行？

### 10.1 各模块缓存什么，保存多久？

| 对象 | 保存内容 | 生命周期 | 本文主路径 |
|---|---|---|---|
| Talker 引擎 KV | 28 层 attention K/V | 请求的 AR 时间轴 | prefill 写入，decode 追加 |
| CodePredictor 工作 buffer | 当前帧 hidden、码本 embeddings | 每次 MTP 调用重新填充 | 短序列 re-prefill，无持久 predictor KV |
| Code2Wav 请求状态 | codec decoder KV、卷积与 prefix/suffix 上下文 | 一个请求的多个音频 chunk | 首块初始化，后续复用 |
| 参考音频/音色缓存 | speaker embedding、ref codec 等 | 依缓存容量与键管理 | 可复用预处理产物 |
| Omni prefix cache | 支持场景下的请求前缀相关数据 | 引擎缓存管理 | 默认 Stage 0 关闭；Stage 1 保持关闭 |

Code2Wav 流式缓存服务于同一请求的后续 chunk；Omni prefix cache 复用匹配请求前缀的数据；CodePredictor 静态 buffer 保存当前帧的计算输入。

### 10.2 使用 Graph 后，15 个残差码本仍需依次预测吗？

```text
Talker forward graph       : main AR backbone
MTP execution graph        : residual-code generation / embedding preparation
Code2Wav inner graph       : eligible decoder shape/state path
```

即使用图回放，CodePredictor 的 `c[t,1] -> ... -> c[t,15]` 依赖仍存在。batch padding、shape bucket 和图捕获控制计算形状与执行方式，15 个残差码本仍依次生成。

本版本默认 deploy 在 CUDA 选择实验性 MRV2，NPU/ROCm/XPU/MUSA 配置选择 V1。NPU 通过模型本地适配设置 dtype、权重布局和算子路径。[vllm_omni/model_executor/models/common/qwen3_code_predictor.py:690](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L690)、[vllm_omni/deploy/qwen3_tts.yaml:145](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L145)、[vllm_omni/platforms/npu/models/qwen3_tts.py:36](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/platforms/npu/models/qwen3_tts.py#L36)

## 11. 一次请求的完整时间线

```text
Client          Adapter / Engine        Stage 0                         Stage 1
  |                     |                  |                               |
  | text + voice ------>|                  |                               |
  |                     | placeholder +   |                               |
  |                     | conditions ---->|                               |
  |                     |                  | build prompt embeddings       |
  |                     |                  | Talker prefill -> h0, c00      |
  |                     |                  |                               |
  |                     |                  | MTP(h0,c00) -> frame C0         |
  |                     |                  | embed(C0)+text -> Talker        |
  |                     |                  |                 -> h1,c10     |
  |                     |                  | -- chunk 0: 1 frame --------->|
  |                     |                  |                               | init state
  |<---------------------------- first waveform chunk --------------------|
  |                     |                  |                               |
  |                     |                  | MTP + Talker continue          |
  |                     |                  | accumulate 25 NEW frames       |
  |                     |                  | -- chunk 1 ------------------->|
  |                     |                  |                               | reuse state
  |<------------------------------ next waveform chunk --------------------|
  |                     |                  |                               |
  |                     |                  | sample EOS; flush tail         |
  |                     |                  | -- final partial chunk ------>|
  |<------------------------------ final waveform chunk -------------------|
  |                     |                  |                               | clear state
```

图展示各阶段的逻辑依赖。Stage 间重叠与设备排队由部署设备和运行时调度决定。首块包含 80 ms 音频内容；TTFA 由排队、预处理、Talker prefill、首次 MTP、codec 解码与传输耗时共同组成。

## 12. 阅读源码时的自检题

1. 两请求的 prefill shape 是 `[27,2048]`，attention metadata 如何确定每个请求的可见范围？
2. `c[0,0]` 采样后，如何补齐第一帧的 16 个码本 ID？MTP 在哪个迭代执行？
3. CodePredictor 有效前缀只有 2 个位置时，默认为什么仍可能对 17 个位置执行 forward？哪个配置能改变它？
4. `codes.audio` 从 `[T,16]` 送到 Stage 1 时，为什么需要按 codebook-major 排列？
5. `non_streaming_mode=True`、`async_chunk=True` 为什么可以同时成立？
6. 处理第二个音频 chunk 时，decoder 从哪些状态中读取上下文？
7. 25 帧对应多少秒音频内容？服务器实际输出间隔由哪些计算与调度耗时决定？

建议实际沿下面这一条最短路径阅读，并手工复述第一帧的状态变化：

```text
build_prompt_embeds
 -> Talker.preprocess
 -> Talker.forward / compute_logits / postprocess
 -> runner MTP hook
 -> Talker.talker_mtp
 -> CodePredictorWrapper.forward
 -> talker2code2wav_async_chunk
 -> Qwen3TTSCode2Wav.forward
 -> decoder.batched_chunked_decode / forward
```

## 13. 来源、定位与证据边界

### 13.1 配置与参考文章

- [1.7B CustomVoice config.json](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/config.json)：Talker、CodePredictor、文本和控制 ID 配置。
- [Speech Tokenizer config.json](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/speech_tokenizer/config.json)：采样率、码本、codec decoder 和上采样配置。
- [参考文档：Qwen3MoE 推理流程详解](https://github.com/wangxiyuan/vllm-study/blob/main/docs/qwen3_moe_inference_flow.md)：提供讲解组织方式的参考。

### 13.2 源码索引

| 固定版本源码 | 阅读入口 / 用途 |
|---|---|
| [vllm_omni/model_executor/models/qwen3_tts/pipeline.py:1](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/pipeline.py#L1) | 两 Stage 注册、停止 ID、转接函数 |
| [vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:345](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py#L345) | preprocess / forward / compute_logits / postprocess / talker_mtp |
| [vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py:45](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py#L45) | HTTP 参数、占位 prompt、结果校验 |
| [vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py:289](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py#L289) | build_prompt_embeds / _generate_icl_prompt / estimate_prompt_len_from_additional_information |
| [vllm_omni/model_executor/models/common/qwen3_code_predictor.py:615](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L615) | 共享 predictor、采样、re-prefill 与长度桶 |
| [vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py:55](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py#L55) | 输入 reshape、batch、请求缓存与音频输出 |
| [vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:106](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py#L106) | 帧收集、chunk 边界、reference prefix、flatten |
| [vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:871](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L871) | RVQ、卷积、Transformer、首块/后续块/批量解码 |
| [vllm_omni/deploy/qwen3_tts.yaml:1](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L1) | 默认部署与平台覆盖 |
| [vllm/model_executor/models/qwen3.py:65](https://github.com/vllm-project/vllm/blob/3dc7a68ce45ce1b98c2879139465833611f117cb/vllm/model_executor/models/qwen3.py#L65) | 相邻 vLLM 源码：dense Qwen3 Attention 与 DecoderLayer |
| [vllm_omni/worker/gpu_model_runner.py:1935](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/worker/gpu_model_runner.py#L1935) | V1 decode preprocess 的 MTP 调用与写回 |
| [vllm_omni/worker_v2/model_states/omni_model_state.py:811](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/worker_v2/model_states/omni_model_state.py#L811) | MRV2 批量 MTP 与输入状态 |
| [vllm_omni/platforms/npu/models/qwen3_tts.py:1](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/platforms/npu/models/qwen3_tts.py#L1) | 模型本地 NPU runtime / dtype / 权重布局适配 |
| [vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py:24](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py#L24) | TTS 专用 wrapper：projection 与 per_call 采样 |

### 13.3 关键结论的来源审核

| Claim | 原始来源 / 类型 | 对象与时效 | 核验及限制 | Verdict |
|---|---|---|---|---|
| Talker 28 层、2048 维；predictor 5 层、1024 维 | 官方 checkpoint config，一手配置 | 指定 1.7B CustomVoice revision，2026-09-24 核对 | 与加载路径匹配，适用于指定 checkpoint | use |
| 每帧 16 码本、1920 采样点，24 kHz | 官方 tokenizer config + decoder 构造，一手 | 同上 | 乘法核验 1920；每帧音频内容时长为 80 ms | use |
| Predictor re-prefill、无持久 KV | [vllm_omni/model_executor/models/common/qwen3_code_predictor.py:615](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L615) 一手实现 | 固定 Omni commit | 通过 attention 接口和 re-prefill 循环核验 | use |
| async chunk 后续只发送新增帧 | [vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:106](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py#L106)、[vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:871](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L871) 一手实现 | 固定 Omni commit | 核对 sender 的增量发送与 decoder 的状态复用 | use |
| batch=2 的 prompt 长度 13/14 与示例 ID | 本文教学推导 | 特定条件组合 | ID 是假设，长度依据 builder 构造；未经真实 tokenizer 运行 | use-with-caveat |
| 默认 MRV2、Graph 与 NPU 分支 | [vllm_omni/model_executor/models/common/qwen3_code_predictor.py:615](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/model_executor/models/common/qwen3_code_predictor.py#L615)、[vllm_omni/deploy/qwen3_tts.yaml:1](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/deploy/qwen3_tts.yaml#L1)、[vllm_omni/platforms/npu/models/qwen3_tts.py:1](https://github.com/vllm-project/vllm-omni/blob/3d43571b94f1683412023c45bd886f9ce30f76bd/vllm_omni/platforms/npu/models/qwen3_tts.py#L1) 一手配置/代码 | 固定 Omni commit | 仅说明选择逻辑；未作硬件捕获、音质或性能验收 | use-with-caveat |

Provenance：一手配置与固定版本源码；2026-09-24 核对；适用于本文指定模型与实现；结构/形状结论为静态证据，运行兼容性、音质和性能未验证。
