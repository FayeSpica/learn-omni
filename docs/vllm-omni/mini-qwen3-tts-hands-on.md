# 手搓 mini-Qwen3-TTS：让"天空是蓝色的"从自建管线发声

> 实践篇。理论对照：[Qwen3-TTS 端到端](qwen3-tts-end-to-end.md)、[三种音色模式](qwen3-tts-voice-modes.md)。
> 代码在 `self-llm/llm/pytorch/`：`mini_qwen.py`（backbone）、`mini_tts_talker.py`（talker壳+装载）、
> `mini_code_predictor.py`（CP）、`mini_tts_e2e.py`（合龙）、`qwen3_tts_codec/`（收编的decoder）。

## 做成了什么

用自己手写的代码重建 Qwen3-TTS-12Hz-1.7B-CustomVoice 的完整推理链，装载真实权重，最终从
自建管线合成出可辨认的语音：

```
"天空是蓝色的"
  → 双通道 prompt 拼装(11+n 格工作单)
  → 手写 talker(28层 Qwen3 同构, 315张量 strict 装载)   → 每帧码本0
  → 手写 code predictor(5层, 15头, 88张量)              → 每帧码本1~15
  → 收编的 codec decoder(speech_tokenizer 子目录权重)    → wav
```

## 核心认知一：talker = 正常 LLM + 特化协议

> **talker = Qwen3 LLM（一字未改）+ 特化的 prompt 协议 + 换了词表和输出头 + CP 细节外挂。**

- backbone 就是 Qwen3：手写的 mini_qwen 容器改 4 个 config 数字直接装下 talker 权重
- 这解释了 vllm-omni 的架构决策：`self.model = Qwen3Model(vllm_config)` 直接复用 vLLM 实现；
  适配工作量集中在 prompt builder 和输入输出两端（1.5k 行 builder vs 零改动 backbone）
- 注意精确化：这个"prompt"特化到**无法用文字写出来**——音色向量、双通道相加只存在于
  embedding 空间，必须走 `inputs_embeds` 入口。多模态的 prompt 工程 = embedding 空间的序列拼装工程

## 核心认知二：工作单——11+n 格双通道表

prompt 的本质是一张声明式的表，每格 = **text 通道 embedding + codec 通道 embedding 相加**
（高维叠放原理：2048 维空间里两通道配对训练、各占子空间，相加不污染，读取靠下游投影按需拆）：

| 位置 | text 通道 | codec 通道 | 含义 |
|------|----------|-----------|------|
| 0-2 | `<|im_start|>` `assistant` `\n` | —（纯 text） | role 头 |
| 3-6 | tts_pad ×4 | think(2154), think_bos(2156), **语言id**(中文2055), think_eos(2157) | 控制区 |
| 7 | tts_pad | **音色id**（vivian=3065 / uncle_fu=3010） | 音色格 |
| 8 | tts_bos | codec_pad(2148) | "语音要开始" |
| 9..8+n | 正文 token ×n | codec_pad ×n | 台词 |
| 9+n | tts_eos | codec_pad | "字念完了" |
| 10+n | tts_pad | **codec_bos(2149)** | 发令枪 |

- 总长 = **11 + n**（n=正文 token 数，按 token 数不按字数）；固定 11 格是协议开销（prefix cache 的完美候选）
- **音色即 token 的外科手术证明**：换音色后对两张 prompt 逐位置 diff，15 格里只有位置 7 非零——
  嗓音 = codec 词表里可插拔的一行（3072 词表中 2048+ 的高位控制区：pad/bos/eos/think/语言/音色全住这里）
- 正文位置 codec 通道必须垫 codec_pad 而非零向量：**神经网络没有天然的"空白"，
  只有训练时用来表示空白的那个学习过的符号**——喂零向量 = 说一句训练分布外的话

## 核心认知三：decode 整帧公式（三次 debug 换来的）

talker 每步的下一个输入不是"码本0的embedding"，而是**整帧表示**（qwen3_tts_talker.py:1105）：

```
下一步输入 = embed(码本0) + Σ CP.codec_embedding[i](残差码i) + text_step(non-streaming恒为tts_pad)
```

RVQ 的本性：一帧声音 = 各量化层向量的叠加，talker 要"听见完整的自己"才能接着说。三次 debug 的病理表：

| 版本 | decode 输入 | 症状 | 教训 |
|------|------------|------|------|
| v1 贪心 | embed(码本0)+pad | 塌缩复读单音 | TTS 必须采样（官方 temp0.9/topk50/rep_pen1.05） |
| v2 | embed(码本0)+pad | 起初正常→漂移塌缩 | 从 streaming 分支错误外推；但"去掉pad"也只是碰巧缓解 |
| v3 | embed(码本0) 裸 | 有人声但非人话 | 缺残差和：talker 听不见完整的自己 |
| 最终 | 整帧三件套 | **出人话** | 公式必须从生产代码逐行抄，不能推断 |

排查方法论：症状分型（复利爆炸=结构 bug；起初正常后退化=累积偏移）→ 顺 runner 数据流追
（`_talker_mtp_forward` 的返回值覆写 inputs_embeds）→ 源头公式。

## 核心认知四：CP 的 16 步小接龙

每帧一局，序列最长 16：

```
位置0 = small_to_mtp_projection(talker hidden 2048→1024)    ← 语义
位置1 = projection(码本0在talker侧的embedding)               ← 轮廓
step 1..15: 过5层(Qwen3DecoderLayer同构) → 取位置step → lm_head[step-1]采样
           → codec_embedding[step-1]查表(2048维,为共享projection而设) → projection → append
```

CP 的 15 本 embedding 是 2048 维（≠自身 hidden 1024），设计目的：与 talker hidden、talker codec
embedding **共用同一个投影**，也让残差 embedding 能直接参与 talker 侧的整帧求和。

## 附：权重的一生（装载全链，通用知识）

```
self.weight = nn.Parameter(...)   属性起名(weight只是惯例,无魔法;ModuleList的".0"=名字就叫"0"的子模块)
→ 属性路径拼 key("model.layers.0.self_attn.q_proj.weight")
→ state_dict() = 花名册快照 {路径: 张量}
→ safetensors 落盘 = JSON目录(dtype/shape/字节住址) + 裸字节
   (无pickle不能藏码;get_slice白嫖shape=零下载侦察;get_tensor按址取=选择性装载)
→ load_state_dict 按名抄数值(骨架先建随机数、后被覆盖;strict=点名员:missing/unexpected)
→ 例外:weight tying 共享内存,save_file 拒收(Qwen3 checkpoint"tie警告"的前因后果)
```

实用推论：
- strict 报告读法：Missing+同名变形 in Unexpected=改名规则错（配对看差异）；Missing 无对应=规则没触发
- mapper 三大坑（亲踩）：removeprefix 漏点、探针字符串与条件不一致、"embedding 在 model 里而
  codec_head 在外"的对称幻觉
- 收编(vendor)三纪律：出处可溯(commit)、改动最小且登记、禁止就地改逻辑

## Shape 流水账（"天空是蓝色的", n=4, prompt=15格）

```
prefill:  [1,15,2048] →28层(层内 q[1,16,15,128] k,v[1,8,15,128]→GQA→分数[1,16,15,15])→ [1,15,2048]
出厂:     h[:,-1] [1,2048] → codec_head → [1,3072] → 采样码本0
decode:   第k步 seq=[1,15+k,2048],无KV cache时分数[1,16,15+k,15+k]——
          有cache只算新行[1,16,1,15+k]:这就是KV cache省的那一列
CP支线:   [1,2,1024]→...→[1,16,1024], lm_head[i]:[1,1024]→[1,2048]
```
