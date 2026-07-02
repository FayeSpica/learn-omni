"""
从零手写一个 decoder-only Transformer(GPT/LLaMA 风格),用于「深入理解」。

刻意不用 nn.Transformer / nn.MultiheadAttention，所有核心组件手写：
    RMSNorm · RoPE · 因果自注意力(支持 GQA) · SwiGLU FFN · Pre-Norm 残差块

配套讲解见 ../handwritten-transformer.md。直接运行本文件会：
    1) 跑一次前向，打印每一步的张量形状；
    2) 在一小段序列上过拟合，验证 loss 能降到接近 0（说明实现是对的）。

    python3 docs/llm-basics/code/minimal_transformer.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128          # 隐藏维度 H
    n_layers: int = 4
    n_heads: int = 8        # query 头数
    n_kv_heads: int = 2     # KV 头数；== n_heads 即 MHA，< 则 GQA，==1 即 MQA
    max_seq_len: int = 128
    ffn_hidden: int = 256   # SwiGLU 中间维度
    rope_theta: float = 10000.0


# ---------------------------------------------------------------------------
# 1. RMSNorm —— 比 LayerNorm 更简单：不减均值、无 bias，只做「按均方根缩放」
#    y = x / sqrt(mean(x^2) + eps) * weight
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放，初始为 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., H]。在最后一维算均方根。用 fp32 算 norm 保数值稳定，再转回原 dtype。
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


# ---------------------------------------------------------------------------
# 2. RoPE(旋转位置编码)—— 把位置信息「旋转」进 Q/K，而不是加到 embedding 上。
#    对每一对相邻维度 (x0, x1) 施加一个随位置 m、频率 θ 变化的二维旋转。
# ---------------------------------------------------------------------------
def build_rope_cache(head_dim: int, max_seq_len: int, theta: float):
    # 频率：每两维共享一个频率，共 head_dim/2 个频率
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, inv_freq)          # [S, head_dim/2]
    return torch.cos(freqs), torch.sin(freqs)   # 各 [S, head_dim/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, n_head, S, head_dim]。把相邻两维看成复数 (even, odd) 做旋转。
    x_even, x_odd = x[..., 0::2], x[..., 1::2]          # 各 [B, nh, S, hd/2]
    cos = cos[None, None, :, :]                         # [1,1,S,hd/2] 广播
    sin = sin[None, None, :, :]
    # 复数旋转：(x_even + i·x_odd) · (cos + i·sin)
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)  # 交错还原
    return out.type_as(x)


# ---------------------------------------------------------------------------
# 3. 因果自注意力（手写 scaled-dot-product + causal mask，支持 GQA）
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0  # 每组 query 头共享一组 KV
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # 每个 KV 头被复用几次

        # 注意：K/V 的投影维度按 KV 头数缩小 —— 这正是 GQA 省 KV Cache 的地方
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        # 投影并拆成多头：[B, S, nh, hd] -> [B, nh, S, hd]
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 施加在 Q/K（只影响相对位置，不动 V）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA：把 KV 头重复 n_rep 次对齐 query 头数
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 缩放点积注意力：softmax(QKᵀ/√d + causal_mask) · V
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,nh,S,S]
        causal = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal            # 未来位置置 -inf，softmax 后为 0
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                      # [B,nh,S,hd]

        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # 合并头
        return self.wo(out)


# ---------------------------------------------------------------------------
# 4. SwiGLU FFN —— gate 分支用 SiLU 激活，逐元素门控 up 分支
#    FFN(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# 5. Pre-Norm 残差块： x = x + Attn(Norm(x)); x = x + FFN(Norm(x))
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)  # 残差 1
        x = x + self.ffn(self.ffn_norm(x))              # 残差 2
        return x


# ---------------------------------------------------------------------------
# 6. 整个 LM：embedding -> N×Block -> final norm -> 词表投影
# ---------------------------------------------------------------------------
class MiniLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重绑定(weight tying)

        cos, sin = build_rope_cache(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # GPT 风格小初始化：否则默认 Embedding ~ N(0,1)，配合权重绑定会让初始 logits
        # 爆炸，step-0 loss 远大于 ln(vocab)。小 std 让初始预测接近均匀分布。
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, S = idx.shape
        x = self.tok_emb(idx)                       # [B, S, H]
        cos, sin = self.cos[:S], self.sin[:S]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)                    # [B, S, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# 自检：形状 + 过拟合
# ---------------------------------------------------------------------------
def _sanity_check():
    torch.manual_seed(0)
    cfg = Config()
    model = MiniLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M  (GQA: {cfg.n_heads} q-heads / {cfg.n_kv_heads} kv-heads)")

    # (1) 形状检查
    B, S = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, S))
    logits, _ = model(idx)
    assert logits.shape == (B, S, cfg.vocab_size), logits.shape
    print(f"前向形状 OK: idx{tuple(idx.shape)} -> logits{tuple(logits.shape)}")

    # (2) 过拟合一小段随机序列：loss 应从 ~ln(vocab) 降到接近 0
    seq = torch.randint(0, cfg.vocab_size, (1, 33))
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"初始 loss ≈ ln(vocab) = {math.log(cfg.vocab_size):.3f}")
    for step in range(200):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 40 == 0 or step == 199:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    assert loss.item() < 0.05, f"未能过拟合，loss={loss.item()}"
    print("✓ 过拟合成功(loss→0)，实现正确。")


if __name__ == "__main__":
    _sanity_check()
