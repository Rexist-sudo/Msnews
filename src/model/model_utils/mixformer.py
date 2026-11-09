"""MixFormer 的 Query Mixer、历史 Cross-Attention 与 Output Fusion。

对应 Huang et al. (2026) §3.3，公式 (3)–(9)；各 Block、各 Head 参数独立。
内部 mask 为 True 表示有效历史，全空历史的注意力增量为零。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rankmixer import token_mixing


class RMSNorm(nn.Module):
    """沿最后一维归一化；缩放参数跨 Head/位置共享，保留输入形状。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0 or eps <= 0:
            raise ValueError(f"RMSNorm 要求 dim、eps > 0，实际 dim={dim}, eps={eps}")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values * scale * self.weight


class PerHeadLinear(nn.Module):
    """对 [B, ..., N, D_in] 执行 N 组独立映射，输出 [B, ..., N, D_out]。"""

    def __init__(self, heads: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if min(heads, input_dim, output_dim) <= 0:
            raise ValueError(f"要求 heads/input_dim/output_dim > 0，实际 {heads}/{input_dim}/{output_dim}")
        self.weight = nn.Parameter(torch.empty(heads, input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(heads, 1, output_dim))
        nn.init.uniform_(self.weight, -1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim))
        nn.init.uniform_(self.bias, -1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        # 合并批次和历史轴；[N, B*T, D] 的 bmm 等价于逐 Head 的 Linear。
        values = values.reshape(-1, shape[-2], shape[-1]).transpose(0, 1)
        output = (torch.bmm(values, self.weight) + self.bias).transpose(0, 1)
        return output.reshape(*shape[:-1], self.weight.shape[-1])


class SwiGLU(nn.Module):
    """silu(W_gate x) * (W_value x) 经输出映射恢复维度；hidden=dim*expansion。

    heads=None 时为完整历史宽度的共享位置 FFN，否则为各 Head 独立 FFN。
    """

    def __init__(self, dim: int, expansion: int, heads: int | None = None) -> None:
        super().__init__()
        if dim <= 0 or expansion <= 0:
            raise ValueError(f"SwiGLU 要求 dim、expansion > 0，实际 {dim}/{expansion}")
        hidden = dim * expansion
        if heads is None:
            self.gate_value = nn.Linear(dim, 2 * hidden)
            self.output = nn.Linear(hidden, dim)
        else:
            self.gate_value = PerHeadLinear(heads, dim, 2 * hidden)
            self.output = PerHeadLinear(heads, hidden, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_value(values).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class QueryMixer(nn.Module):
    """公式 (3)–(4)：Pre-RMSNorm → HeadMixing → Per-head SwiGLU。

    输入输出均为 [B,N,D]，D 为 N 的倍数；Dropout 作用于残差增量。
    """

    def __init__(self, heads: int, dim: int, expansion: int, dropout: float = 0.0) -> None:
        super().__init__()
        if heads <= 0 or dim <= 0 or dim % heads:
            raise ValueError(f"HeadMixing 要求 D 为 N 的正整数倍，实际 N={heads}, D={dim}")
        self.mix_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, expansion, heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B,N,D] -> [B,N,N,D/N]，交换 Head 与子空间轴后恢复原形状。
        mixed = tokens + self.dropout(token_mixing(self.mix_norm(tokens)))
        return mixed + self.dropout(self.ffn(self.ffn_norm(mixed)))


class HistoryCrossAttention(nn.Module):
    """公式 (5)–(8)：每个 D 维 Query 读取历史的对应 Head，返回 [B,N,D] 增量。

    history 为共享投影后的 [B,T,N*D]，mask 为 [B,T]。FFN/K/V 只处理
    有效位置，再恢复序列形状；不递归修改历史。project_history 仅供旧权重适配。
    """

    def __init__(
        self, history_dim: int, heads: int, dim: int, expansion: int,
        dropout: float = 0.0, project_history: bool = False,
    ) -> None:
        super().__init__()
        self.heads, self.dim = heads, dim
        width = heads * dim
        if not project_history and history_dim != width:
            raise ValueError(f"共享历史投影要求 history_dim=N*D={width}，实际 {history_dim}")
        self.projection = nn.Linear(history_dim, width) if project_history else nn.Identity()
        self.norm = RMSNorm(width)
        self.ffn = SwiGLU(width, expansion)
        self.key = PerHeadLinear(heads, dim, dim)
        self.value = PerHeadLinear(heads, dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, history: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_history = mask > 0
        aligned = self.projection(history[valid_history])
        hidden = aligned + self.dropout(self.ffn(self.norm(aligned)))
        hidden = hidden.reshape(-1, self.heads, self.dim)
        shape = (*history.shape[:2], self.heads, self.dim)
        key, value = history.new_zeros(shape), history.new_zeros(shape)
        key[valid_history], value[valid_history] = self.key(hidden), self.value(hidden)
        # 每个 Head 在 T 轴归一化；有限最小值避免全 mask 时 softmax(-inf) 产生 NaN。
        logits = torch.einsum("bnd,btnd->bnt", query, key) / math.sqrt(self.dim)
        valid = valid_history.unsqueeze(1)
        weights = torch.softmax(logits.masked_fill(~valid, torch.finfo(logits.dtype).min), dim=-1)
        weights = weights * valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.dropout(torch.einsum("bnt,btnd->bnd", weights, value))


class MixFormerBlock(nn.Module):
    """混合字段、读取历史、融合输出，输入输出 Query 均为 [B,N,D]。

    history: [B,T,N*D]；mask: [B,T]，True 有效。两套 Per-head FFN 不共享，
    历史 FFN/K/V 也不跨层共享。Cross-Attention 残差在此处加入。
    """

    def __init__(
        self, heads: int, dim: int, expansion: int, history_dim: int,
        history_expansion: int = 1, output_expansion: int | None = None,
        dropout: float = 0.0, project_history: bool = False,
    ) -> None:
        super().__init__()
        if output_expansion is None:
            output_expansion = expansion
        self.query_mixer = QueryMixer(heads, dim, expansion, dropout)
        self.cross_attention = HistoryCrossAttention(
            history_dim, heads, dim, history_expansion, dropout, project_history,
        )
        self.output_norm = RMSNorm(dim)
        self.output_ffn = SwiGLU(dim, output_expansion, heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, history: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_mixer(tokens)
        fused = query + self.cross_attention(query, history, mask)
        return fused + self.dropout(self.output_ffn(self.output_norm(fused)))


def recent_position_ids(mask: torch.Tensor) -> torch.Tensor:
    """将 [B,T] 有效 mask 转为 long 位置：最新有效点击为 1，padding 为 0。

    输入历史按从旧到新排列；反向累计有效点击数，使左/右/间断 padding 不影响位置。
    """
    valid = mask > 0
    return valid.flip(1).long().cumsum(1).flip(1) * valid
