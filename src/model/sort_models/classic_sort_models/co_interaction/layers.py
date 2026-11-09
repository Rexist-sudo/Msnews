"""M3 专用历史更新与兴趣增量写回；DIN、字段投影和 RankMixer 由模型共享。"""

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ContextFFN(nn.Module):
    """字段均值条件化逐位置历史残差，输入 [B,N,D]、[B,T,H]、有效 mask。

    使用 SiLU(W_s LN(S)) * W_c MeanPool(X)，输出投影零初始化。
    mask=True 表示有效历史；padding 在进入 FFN 前及残差后均清零。
    """

    def __init__(self, history_dim: int, field_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if min(history_dim, field_dim, hidden_dim) <= 0:
            raise ValueError(f"ContextFFN 维度须为正数，实际为 {history_dim, field_dim, hidden_dim}")
        self.norm = nn.LayerNorm(history_dim)
        self.history_projection = nn.Linear(history_dim, hidden_dim)
        self.context_projection = nn.Linear(field_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, history_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, tokens: torch.Tensor, history: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回更新的 [B,T,H]；空历史及 padding 的输出恒为零。"""
        valid = (mask > 0).unsqueeze(-1)
        history = torch.where(valid, history, 0.0)
        hidden = F.silu(self.history_projection(self.norm(history)))
        context = self.context_projection(tokens.mean(dim=1)).unsqueeze(1)
        updated = history + self.output_projection(hidden * context)
        return torch.where(valid, updated, 0.0)


class CoBlock(nn.Module):
    """Context FFN → 共享 DIN → 兴趣差值写回 → 原 RankMixer Block。

    仅持有本层 ContextFFN 参数。外部传入 DIN、投影、RankMixer，避免同一
    公共参数重复注册；旧 M2_2 state_dict 键保持原位置，可按名称直接加载。
    """

    def __init__(self, history_dim: int, field_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context_ffn = ContextFFN(history_dim, field_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        history: torch.Tensor,
        previous_interests: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
        read_interests: Callable,
        interest_indices: Sequence[int],
        field_projections: nn.ModuleList,
        rankmixer: nn.Module,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], dict]:
        """传递字段、历史、三路兴趣及可选轨迹；三路顺序固定为 ID/Cat/Sub。"""
        updated_history = self.context_ffn(tokens, history, mask)
        interests, attention = read_interests(updated_history)
        deltas = tuple(current - previous for current, previous in zip(interests, previous_interests))
        updates = torch.zeros_like(tokens)
        for index, delta in zip(interest_indices, deltas):
            # bias 已包含于初始 P_f(0)，每层只写入线性差值，避免重复加 bias。
            updates[:, index] = F.linear(delta, field_projections[index].weight)
        interest_tokens = tokens + updates
        output = rankmixer(interest_tokens)
        states = ({"history": updated_history, "interests": interests, "attention": attention,
                   "interest_deltas": deltas, "interest_tokens": interest_tokens, "tokens": output}
                  if trace else {})
        return output, updated_history, interests, states
