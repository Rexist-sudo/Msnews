"""Dense RankMixer 的字段混合与前馈组件。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def token_mixing(tokens):
    """[B, T, D] -> [B, T, T, D/T] -> 交换 token/head -> [B, T, D]。"""
    batch, count, dim = tokens.shape
    if dim % count:
        raise ValueError("Token Mixing 要求 D 能被 token 数整除")
    return tokens.reshape(batch, count, count, dim // count).transpose(1, 2).reshape(batch, count, dim)


class PerTokenFFN(nn.Module):
    """每个 token 使用独立权重；批矩阵乘法避免 8 次 Python 循环。"""

    def __init__(self, tokens, dim, expansion):
        super().__init__()
        hidden = dim * expansion
        self.weight1 = nn.Parameter(torch.empty(tokens, dim, hidden))
        self.bias1 = nn.Parameter(torch.empty(tokens, 1, hidden))
        self.weight2 = nn.Parameter(torch.empty(tokens, hidden, dim))
        self.bias2 = nn.Parameter(torch.empty(tokens, 1, dim))
        nn.init.uniform_(self.weight1, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
        nn.init.uniform_(self.bias1, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
        nn.init.uniform_(self.weight2, -1 / math.sqrt(hidden), 1 / math.sqrt(hidden))
        nn.init.uniform_(self.bias2, -1 / math.sqrt(hidden), 1 / math.sqrt(hidden))

    def forward(self, tokens):
        # [T, B, D]，每个字段的参数相互独立。
        values = tokens.transpose(0, 1)
        hidden = F.gelu(torch.bmm(values, self.weight1) + self.bias1)
        return (torch.bmm(hidden, self.weight2) + self.bias2).transpose(0, 1)


class RankMixerBlock(nn.Module):
    def __init__(self, tokens, dim, expansion, mixing=True, shared=False):
        super().__init__()
        self.mixing = mixing
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = (nn.Sequential(nn.Linear(dim, dim * expansion), nn.GELU(), nn.Linear(dim * expansion, dim))
                    if shared else PerTokenFFN(tokens, dim, expansion))

    def forward(self, tokens):
        # 消融只移除 mixing 分支，保留两处 LayerNorm 与 FFN 残差。
        hidden = self.norm1(tokens + token_mixing(tokens) if self.mixing else tokens)
        return self.norm2(hidden + self.ffn(hidden))


