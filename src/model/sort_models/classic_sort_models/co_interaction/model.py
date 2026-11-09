"""继承 M2_2 编码和公共权重，装配两层 Field→History→Interest→Field 主干。"""

from functools import partial

import torch
from torch import nn

from ..rankmixer.model import RankMixerDIN
from .layers import CoBlock


HISTORY_FIELDS = ("user_history", "user_history_clicked_category", "user_history_clicked_subcategory")
HISTORY_DIMS = (32, 16, 16)
FIELD_NAMES = ("user_click_category", "user_id", *HISTORY_FIELDS, "category", "item_id", "subcategory")


class CoInteractionRankMixer(RankMixerDIN):
    """M3：固定 T=50、8×128 字段、50×64 历史、两层独立 ContextFFN。

    batch 沿用 M2_2：五个整数基础字段 [B,1]，三路整数历史 [B,50] 及
    有效 mask，候选/历史浮点 statistics [B,4]/[B,50,4] 和 state 整数。
    forward/inference 返回 [B,1] 概率；forward_with_trace 另返回 logit 和层状态。
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        cfg = self.config.co_interaction_cfg
        if int(cfg.hidden_dim) != 32:
            raise ValueError(f"M3 要求 co_interaction_cfg.hidden_dim=32，实际为 {cfg.hidden_dim}")
        expected = {"token_count": 8, "dim": 128, "layers": 2, "expansion": 2,
                    "interaction": "rankmixer", "mixing": True, "shared_ffn": False}
        for name, value in expected.items():
            if self.config.rankmixer_cfg.get(name) != value:
                raise ValueError(f"M3 要求 rankmixer_cfg.{name}={value}，实际为 {self.config.rankmixer_cfg.get(name)}")
        if self.statistics_mode != "smooth" or self.statistics_state.embedding_dim != 8:
            raise ValueError("M3 要求 statistics_cfg.mode=smooth、state_dim=8")
        if tuple(self.field_names) != FIELD_NAMES:
            raise ValueError(f"M3 字段顺序须为 {FIELD_NAMES}，实际为 {self.field_names}")
        pairs = dict(zip(HISTORY_FIELDS, ("item_id", "category", "subcategory")))
        if self.array_item_feature_pairs != pairs:
            raise ValueError(f"M3 DIN 配对须为 {pairs}，实际为 {self.array_item_feature_pairs}")
        dims = tuple(self._get_feature_embedding_dim(name) for name in HISTORY_FIELDS)
        if dims != HISTORY_DIMS or any(self.array_max_length[name] != 50 for name in HISTORY_FIELDS):
            raise ValueError(f"M3 历史要求 T=50、维度={HISTORY_DIMS}，实际维度={dims}")
        self.interest_indices = tuple(self.field_names.index(name) for name in HISTORY_FIELDS)
        self.co_blocks = nn.ModuleList([
            CoBlock(sum(HISTORY_DIMS), int(self.config.rankmixer_cfg.dim), int(cfg.hidden_dim))
            for _ in range(2)
        ])

    def initialization_state_dict(self) -> dict[str, torch.Tensor]:
        """公共初始化仅含 M2_2 参数，新增 ContextFFN 保留常规/零输出初始化。"""
        return {name: value for name, value in self.state_dict().items() if not name.startswith("co_blocks.")}

    def _build_inputs(self, batch: dict) -> tuple:
        """共享 M2 新闻投影，建立 P_f(0)、原始候选 Query 和已屏蔽历史。"""
        mask = batch[HISTORY_FIELDS[0] + "_mask"] > 0
        if mask.ndim != 2 or mask.shape[1] != 50:
            raise ValueError(f"M3 历史 mask 要求 [B,50]，实际为 {tuple(mask.shape)}")
        histories, queries = [], {}
        for name in HISTORY_FIELDS:
            if not torch.equal(mask, batch[name + "_mask"] > 0):
                raise ValueError(f"三路历史 mask 必须一致，{name}_mask 不匹配")
            value = self.get_batch_feature_embedding("base_embedding_table", name, batch)
            histories.append(torch.where(mask.unsqueeze(-1), value, 0.0))
            queries[name] = self._get_single_feature_embedding_from_batch(batch, self.array_item_feature_pairs[name])
        history = torch.cat(histories, dim=-1)
        interests = tuple(history.new_zeros((len(history), dim)) for dim in HISTORY_DIMS)
        zero_interests = dict(zip(HISTORY_FIELDS, interests))
        fields = [zero_interests[name] if name in zero_interests
                  else self._get_single_feature_embedding_from_batch(batch, name) for name in self.field_names]
        tokens = torch.stack([projection(value) for projection, value in zip(self.field_projections, fields)], dim=1)
        return tokens, history, interests, queries, mask

    def _read_interests(self, history: torch.Tensor, *, queries: dict, mask: torch.Tensor) -> tuple:
        """两层调用同一套 DIN 打分网络和候选 Query，按 32/16/16 切片读取。"""
        interests, weights = [], []
        for name, values in zip(HISTORY_FIELDS, history.split(HISTORY_DIMS, dim=-1)):
            attention = self.din_attention_weights(
                values, mask, {"feature_name": name, "query_embedding_map": queries},
            )
            interests.append((values * attention.unsqueeze(-1)).sum(dim=1))
            weights.append(attention)
        return tuple(interests), tuple(weights)

    def _forward_logits(self, batch: dict, trace: bool = False) -> tuple[torch.Tensor, dict]:
        tokens, history, interests, queries, mask = self._build_inputs(batch)
        states = {"initial_tokens": tokens, "initial_history": history, "mask": mask, "layers": []} if trace else {}
        reader = partial(self._read_interests, queries=queries, mask=mask)
        for block, rankmixer in zip(self.co_blocks, self.interaction):
            tokens, history, interests, layer = block(
                tokens, history, interests, mask, reader, self.interest_indices,
                self.field_projections, rankmixer, trace=trace,
            )
            if trace:
                states["layers"].append(layer)
        return self.score_head(tokens.mean(dim=1)), states

    def forward(self, batch: dict) -> torch.Tensor:
        """编码 → 两层 CoBlock → 原均值池化与打分头，返回 [B,1] 概率。"""
        logits, _ = self._forward_logits(batch)
        return torch.sigmoid(logits)

    def forward_with_trace(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """分析接口返回 sigmoid 前的 logit 及状态；调用方控制 eval/no_grad。"""
        return self._forward_logits(batch, trace=True)
