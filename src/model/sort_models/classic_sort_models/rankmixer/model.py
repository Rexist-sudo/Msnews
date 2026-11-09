"""复用 DIN 三路兴趣提取，以 Dense RankMixer 完成字段交互。"""

import torch
from torch import nn

from ..din.model import DIN
from ....model_utils.rankmixer import RankMixerBlock
from ....model_utils.utils import MLP


class RankMixerDIN(DIN):
    """三路 DIN 兴趣与用户、候选字段投影为 token，交互后输出点击概率。"""

    def __init__(self, config):
        super().__init__(config)
        cfg = self.config.rankmixer_cfg
        dim, count = int(cfg.dim), int(cfg.token_count)
        self.field_names = (self.user_non_array_feature_names + self.user_array_feature_names_din
                            + self.user_array_feature_names_default + sorted(self.item_feature_names))
        self.field_dims = [self._get_feature_embedding_dim(name) for name in self.field_names]
        if len(self.field_names) != count or dim % count:
            raise ValueError(f"字段数应为 {count}，D 应能被字段数整除，实际字段为 {self.field_names}")
        self.field_projections = nn.ModuleList([nn.Linear(size, dim) for size in self.field_dims])
        del self.score_fc
        interaction = cfg.get("interaction", "rankmixer")
        if interaction == "mlp":
            self.interaction = MLP([count * dim] + list(cfg.mlp_hidden_dims))
            self.score_head = None
        elif interaction == "rankmixer":
            self.interaction = nn.Sequential(*[
                RankMixerBlock(count, dim, int(cfg.expansion),
                               mixing=cfg.get("mixing", True), shared=cfg.get("shared_ffn", False))
                for _ in range(int(cfg.layers))])
            self.score_head = MLP([dim, 64, 1])
        else:
            raise ValueError(f"未知字段交互方式: {interaction}")

        statistics = self.config.get("statistics_cfg", {})
        self.statistics_mode = statistics.get("mode")
        if self.statistics_mode is not None:
            if self.statistics_mode not in ("raw", "smooth"):
                raise ValueError(f"未知统计方式: {self.statistics_mode}")
            state_dim = int(statistics.get("state_dim", 8)) if self.statistics_mode == "smooth" else 0
            self.statistics_state = nn.Embedding(26, state_dim) if state_dim else None
            news_dim = self._get_feature_embedding_dim("item_id")
            # 候选与每条历史共用投影，保持新闻 ID 路及字段 token 的输入维度。
            self.statistics_projection = nn.Linear(news_dim + 4 + state_dim, news_dim)

    def get_batch_feature_embedding(self, embedding_tables_name, feature_name, batch):
        embedding = super().get_batch_feature_embedding(embedding_tables_name, feature_name, batch)
        if self.statistics_mode is not None and feature_name in ("item_id", "user_history"):
            prefix = "item" if feature_name == "item_id" else "history"
            parts = [embedding, batch[prefix + "_statistics"]]
            if self.statistics_state is not None:
                parts.append(self.statistics_state(batch[prefix + "_state"].long()))
            embedding = self.statistics_projection(torch.cat(parts, dim=-1))
        return embedding

    def forward(self, batch):
        fields = self.get_inp_embedding(batch).split(self.field_dims, dim=1)
        tokens = torch.stack([projection(value) for projection, value in zip(self.field_projections, fields)], dim=1)
        if self.score_head is None:
            return torch.sigmoid(self.interaction(tokens.flatten(1)))
        tokens = self.interaction(tokens)
        return torch.sigmoid(self.score_head(tokens.mean(dim=1)))
