"""复用 DIN 三路兴趣提取，以 Dense RankMixer 完成字段交互。"""

import torch
from torch import nn
from torch.nn import functional as F

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

        self.reread_source = self.config.get("reread_cfg", {}).get("query_source")
        if self.reread_source is not None:
            if self.reread_source not in ("static", "feedback") or interaction != "rankmixer" or len(self.interaction) != 2:
                raise ValueError("历史重读使用两层 RankMixer，query_source 为 static 或 feedback")
            self.reread_indices = [self.field_names.index(name) for name in self.user_array_feature_names_din]
            self.query_dims = [self._get_feature_embedding_dim(self.array_item_feature_pairs[name])
                               for name in self.user_array_feature_names_din]
            self.query_boost = nn.Linear(count * dim, sum(self.query_dims))
            # 初始 query 增量为零，两组从相同的首次读取开始学习。
            nn.init.zeros_(self.query_boost.weight)
            nn.init.zeros_(self.query_boost.bias)

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
        if self.reread_source is None:
            tokens = self.interaction(tokens)
        else:
            hidden = self.interaction[0](tokens)
            context = tokens if self.reread_source == "static" else hidden
            query_deltas = self.query_boost(context.flatten(1)).split(self.query_dims, dim=1)
            updates = torch.zeros_like(hidden)
            for name, index, delta in zip(self.user_array_feature_names_din, self.reread_indices, query_deltas):
                query = self._get_single_feature_embedding_from_batch(batch, self.array_item_feature_pairs[name])
                history = self.get_batch_feature_embedding("base_embedding_table", name, batch)
                reread = self.din_array_attention_pooling(history, batch[name + "_mask"],
                            {"feature_name": name, "query_embedding_map": {name: query + delta}})
                # 复用字段投影的线性部分，使 r1=r0 时回注严格为零。
                updates[:, index] = F.linear(reread - fields[index], self.field_projections[index].weight)
            tokens = self.interaction[1](hidden + updates)
        return torch.sigmoid(self.score_head(tokens.mean(dim=1)))
