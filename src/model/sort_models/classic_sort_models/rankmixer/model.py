"""排序模型装配：M1/M2 使用 DIN+RankMixer，M3 使用完整 MixFormer。

候选和逐条历史共用基础 embedding 与 M2 统计投影；训练、验证沿用公共接口。
"""

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..din.model import DIN
from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.rankmixer import RankMixerBlock
from ....model_utils.mixformer import MixFormerBlock, PerHeadLinear, RMSNorm, recent_position_ids
from ....model_utils.utils import MLP


class RankMixerDIN(DIN):
    """按配置选择主干，返回 [B,1] 点击概率，继承 DIN 的 BCE/AdamW 接口。

    M3 仅初始化 BaseModelSort 的公共 embedding/评估状态，不创建 DIN 注意力。
    必需输入：五个非序列 ID [B]、三路历史 ID/mask [B,T]、候选统计 [B,4]、
    历史统计 [B,T,4] 和对应 long 状态 ID；ID 为整数，统计为 float32。
    """

    def __init__(self, config: DictConfig | str) -> None:
        config = OmegaConf.load(config) if isinstance(config, str) else config
        self.use_mixformer = config.rankmixer_cfg.get("interaction", "rankmixer") == "mixformer"
        if self.use_mixformer:
            BaseModelSort.__init__(self, config)
            self._build_mixformer()
        else:
            super().__init__(config)
            self._build_rankmixer()
        self._build_statistics()
        if self.use_mixformer and self.statistics_mode != "smooth":
            raise ValueError("MixFormer 沿用完整 M2 的平滑统计和联合状态输入")

    def _build_rankmixer(self) -> None:
        """保留 M1/M2 的字段顺序、模块名称和初始化顺序，以兼容既有权重。"""
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

    def _build_mixformer(self) -> None:
        """112 维字段等分八段；80 维历史共享投影；两层 Block 后展平打分。"""
        cfg, mixer = self.config.rankmixer_cfg, self.config.mixformer_cfg
        if mixer.get("architecture_version") != 2:
            raise ValueError("当前 M3 要求 architecture_version=2；旧权重请使用 checkpoint.load_model")
        self.num_heads, self.head_dim = int(cfg.token_count), int(cfg.dim)
        layers, dropout = int(cfg.layers), float(mixer.dropout)
        self.field_names = ["user_id", "user_click_category", "item_id", "category", "subcategory"]
        self.field_dims = [self._get_feature_embedding_dim(name) for name in self.field_names]
        self.history_names = ["user_history", "user_history_clicked_category", "user_history_clicked_subcategory"]
        self.history_dims = [self._get_feature_embedding_dim(name) for name in self.history_names]
        width = self.num_heads * self.head_dim
        field_width = sum(self.field_dims)
        if self.num_heads <= 0 or field_width % self.num_heads or layers <= 0:
            raise ValueError(f"字段总维度 {field_width} 必须能被正数 N={self.num_heads} 整除，layers={layers} > 0")
        self.field_projections = PerHeadLinear(self.num_heads, field_width // self.num_heads, self.head_dim)
        self.history_length = int(self.config.features.array_max_length.user_history)
        if self.history_length <= 0 or any(
            int(self.config.features.array_max_length[name]) != self.history_length for name in self.history_names
        ):
            raise ValueError(f"三路历史要求相同正长度，实际 {dict(self.config.features.array_max_length)}")
        position_dim = int(mixer.position_dim)
        if position_dim <= 0:
            raise ValueError(f"position_dim 必须 > 0，实际 {position_dim}")
        self.history_position = nn.Embedding(self.history_length + 1, position_dim, padding_idx=0)
        self.history_projection = nn.Linear(sum(self.history_dims) + position_dim, width)
        self.interaction = nn.ModuleList([
            MixFormerBlock(
                heads=self.num_heads, dim=self.head_dim, expansion=int(cfg.expansion), history_dim=width,
                history_expansion=int(mixer.history_expansion), output_expansion=int(mixer.output_expansion),
                dropout=dropout,
            ) for _ in range(layers)
        ])
        self.score_head = nn.Sequential(
            RMSNorm(width), nn.Linear(width, int(mixer.score_hidden_dim)), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(int(mixer.score_hidden_dim), 1),
        )

    def _build_statistics(self) -> None:
        """M2 候选/历史共享 44→32 投影；状态 0 为可学习的零曝光状态。"""
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

    def get_batch_feature_embedding(
        self, embedding_tables_name: str, feature_name: str, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """共用 BaseModel 字段接口，新闻 ID 字段额外融合对应时点的 M2 统计。"""
        embedding = super().get_batch_feature_embedding(embedding_tables_name, feature_name, batch)
        if self.statistics_mode is not None and feature_name in ("item_id", "user_history"):
            prefix = "item" if feature_name == "item_id" else "history"
            parts = [embedding, batch[prefix + "_statistics"]]
            if self.statistics_state is not None:
                parts.append(self.statistics_state(batch[prefix + "_state"].long()))
            embedding = self.statistics_projection(torch.cat(parts, dim=-1))
        return embedding

    def _build_feature_heads(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """按 field_names 拼接 [B,112]，连续等分为 [B,8,14] 后独立投影。"""
        fields = torch.cat([
            self.get_batch_feature_embedding("base_embedding_table", name, batch)
            for name in self.field_names
        ], dim=-1)
        return self.field_projections(fields.reshape(fields.shape[0], self.num_heads, -1))

    def _build_history_repr(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回共享 S^proj [B,T,512] 和有效 mask [B,T]；padding 投影结果置零。"""
        mask = batch["user_history_mask"] > 0
        if mask.shape[1] > self.history_length:
            raise ValueError(f"历史长度 {mask.shape[1]} 超过配置 {self.history_length}")
        history = torch.cat([
            self.get_batch_feature_embedding("base_embedding_table", name, batch)
            for name in self.history_names
        ] + [self.history_position(recent_position_ids(mask))], dim=-1)
        # 共享投影每个候选仅执行一次，各 Block 接收同一份 S^proj。
        projected = self.history_projection(history)
        return projected.masked_fill(~mask.unsqueeze(-1), 0), mask

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """编码 → 交互 → 打分；sigmoid 仅执行一次，输出供继承的 BCE 使用。"""
        if self.use_mixformer:
            tokens = self._build_feature_heads(batch)
            history, mask = self._build_history_repr(batch)
            for block in self.interaction:
                tokens = block(tokens, history, mask)
            logits = self.score_head(tokens.flatten(1))
            return torch.sigmoid(logits)
        fields = self.get_inp_embedding(batch).split(self.field_dims, dim=1)
        tokens = torch.stack([projection(value) for projection, value in zip(self.field_projections, fields)], dim=1)
        if self.score_head is None:
            return torch.sigmoid(self.interaction(tokens.flatten(1)))
        tokens = self.interaction(tokens)
        return torch.sigmoid(self.score_head(tokens.mean(dim=1)))
