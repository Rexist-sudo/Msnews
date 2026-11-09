"""加载排序 checkpoint；历史 M1/M3 的配置与结构适配集中在此处。"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from .mixformer import MixFormerBlock


ROOT = Path(__file__).resolve().parents[3]


def _legacy_mixformer_model(model_class, config):
    """重建旧版 DIN+MixFormer，仅用于重放既有 128 维、均值池化权重。"""
    class LegacyMixFormer(model_class):
        def __init__(self, cfg):
            base_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            base_config.rankmixer_cfg.interaction = "rankmixer"
            super().__init__(base_config)
            self.config = cfg
            self.history_names = ["user_history", "user_history_clicked_category", "user_history_clicked_subcategory"]
            history_dim = sum(self._get_feature_embedding_dim(name) for name in self.history_names)
            self.interaction = nn.ModuleList([
                MixFormerBlock(
                    heads=int(cfg.rankmixer_cfg.token_count), dim=int(cfg.rankmixer_cfg.dim),
                    expansion=int(cfg.rankmixer_cfg.expansion), history_dim=history_dim,
                    history_expansion=int(cfg.mixformer_cfg.history_expansion), project_history=True,
                ) for _ in range(int(cfg.rankmixer_cfg.layers))
            ])

        def forward(self, batch):
            fields = self.get_inp_embedding(batch).split(self.field_dims, dim=1)
            tokens = torch.stack([projection(value) for projection, value in zip(self.field_projections, fields)], dim=1)
            history = torch.cat([self.get_batch_feature_embedding("base_embedding_table", name, batch)
                                 for name in self.history_names], dim=-1)
            for block in self.interaction:
                tokens = block(tokens, history, batch["user_history_mask"])
            return torch.sigmoid(self.score_head(tokens.mean(dim=1)))

    return LegacyMixFormer(config)


def _query_reread_model(model_class, config):
    """历史 M3 权重仅在加载时恢复原来的 DIN 重读结构。"""
    class QueryReread(model_class):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.query_dims = [self._get_feature_embedding_dim(self.array_item_feature_pairs[name])
                               for name in self.user_array_feature_names_din]
            self.query_boost = nn.Linear(int(cfg.rankmixer_cfg.token_count) * int(cfg.rankmixer_cfg.dim),
                                        sum(self.query_dims))

        def forward(self, batch):
            fields = self.get_inp_embedding(batch).split(self.field_dims, dim=1)
            tokens = torch.stack([projection(value) for projection, value in zip(self.field_projections, fields)], dim=1)
            hidden = self.interaction[0](tokens)
            context = tokens if self.config.reread_cfg.query_source == "static" else hidden
            deltas = self.query_boost(context.flatten(1)).split(self.query_dims, dim=1)
            updates = torch.zeros_like(hidden)
            for name, delta in zip(self.user_array_feature_names_din, deltas):
                index = self.field_names.index(name)
                query = self._get_single_feature_embedding_from_batch(batch, self.array_item_feature_pairs[name])
                history = self.get_batch_feature_embedding("base_embedding_table", name, batch)
                reread = self.din_array_attention_pooling(history, batch[name + "_mask"],
                            {"feature_name": name, "query_embedding_map": {name: query + delta}})
                updates[:, index] = F.linear(reread - fields[index], self.field_projections[index].weight)
            return torch.sigmoid(self.score_head(self.interaction[1](hidden + updates).mean(dim=1)))

    return QueryReread(config)


def load_model(model_class, path):
    saved = torch.load(path, map_location="cpu")
    legacy = "model" in saved
    config = OmegaConf.create(saved["config"] if legacy else saved["hyper_parameters"])
    state = saved["model"] if legacy else saved["state_dict"]
    if legacy:
        variant = config.experiment.get("model_variant", config.experiment.variant)
        if variant not in ("din", "m1", "m1_mlp", "m1_no_mixing", "m1_shared_ffn"):
            raise ValueError(f"不支持已移除的实验变体: {variant}")
        if variant != "din":
            config.rankmixer_cfg.interaction = "mlp" if variant == "m1_mlp" else "rankmixer"
            config.rankmixer_cfg.mixing = variant != "m1_no_mixing"
            config.rankmixer_cfg.shared_ffn = variant == "m1_shared_ffn"
        state = {key: value for key, value in state.items() if key not in ("item_count", "user_count")}
        config.paths.out_basedir = str(ROOT / "src/tmp")
        config.embeddings.embedding_tables.base_embedding_table.pretrain_embedding.item_title_entity_id = str(
            ROOT / "pretrain_embedding/title_entity_embedding.ckpt")
    if config.get("reread_cfg"):
        model = _query_reread_model(model_class, config)
    elif (config.get("rankmixer_cfg", {}).get("interaction") == "mixformer"
          and config.get("mixformer_cfg", {}).get("architecture_version") is None):
        model = _legacy_mixformer_model(model_class, config)
    else:
        model = model_class(config)
    model.load_state_dict(state, strict=True)
    return model
