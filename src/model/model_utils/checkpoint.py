"""加载排序 checkpoint，旧 M1 仅在此处转换配置和移除无用计数 buffer。"""

from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[3]


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
    model = model_class(config)
    model.load_state_dict(state, strict=True)
    return model
