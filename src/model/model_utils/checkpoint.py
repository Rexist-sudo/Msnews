"""加载排序 checkpoint 与公共初始化；集中适配旧 M1 配置和无用计数 buffer。"""

import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[3]


def load_initial_parameters(model, path) -> dict:
    """按完整公共键集合加载初始化快照，拒绝缺失或额外参数，返回来源审计。

    模型可用 initialization_state_dict 排除新增模块；新路径保存公共随机
    初始化，不保存训练 checkpoint。已有文件始终只读。
    """
    path = Path(path)
    expected = (model.initialization_state_dict() if hasattr(model, "initialization_state_dict")
                else model.state_dict())
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(expected, path)
    state = torch.load(path, map_location="cpu")
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"公共初始化键不完整: missing={missing}, unexpected={unexpected}")
    incompatible = model.load_state_dict(state, strict=False)
    return {"source": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "loaded_keys": len(state), "new_keys": incompatible.missing_keys,
            "mapping": "公共参数按名称一一加载；新增参数保留模型初始化"}


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
