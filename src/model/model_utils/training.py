"""排序训练共用入口：MINDDataModule、Lightning 与最佳 GAUC checkpoint。"""

import argparse
import json
import os
import platform
import time
import zipfile
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from .checkpoint import ROOT
from .evaluation import benchmark
from .ranking_metrics import evaluate
from ...dataset.DataReader.pl_dataloader import MINDDataModule
from ...dataset.FeaturesGenerator.ranking_support import load_support


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class RankingCheckpoint(L.Callback):
    """保留一份最佳权重和预测；训练中另存 last.ckpt 用于中断恢复。"""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.best_key = (-float("inf"), -float("inf"))
        self.history = []
        self.best_result = None

    def state_dict(self):
        return {"best_key": self.best_key, "history": self.history, "best_result": self.best_result}

    def load_state_dict(self, state):
        self.best_key, self.history, self.best_result = state["best_key"], state["history"], state["best_result"]

    def on_train_epoch_start(self, trainer, model):
        # 奇数批次也在半轮、轮末各验证一次，与已有 M1 的选权重时点一致。
        if trainer.val_check_interval == 0.5:
            trainer.val_check_batch = max(1, int(trainer.num_training_batches) // 2)

    def on_validation_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        overall = model.validation_metrics["overall"]
        key = (overall["GAUC"], -overall["LogLoss"])
        # Lightning 恢复验证边界时可能重放该次验证，避免重复记录。
        if self.history and self.history[-1]["step"] == trainer.global_step:
            self.history.pop()
        self.history.append({"epoch": trainer.current_epoch, "step": trainer.global_step, "overall": overall})
        write_json(self.directory / "history.json", self.history)
        if key > self.best_key:
            self.best_key = key
            self.best_result = {"name": model.config.name, "seed": int(model.train_hparams.seed),
                                "best_step": trainer.global_step, "metrics": model.validation_metrics}
            temporary = self.directory / "best.ckpt.tmp"
            trainer.save_checkpoint(temporary, weights_only=True)
            temporary.replace(self.directory / "best.ckpt")
            np.savez_compressed(self.directory / "best_predictions.npz", **model.validation_predictions)
            write_json(self.directory / "result.json", self.best_result)
        temporary = self.directory / "last.ckpt.tmp"
        trainer.save_checkpoint(temporary)
        temporary.replace(self.directory / "last.ckpt")
        if trainer.val_check_interval == 0.5:
            trainer.val_check_batch = int(trainer.num_training_batches)


def fit(model_class, config, directory, resume=None, **trainer_options):
    """模型负责 loss/优化器和验证；此处只组织训练与产物。"""
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(f"输出目录非空，请使用新目录或 --resume: {directory}")
    if resume is not None and Path(resume).resolve().parent != directory:
        raise ValueError("续训 checkpoint 必须位于本次输出目录")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("请在启动 Python 前设置 PYTHONHASHSEED=42，固定原模型的 embedding 初始化顺序")
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.train_hparams.seed = 42
    if resume is not None and OmegaConf.load(directory / "config.yaml") != config:
        raise ValueError("续训配置必须与原运行一致")
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, directory / "config.yaml")
    if resume is None:
        source_root = Path(__file__).resolve().parents[2]
        with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source_root.rglob("*")):
                if path.suffix in (".py", ".yaml") and "tmp" not in path.relative_to(source_root).parts:
                    archive.write(path, path.relative_to(source_root.parent))
        provenance = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if config.get("statistics_cfg", {}).get("mode") is not None:
            metadata = Path(config.paths.out_basedir) / "extractored_feature/news_statistics.json"
            provenance["statistics"] = json.loads(metadata.read_text(encoding="utf-8"))
        write_json(directory / "provenance.json", provenance)
    L.seed_everything(42, workers=True)
    torch.set_num_threads(4)
    model = model_class(config)
    model._checkpoint_managed_by_trainer = True
    model.support = load_support(config)
    data = MINDDataModule(config)
    checkpoint = RankingCheckpoint(directory)
    options = dict(max_epochs=int(config.train_hparams.max_epoch),
                   val_check_interval=config.train_hparams.val_freq,
                   accelerator=config.train_hparams.get("device", "gpu"),
                   devices=list(config.train_hparams.gpus) if config.train_hparams.get("device", "gpu") == "gpu" else 1,
                   deterministic=True, precision="32-true", num_sanity_val_steps=0,
                   enable_checkpointing=False, callbacks=[checkpoint],
                   logger=TensorBoardLogger(save_dir=str(directory), name="", version=""))
    options.update(trainer_options)
    trainer = L.Trainer(**options)
    started = time.perf_counter()
    try:
        trainer.fit(model, datamodule=data, ckpt_path=resume)
    finally:
        data.teardown("fit")
    if checkpoint.best_result is None:
        raise RuntimeError("训练结束但没有验证结果")
    fit_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(directory / "best.ckpt", map_location="cpu")["state_dict"])
    model.to(trainer.strategy.root_device)
    batch = next(iter(data.val_dataloader()))
    batch = {name: value[:512] for name, value in batch.items()}
    result = dict(checkpoint.best_result, total_steps=trainer.global_step, fit_seconds=fit_seconds)
    with np.load(directory / "best_predictions.npz") as predictions:
        names = ("user_id", "item_id", "label", "score", "impression_id")
        result["metrics"] = evaluate(*(predictions[name] for name in names), support=model.support)
    result["cost"] = {"parameters": sum(p.numel() for p in model.parameters()),
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "interaction_parameters": sum(p.numel() for name, p in model.named_parameters()
                                                    if not name.startswith(("embedding_tables.", "array_attention_nets.", "array_query_projs."))),
                      "inference": benchmark(model, batch)}
    data.teardown("fit")
    write_json(directory / "result.json", result)
    (directory / "last.ckpt").unlink()
    return result


def main(model_class, model_dir, defaults):
    parser = argparse.ArgumentParser(description=f"{model_class.__name__} Training")
    parser.add_argument("--config", "-c", type=Path, default=model_dir.parent / "base_sort_conf.yaml")
    parser.add_argument("--extra_config", "-e", type=Path, nargs="+", default=[])
    parser.add_argument("--output", type=Path, help="新训练目录；默认 experiments/<模型>_<时间>")
    parser.add_argument("--resume", type=Path, help="从本次运行的 last.ckpt 继续")
    args = parser.parse_args()
    config = OmegaConf.merge(*(OmegaConf.load(path) for path in [args.config, *defaults, *args.extra_config]))
    directory = args.output or (args.resume.parent if args.resume else ROOT / "experiments" / (
        config.name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    result = fit(model_class, config, directory, resume=args.resume)
    print(json.dumps(result["metrics"]["overall"], ensure_ascii=False, indent=2))
