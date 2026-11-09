"""排序训练共用入口：MINDDataModule、Lightning 与最佳 GAUC checkpoint。"""

import argparse
import hashlib
import json
import os
import platform
import shutil
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
from .validation_guard import ValidationGuard
from .training_state import TrainingState
from ...dataset.DataReader.pl_dataloader import MINDDataModule
from ...dataset.FeaturesGenerator.ranking_support import load_support


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class TrainingProgress(L.Callback):
    """按优化器 step 输出低频进度，避免逐 batch 刷新终端和 IDE。

    每 interval 个 step 向标准输出和实验目录 progress.log 追加一行。
    只记录当前 batch 的标量 loss，不保留计算图；梯度累积时同一步仅输出一次。
    """

    def __init__(self, directory: Path, interval: int = 100) -> None:
        super().__init__()
        if interval <= 0:
            raise ValueError(f"log_every_n_steps 必须 > 0，实际 {interval}")
        self.path = Path(directory) / "progress.log"
        self.interval = interval
        self.last_logged_step = 0

    def on_train_start(self, trainer: L.Trainer, model: L.LightningModule) -> None:
        # 续训时从已恢复的 global_step 继续，不重复输出恢复边界。
        self.last_logged_step = trainer.global_step

    def on_train_batch_end(
        self, trainer: L.Trainer, model: L.LightningModule,
        outputs: dict[str, torch.Tensor] | torch.Tensor | None,
        batch: dict[str, torch.Tensor], batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if (not trainer.is_global_zero or step <= self.last_logged_step
                or step % self.interval != 0):
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        loss_text = f"{loss.detach().item():.6f}" if isinstance(loss, torch.Tensor) else "n/a"
        epoch_limit = "unlimited" if trainer.max_epochs == -1 else str(trainer.max_epochs)
        line = (f"epoch={trainer.current_epoch + 1}/{epoch_limit} "
                f"step={step} batch={batch_idx + 1}/{int(trainer.num_training_batches)} "
                f"loss={loss_text}")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
        self.last_logged_step = step


class RankingCheckpoint(L.Callback):
    """保留一份最佳权重和预测；训练中另存 last.ckpt 用于中断恢复。"""

    def __init__(self, directory, guard: ValidationGuard | None = None):
        self.directory = Path(directory)
        self.guard = guard
        self.pause_reason = None
        self.best_key = (-float("inf"), -float("inf"))
        self.history = []
        self.best_result = None
        self.resume_step = None

    def state_dict(self):
        return {"best_key": self.best_key, "history": self.history, "best_result": self.best_result}

    def load_state_dict(self, state):
        self.best_key, self.history, self.best_result = state["best_key"], state["history"], state["best_result"]
        self.resume_step = self.history[-1]["step"] if self.history else None

    def on_train_start(self, trainer, model):
        if self.guard is not None:
            if trainer.val_check_interval != 0.5 or trainer.accumulate_grad_batches != 1:
                raise ValueError("validation_guard 当前要求 val_freq=0.5、accumulate_grad_batches=1")
            self.guard.validate_schedule(int(trainer.num_training_batches), trainer.max_epochs)

    def on_train_epoch_start(self, trainer, model):
        # 奇数批次也在半轮、轮末各验证一次，与已有 M1 的选权重时点一致。
        if trainer.val_check_interval == 0.5:
            batches = int(trainer.num_training_batches)
            half = max(1, batches // 2)
            # 半轮 checkpoint 已包含完整验证，恢复后下一次应在本轮末验证。
            resumed_half = (trainer.global_step == self.resume_step
                            and half <= trainer.global_step % batches < batches)
            trainer.val_check_batch = batches if resumed_half else half

    def on_validation_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        overall = model.validation_metrics["overall"]
        key = (overall["GAUC"], -overall["LogLoss"])
        # Lightning 恢复验证边界时可能重放该次验证，避免重复记录。
        replayed = bool(self.history and self.history[-1]["step"] == trainer.global_step)
        comparison = None
        if replayed:
            comparison = self.history[-1].get("comparison")
            self.history.pop()
        record = {"epoch": trainer.current_epoch, "step": trainer.global_step, "overall": overall}
        if self.guard is not None:
            # 显式续训允许越过已暂停的验证边界；重复验证不重复触发同一个暂停。
            if not replayed:
                comparison = self.guard.check(record, self.history[-1] if self.history else None)
            record["comparison"] = comparison
        self.history.append(record)
        write_json(self.directory / "history.json", self.history)
        if self.guard is not None:
            write_json(self.directory / "validation_comparison.json", self.history)
            for name, values in comparison["references"].items():
                deltas = ", ".join(f"{metric}={value:+.6f}" for metric, value in values["improvement"].items())
                print(f"Validation comparison step={trainer.global_step} vs {name}: "
                      f"worse={len(values['worse_metrics'])}/8, improvement: {deltas}", flush=True)
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
        if comparison is not None and comparison["pause"] and not replayed:
            self.pause_reason = comparison["reasons"]
            trainer.should_stop = True
            print(f"Training paused at step={trainer.global_step}: {self.pause_reason}; "
                  "last.ckpt saved for explicit resume.", flush=True)
        if trainer.val_check_interval == 0.5:
            trainer.val_check_batch = int(trainer.num_training_batches)


def _prepare_resume(config, directory: Path, resume: Path) -> tuple[Path, dict]:
    """只允许延长预算或改停止规则；跨目录续训复制恢复点及原最佳产物并记录来源。"""
    source = resume.resolve().parent
    original = OmegaConf.load(source / "config.yaml")
    expected, actual = [OmegaConf.to_container(value, resolve=True) for value in (original, config)]
    for value in (expected, actual):
        value["train_hparams"].pop("max_epoch", None)
        value.pop("validation_guard", None)
    if expected != actual:
        raise ValueError("续训只允许修改 train_hparams.max_epoch 与 validation_guard，模型/数据/优化器必须一致")
    if source == directory and original != config:
        raise ValueError("修改续训预算或停止规则时请指定新的空输出目录，以保留原运行记录")
    if source != directory and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"续训分支输出目录非空: {directory}")
    saved = torch.load(resume, map_location="cpu")
    if not saved.get("optimizer_states"):
        raise ValueError("续训必须使用含优化器状态的完整 checkpoint（last.ckpt）")
    progress = saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]
    previous_result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    info = {"source_checkpoint": str(resume.resolve()),
            "sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
            "epoch": int(saved["epoch"]), "step": int(saved["global_step"]),
            "completed_batches": int(progress["completed"]),
            "rng_available": "ranking_rng_state" in saved,
            "prior_fit_seconds": float(previous_result.get("fit_seconds", 0)),
            "original_config": OmegaConf.to_container(original, resolve=True)}
    del saved
    directory.mkdir(parents=True, exist_ok=True)
    if source != directory:
        shutil.copy2(resume, directory / "resume.ckpt")
        for name in ("best.ckpt", "best_predictions.npz", "history.json", "validation_comparison.json",
                     "validation_reference.json", "result.json"):
            if (source / name).exists():
                shutil.copy2(source / name, directory / name)
        resume = directory / "resume.ckpt"
    write_json(directory / "resume_info.json", info)
    return resume, info


def fit(model_class, config, directory, resume=None, **trainer_options):
    """模型负责 loss/优化器和验证；此处只组织训练与产物。"""
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(f"输出目录非空，请使用新目录或 --resume: {directory}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("请在启动 Python 前设置 PYTHONHASHSEED=42，固定原模型的 embedding 初始化顺序")
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.train_hparams.seed = 42
    log_interval = int(config.train_hparams.get("log_every_n_steps", 100))
    progress = TrainingProgress(directory, interval=log_interval)
    branch_resume = resume is not None and Path(resume).resolve().parent != directory
    resume_info = None
    if resume is not None:
        resume, resume_info = _prepare_resume(config, directory, Path(resume))
    guard = None
    reference_snapshot = None
    guard_config = config.get("validation_guard", {})
    if guard_config.get("enabled", False):
        mode = guard_config.get("mode", "baseline_and_previous")
        if mode == "baseline_and_previous":
            if resume is not None:
                reference_snapshot = json.loads((directory / "validation_reference.json").read_text(encoding="utf-8"))
            else:
                reference_path = Path(guard_config.reference_history).resolve()
                reference_bytes = reference_path.read_bytes()
                reference_snapshot = {"source": str(reference_path),
                                      "sha256": hashlib.sha256(reference_bytes).hexdigest(),
                                      "history": json.loads(reference_bytes)}
        guard = ValidationGuard(reference_snapshot["history"] if reference_snapshot else None, mode=mode)
    directory.mkdir(parents=True, exist_ok=True)
    if reference_snapshot is not None:
        write_json(directory / "validation_reference.json", reference_snapshot)
    OmegaConf.save(config, directory / "config.yaml")
    if resume is None or branch_resume:
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
    if resume_info is not None:
        data.set_resume_position(resume_info["epoch"], resume_info["completed_batches"])
        print(f"Resuming step={resume_info['step']}, epoch={resume_info['epoch']}, "
              f"completed_batches={resume_info['completed_batches']}, "
              f"saved_rng={resume_info['rng_available']}", flush=True)
    checkpoint = RankingCheckpoint(directory, guard=guard)
    options = dict(max_epochs=int(config.train_hparams.max_epoch),
                   val_check_interval=config.train_hparams.val_freq,
                   accelerator=config.train_hparams.get("device", "gpu"),
                   devices=list(config.train_hparams.gpus) if config.train_hparams.get("device", "gpu") == "gpu" else 1,
                   deterministic=True, precision=config.train_hparams.get("precision", "32-true"),
                   gradient_clip_val=float(config.train_hparams.get("grad_clip", 0.0)),
                   gradient_clip_algorithm="norm", num_sanity_val_steps=0,
                   enable_checkpointing=False, enable_progress_bar=False,
                   log_every_n_steps=log_interval, callbacks=[checkpoint, progress, TrainingState()],
                   logger=TensorBoardLogger(save_dir=str(directory), name="", version=""))
    options.update(trainer_options)
    trainer = L.Trainer(**options)
    started = time.perf_counter()
    write_json(directory / "training_status.json", {
        "status": "running", "pid": os.getpid(), "max_epochs": trainer.max_epochs,
        "max_steps": trainer.max_steps, "start_step": resume_info["step"] if resume_info else 0,
        "validation_guard_mode": guard.mode if guard else None,
    })
    try:
        trainer.fit(model, datamodule=data, ckpt_path=resume)
    except BaseException as error:
        write_json(directory / "training_status.json", {
            "status": "failed", "step": trainer.global_step, "error": str(error),
        })
        raise
    finally:
        data.teardown("fit")
    if checkpoint.best_result is None:
        raise RuntimeError("训练结束但没有验证结果")
    session_fit_seconds = time.perf_counter() - started
    fit_seconds = session_fit_seconds + (resume_info["prior_fit_seconds"] if resume_info else 0)
    if checkpoint.pause_reason is not None:
        # 暂停是可续训的中断；保留优化器/循环状态，不执行正常完成后的清理。
        status = {"status": "paused", "step": trainer.global_step,
                  "reasons": checkpoint.pause_reason, "fit_seconds": fit_seconds,
                  "session_fit_seconds": session_fit_seconds,
                  "resume_checkpoint": str(directory / "last.ckpt")}
        write_json(directory / "training_status.json", status)
        result = dict(checkpoint.best_result, total_steps=trainer.global_step, **status)
        write_json(directory / "result.json", result)
        return result
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
    write_json(directory / "training_status.json", {
        "status": "completed", "step": trainer.global_step, "fit_seconds": fit_seconds,
    })
    return result


def main(model_class, model_dir, defaults):
    parser = argparse.ArgumentParser(description=f"{model_class.__name__} Training")
    parser.add_argument("--config", "-c", type=Path, default=model_dir.parent / "base_sort_conf.yaml")
    parser.add_argument("--extra_config", "-e", type=Path, nargs="+", default=[])
    parser.add_argument("--output", type=Path, help="新训练目录；默认 experiments/<模型>_<时间>")
    parser.add_argument("--resume", type=Path, help="从完整 last.ckpt 续训；可指定新输出目录保留原运行")
    args = parser.parse_args()
    config = OmegaConf.merge(*(OmegaConf.load(path) for path in [args.config, *defaults, *args.extra_config]))
    directory = args.output or (args.resume.parent if args.resume else ROOT / "experiments" / (
        config.name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    result = fit(model_class, config, directory, resume=args.resume)
    print(json.dumps(result["metrics"]["overall"], ensure_ascii=False, indent=2))
