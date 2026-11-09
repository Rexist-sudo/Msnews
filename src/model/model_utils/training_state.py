"""排序续训的采样确认与随机数状态保存，配合 Lightning 的优化器/循环恢复。"""

import random

import lightning as L
import numpy as np
import torch

from ...dataset.DataReader.pl_dataloader import ResumableDataLoader


class TrainingState(L.Callback):
    """每 batch 确认已训练的样本数，并为后续 checkpoint 保存随机数状态。"""

    def __init__(self) -> None:
        self.pending_rng = None

    def on_train_batch_end(self, trainer: L.Trainer, model: L.LightningModule,
                           outputs: object, batch: object, batch_idx: int) -> None:
        loader = trainer.train_dataloader
        if isinstance(loader, ResumableDataLoader):
            loader.mark_completed(trainer.current_epoch, batch_idx + 1)

    def on_save_checkpoint(self, trainer: L.Trainer, model: L.LightningModule, checkpoint: dict) -> None:
        checkpoint["ranking_rng_state"] = {
            "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if trainer.strategy.root_device.type == "cuda" else None,
        }

    def on_load_checkpoint(self, trainer: L.Trainer, model: L.LightningModule, checkpoint: dict) -> None:
        self.pending_rng = checkpoint.get("ranking_rng_state")

    def on_train_start(self, trainer: L.Trainer, model: L.LightningModule) -> None:
        # 在模型初始化、loader 迭代器创建之后恢复，避免初始化消耗已保存的随机流。
        if self.pending_rng is not None:
            state = self.pending_rng
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            if state["cuda"] is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
            self.pending_rng = None
