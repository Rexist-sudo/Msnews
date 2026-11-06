"""排序模型共用验证入口，指标计算放在 model_utils。"""

from pathlib import Path

import numpy as np

from .base_model import BaseModel
from ..model_utils.ranking_metrics import METRIC_NAMES, _compute_user_metrics, evaluate
from ...Logger.logging import Logger

logger = Logger.get_logger("BaseModelSort")


class BaseModelSort(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self._validation_batches = []
        self.validation_metrics = {}
        self.validation_predictions = {}
        self.support = None

    def inference(self, batch):
        raise NotImplementedError("Subclasses must implement inference()")

    def on_validation_epoch_start(self):
        self._validation_batches.clear()
        self.validation_predictions = {}

    def validation_step(self, batch, batch_idx):
        if "impression_id" not in batch:
            raise ValueError("HR、NDCG、MRR 评估需要每条候选的 impression_id")
        scores = self.inference(batch).detach().reshape(-1).cpu().numpy()
        row = {name: batch[name].detach().reshape(-1).cpu().numpy()
               for name in ("user_id", "impression_id", "label")}
        item = next((batch[name] for name in ("item_id", "news_id", "movie_id", "doc_id", "mid")
                     if name in batch), None)
        row["item_id"] = item.detach().reshape(-1).cpu().numpy() if item is not None else np.zeros(len(scores), dtype=int)
        row["score"] = scores
        self._validation_batches.append(row)

    def on_validation_epoch_end(self):
        names = ("user_id", "item_id", "label", "score", "impression_id")
        data = {name: np.concatenate([batch[name] for batch in self._validation_batches])
                if self._validation_batches else np.array([], dtype=np.int64) for name in names}
        self._validation_batches.clear()
        self.validation_predictions = data
        train_users = getattr(self, "user_in_train_set", set())
        # 训练中按整体 GAUC 选权重，最佳预测的完整分桶由训练入口统一计算。
        selecting = getattr(self, "_checkpoint_managed_by_trainer", False)
        self.validation_metrics = evaluate(*(data[name] for name in names),
                                           support=None if selecting else self.support,
                                           train_users=None if selecting else train_users)
        results = {title: self.validation_metrics[key] for title, key in
                   (("Overall", "overall"), ("Warm_Start", "seen_uid"), ("Cold_Start", "unseen_uid"))
                   if key in self.validation_metrics}
        lines = [f"Epoch {self.current_epoch}, step {self.global_step} Validation Results:"]
        for name, values in results.items():
            lines.append(name + ": " + ", ".join(f"{key}={values[key]:.4f}" for key in METRIC_NAMES))
        message = "\n".join(lines) + "\n"
        logger.info(message)
        if hasattr(self, "log_dir"):
            with (Path(self.log_dir) / "val_log.log").open("a", encoding="utf-8") as stream:
                stream.write(message)
        if self._trainer is not None:
            self.log_dict({"val_" + key: results["Overall"][key] for key in METRIC_NAMES}, logger=True)
        self.val_logloss_history.append((self.global_step, results["Overall"]["LogLoss"]))
        self._plot_loss_curve()
        return results
