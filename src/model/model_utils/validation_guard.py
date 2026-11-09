"""比较每次验证的八项指标，在全面退步时请求暂停训练。"""

import math

from .ranking_metrics import METRIC_NAMES


class ValidationGuard:
    """比较固定基线及上次验证，或仅比较本次运行上次验证。

    LogLoss 越低越好，其余七项指标越高越好；只有八项都严格变差
    才触发暂停，相等或任一改善均不算全面退步。第一轮验证没有自身参照。
    previous_validation 模式不要求基线覆盖训练预算，支持无限轮数。
    """

    def __init__(self, reference_history: list[dict] | None = None, mode: str = "baseline_and_previous") -> None:
        if mode not in ("baseline_and_previous", "previous_validation"):
            raise ValueError(f"未知 validation_guard.mode: {mode}")
        self.mode = mode
        if mode == "baseline_and_previous" and not reference_history:
            raise ValueError("validation_guard 的 reference_history 不得为空")
        self.references = {}
        for row in reference_history or []:
            self._validate_metrics(row["overall"])
            key = (int(row["epoch"]), int(row["step"]))
            if key in self.references:
                raise ValueError(f"validation_guard 基线验证点重复: {key}")
            self.references[key] = row

    @staticmethod
    def _validate_metrics(metrics: dict) -> None:
        for name in METRIC_NAMES:
            if name not in metrics or not math.isfinite(float(metrics[name])):
                raise ValueError(f"validation_guard 要求有限指标 {name}，实际 {metrics.get(name)}")

    def validate_schedule(self, steps_per_epoch: int, max_epochs: int) -> None:
        """训练前核对半轮/轮末验证点，防止不同训练预算错位比较。"""
        if self.mode == "previous_validation":
            return
        if max_epochs < 0:
            raise ValueError("无限轮数只能使用 previous_validation 比较模式")
        expected = {
            (epoch, epoch * steps_per_epoch + offset)
            for epoch in range(max_epochs)
            for offset in (max(1, steps_per_epoch // 2), steps_per_epoch)
        }
        missing = expected - self.references.keys()
        if missing:
            raise ValueError(f"validation_guard 基线缺少对应验证点: {sorted(missing)}")

    def _compare(self, current: dict, reference: dict) -> dict:
        self._validate_metrics(reference["overall"])
        # 验证候选与分组规模必须一致；覆盖率不是优化指标，不参与八项判断。
        for name in ("rows", "positive", "users_with_auc", "User_Count", "Impression_Count"):
            if current["overall"].get(name) != reference["overall"].get(name):
                raise ValueError(f"validation_guard 验证规模不一致: {name}")
        improvements = {
            name: (float(current["overall"][name]) - float(reference["overall"][name]))
            * (-1 if name == "LogLoss" else 1)
            for name in METRIC_NAMES
        }
        worse = [name for name, value in improvements.items() if value < 0]
        return {
            "epoch": reference["epoch"], "step": reference["step"],
            "overall": reference["overall"],
            "improvement": improvements, "worse_metrics": worse,
            "all_worse": len(worse) == len(METRIC_NAMES),
        }

    def check(self, current: dict, previous: dict | None) -> dict:
        """返回参照指标、按好坏方向统一的差值和暂停原因；正差值表示改善。"""
        self._validate_metrics(current["overall"])
        key = (int(current["epoch"]), int(current["step"]))
        comparisons = {}
        if self.mode == "baseline_and_previous":
            if key not in self.references:
                raise ValueError(f"validation_guard 基线缺少同 epoch/step 验证点: {key}")
            comparisons["baseline_same_step"] = self._compare(current, self.references[key])
        if previous is not None:
            comparisons["previous_validation"] = self._compare(current, previous)
        reasons = [name for name, result in comparisons.items() if result["all_worse"]]
        return {"references": comparisons, "pause": bool(reasons), "reasons": reasons}
