"""M3 最佳预测的历史切片、GAUC 配对归因及固定样本的逐层机制统计。"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .model import HISTORY_FIELDS
from ....model_utils.ranking_metrics import METRIC_NAMES, calculate_metrics
from ....model_utils.ranking_attribution import support_pair_contributions
from .....dataset.FeaturesGenerator.history_analysis import history_group_masks
from .....dataset.FeaturesGenerator.ranking_support import load_support
from .....dataset.DataReader.data_reader_mmap import DataReaderMmap


def _distribution(values: list) -> dict:
    """汇总已采样数值的分布；空组显式保留样本数零。"""
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(values.mean()) if len(values) else 0.0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 0.0}


@torch.no_grad()
def mechanism_diagnostics(model, reader, batches: int = 8, batch_size: int = 512) -> dict:
    """均匀选取连续候选窗口，统计历史/兴趣残差和同请求候选间表示差异。

    保留连续候选以覆盖同一请求的不同候选；诊断为固定样本统计，不作为
    全量评估。每路注意力熵按实际有效历史归一化，空历史单独计数。
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    values = [{}, {}]
    rows, empty = 0, 0
    starts = np.unique(np.linspace(0, max(0, len(reader) - batch_size), batches, dtype=int))
    try:
        for start in starts:
            batch = {name: value.to(device) for name, value in
                     reader.get_batch(np.arange(start, min(start + batch_size, len(reader)))).items()}
            _, trace = model.forward_with_trace(batch)
            mask = trace["mask"]
            rows += len(mask)
            empty += int((~mask.any(dim=1)).sum())
            previous_history = trace["initial_history"]
            previous_attention = None
            # 同一原始请求中的相邻不同候选。序列输入相同，差异来自候选条件。
            same_request = (batch["impression_id"].reshape(-1)[1:] == batch["impression_id"].reshape(-1)[:-1])
            same_request &= batch["user_id"].reshape(-1)[1:] == batch["user_id"].reshape(-1)[:-1]
            same_request &= batch["item_id"].reshape(-1)[1:] != batch["item_id"].reshape(-1)[:-1]
            has_history = mask.any(dim=1)
            same_request &= has_history[1:] & has_history[:-1]
            for number, layer in enumerate(trace["layers"]):
                current = values[number]

                def record(name, tensor):
                    current.setdefault(name, []).extend(tensor.detach().float().cpu().reshape(-1).tolist())

                record("history_residual_l2", (layer["history"] - previous_history).flatten(1).norm(dim=1))
                record("candidate_history_difference_l2",
                       (layer["history"][1:] - layer["history"][:-1]).flatten(1).norm(dim=1)[same_request])
                for route, (name, delta, attention) in enumerate(zip(HISTORY_FIELDS, layer["interest_deltas"], layer["attention"])):
                    record(name + "_interest_delta_l2", delta.norm(dim=1))
                    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=1)
                    normalizer = mask.sum(dim=1).clamp_min(2).float().log()
                    record(name + "_normalized_entropy", (entropy / normalizer)[has_history])
                    record(name + "_candidate_attention_l1", (attention[1:] - attention[:-1]).abs().sum(dim=1)[same_request])
                    if previous_attention is not None:
                        record(name + "_attention_change_l1", (attention - previous_attention[route]).abs().sum(dim=1)[has_history])
                    item_name = model.array_item_feature_pairs[name]
                    matches = (batch[name] == batch[item_name].reshape(-1, 1)) & mask
                    record(name + "_matching_attention_mass", (attention * matches).sum(dim=1)[has_history])
                previous_history = layer["history"]
                previous_attention = layer["attention"]
    finally:
        model.train(was_training)
    return {"scope": "固定连续候选窗口抽样；候选间差异仅统计同请求、不同新闻且非空历史",
            "rows": rows, "empty_history_rows": empty, "batch_size": batch_size,
            "window_starts": starts.tolist(),
            "layers": [{name: _distribution(samples) for name, samples in layer.items()} for layer in values]}


def analyze_predictions(reader, predictions, support, reference=None) -> dict:
    """使用公共八指标和 GAUC 归因器评估全量历史切片，可对齐 M2_2 预测。"""
    names = ("user_id", "item_id", "label", "score", "impression_id")
    user, item, label, score, impression = [np.asarray(predictions[name]).reshape(-1) for name in names]
    if len(user) != len(reader) or reader.selected_indices is not None:
        raise ValueError("M3 全量分析要求完整验证集，与 dataset.only_test_warm_up_user=false 一致")
    arrays = reader._ensure_arrays()
    for name, value in (("user_id", user), ("item_id", item), ("label", label), ("impression_id", impression)):
        np.testing.assert_array_equal(value, arrays[name].reshape(-1))
    masks = history_group_masks(reader)
    exposure = support["candidate"][item]
    bucket = np.select([exposure == 0, exposure < 10, exposure < 100], [0, 1, 2], default=3)
    pairs, row_contributions = support_pair_contributions(user, label, score, bucket)
    slices = {}
    reference_slices = {}
    if reference is not None:
        for name in ("user_id", "item_id", "label", "impression_id"):
            np.testing.assert_array_equal(np.asarray(reference[name]).reshape(-1), predictions[name].reshape(-1))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, mask in masks.items():
            metrics = calculate_metrics(user[mask], label[mask], score[mask], impression[mask], pool)
            metrics["row_coverage"] = float(mask.mean())
            metrics["positive_coverage"] = float(label[mask].sum() / max(1, label.sum()))
            slices[name] = metrics
            if reference is not None:
                baseline_metrics = calculate_metrics(user[mask], label[mask], reference["score"][mask], impression[mask], pool)
                reference_slices[name] = {"reference": baseline_metrics,
                                          "delta": {key: metrics[key] - baseline_metrics[key] for key in METRIC_NAMES}}
    # 联合切片采用正样本归属的 GAUC 贡献，保留每个用户全量候选配对分母。
    joint = {}
    for history_name in ("history_empty", "history_1_9", "history_10_19", "history_20_50"):
        for field in ("category", "subcategory"):
            for match in ("recent_only", "earlier_only", "both", "neither"):
                mask = masks[history_name] & masks[f"{field}_{match}"]
                for number, title in enumerate(("0", "1_9", "10_99", "100_plus")):
                    keep = mask & (bucket == number)
                    joint[f"{history_name}/{field}_{match}/exposure_{title}"] = {
                        "rows": int(keep.sum()), "positive": int(label[keep].sum()),
                        **{name: float(values[keep].sum()) for name, values in row_contributions.items()},
                    }
    result = {"history_slices": slices, "gauc_support_pairs": pairs, "joint_positive_contributions": joint}
    if reference is not None:
        baseline, _ = support_pair_contributions(user, label, reference["score"], bucket)
        result["history_slice_comparison"] = reference_slices
        result["m2_gauc_support_pairs"] = baseline
        result["gauc_delta"] = {name: pairs[name] - baseline[name] for name in ("overall", "same_bucket", "cross_bucket")}
    return result


def analyze_experiment(model, reader, predictions, support) -> dict:
    """供公共训练及独立评估调用；参考预测不存在时仍可从零完成 M3 评估。"""
    reference = Path(model.config.analysis_cfg.reference_predictions)
    if reference.exists():
        with np.load(reference) as saved:
            result = analyze_predictions(reader, predictions, support, reference=saved)
            with ProcessPoolExecutor(max_workers=4) as pool:
                overall = calculate_metrics(predictions["user_id"], predictions["label"], predictions["score"], predictions["impression_id"], pool)
                baseline = calculate_metrics(saved["user_id"], saved["label"], saved["score"], saved["impression_id"], pool)
            result["m3_minus_m2_overall"] = {name: overall[name] - baseline[name] for name in METRIC_NAMES}
    else:
        result = analyze_predictions(reader, predictions, support)
    result["reference_predictions"] = {"path": str(reference), "available": reference.exists()}
    result["mechanism"] = mechanism_diagnostics(model, reader)
    return result


def main() -> None:
    """训练前直接分析 M1/M2_2 已有预测，保留原始请求及全量配对分母。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("src/model/sort_models/classic_sort_models/base_sort_conf.yaml"))
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有分析: {args.output}")
    config = OmegaConf.load(args.config)
    reader = DataReaderMmap(config, Path(config.paths.out_basedir) / "extractored_feature/dev_features_mmap")
    try:
        support = load_support(config)
        result = {}
        for path in args.predictions:
            print(f"分析 {path.parent.name} 的历史切片与曝光配对贡献…", flush=True)
            with np.load(path) as saved:
                predictions = dict(saved)
                legacy = set(predictions) == {"score"}
                if legacy:
                    # 旧 M1 只保存 score；沿其缓存行序补齐标识，并明确记录依据。
                    if len(predictions["score"]) != len(reader):
                        raise ValueError(f"旧预测行数与完整验证缓存不一致: {path}")
                    arrays = reader._ensure_arrays()
                    predictions.update({name: arrays[name].reshape(-1)
                                        for name in ("user_id", "item_id", "label", "impression_id")})
                result[str(path)] = analyze_predictions(reader, predictions, support)
                result[str(path)]["alignment"] = ("旧 M1 score-only 按现有完整 Dev 缓存行序对齐"
                                                   if legacy else "逐项核对 user/item/label/impression 标识")
            print(f"完成 {path.parent.name}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
