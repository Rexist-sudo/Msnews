"""汇总当前 M3 与既有 DIN/M1/M2 指标，完成同设备的 M2/M3 推理成本比较。

先运行公共 evaluate --verify --output 得到本次 evaluation.json，再调用本入口。
仅接受已完成或按验证规则暂停的运行；不启动、恢复或修改训练。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.dataset.DataReader.data_reader_mmap import DataReaderMmap
from src.model.model_utils.checkpoint import load_model
from src.model.model_utils.evaluation import benchmark
from src.model.model_utils.ranking_metrics import METRIC_NAMES
from src.model.model_utils.training import write_json
from src.model.sort_models.classic_sort_models.rankmixer.model import RankMixerDIN


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/ranking_study"


def read_json(path: Path) -> dict | list:
    """读取既有实验结果，不修改其内容。"""
    return json.loads(path.read_text(encoding="utf-8"))


def collect_results(run: Path) -> tuple[dict, dict]:
    """核对最佳点、预测行顺序和分桶覆盖后，返回五个模型结果及 M3−M2 差值。"""
    status = read_json(run / "training_status.json")
    if status["status"] not in ("paused", "completed"):
        raise ValueError(f"M3 尚未终止，不能进行独立成本比较: {status}")
    history = read_json(run / "history.json")
    config = OmegaConf.load(run / "config.yaml")
    manifest = read_json(ROOT / config.paths.out_basedir / "extractored_feature/train_features_mmap/manifest.json")
    rows = sum(part["rows"] for part in manifest["parts"])
    batches = (rows + int(config.dataset.batch_size) - 1) // int(config.dataset.batch_size)
    expected_steps = [index // 2 * batches + (batches if index % 2 else max(1, batches // 2))
                      for index in range(len(history))]
    if [row["step"] for row in history] != expected_steps:
        raise ValueError("M3 验证历史未按计划的半轮时点连续记录")
    result = read_json(run / "result.json")
    best = max(history, key=lambda row: (row["overall"]["GAUC"], -row["overall"]["LogLoss"]))
    if result["best_step"] != best["step"]:
        raise ValueError("M3 最佳权重记录与 GAUC/LogLoss 选取规则不一致")
    metrics = read_json(run / "evaluation.json")
    for metric in METRIC_NAMES:
        if abs(metrics["overall"][metric] - best["overall"][metric]) > 1e-12:
            raise ValueError(f"M3 独立评估与所选验证点不一致: {metric}")
    baseline = read_json(STUDY / "m2_comparison.json")
    runs = {name: dict(values) for name, values in baseline["runs"].items()}
    for name, values in runs.items():
        path = ROOT / values["source"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != values["sha256"]:
            raise ValueError(f"{name} 已保存预测的来源指纹发生变化")
    predictions = run / "best_predictions.npz"
    with np.load(predictions) as current, np.load(ROOT / runs["M2_2"]["source"]) as previous:
        for name in ("user_id", "item_id", "label", "impression_id"):
            np.testing.assert_array_equal(current[name].reshape(-1), previous[name].reshape(-1))
    runs["M3"] = {
        "source": str(predictions.relative_to(ROOT)),
        "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "best_step": best["step"], "total_steps": result["total_steps"],
        "status": status, "fit_seconds": result["fit_seconds"], "metrics": metrics,
    }
    deltas = {}
    for bucket, values in metrics.items():
        reference = runs["M2_2"]["metrics"][bucket]
        for name in ("rows", "positive", "users_with_auc", "User_Count", "Impression_Count"):
            if values[name] != reference[name]:
                raise ValueError(f"{bucket} 的 {name} 与 M2 不一致")
        deltas[bucket] = {metric: values[metric] - reference[metric] for metric in METRIC_NAMES}
    return runs, deltas


def measure_costs(run: Path, device: str) -> dict:
    """同一 Dev 前 512 条候选，交替顺序测三轮；每轮预热 20 次、计时 100 次。"""
    checkpoints = {"M2_2": STUDY / "m2_smooth_seed42/best.ckpt", "M3": run / "best.ckpt"}
    costs = {name: {"rounds": []} for name in checkpoints}
    for round_index in range(3):
        order = list(checkpoints) if round_index % 2 == 0 else list(reversed(checkpoints))
        for name in order:
            model = load_model(RankMixerDIN, checkpoints[name]).to(device)
            data = DataReaderMmap(model.config, ROOT / "src/tmp/extractored_feature/dev_features_mmap")
            batch = data.get_batch(np.arange(512))
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            measured = benchmark(model, batch)
            if device.startswith("cuda"):
                measured["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            costs[name]["rounds"].append(measured)
            costs[name].update({
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
                "backbone_parameters": sum(p.numel() for key, p in model.named_parameters()
                                           if key.startswith("interaction.")),
                "parameters_by_module": {key: sum(p.numel() for p in module.parameters())
                                         for key, module in model.named_children()},
            })
            data.close()
            del model, data, batch
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    for values in costs.values():
        medians = [row["median_ms"] for row in values["rounds"]]
        values["median_ms"] = float(np.median(medians))
        values["median_range_ms"] = [min(medians), max(medians)]
        values["p95_ms"] = float(np.median([row["p95_ms"] for row in values["rounds"]]))
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="已结束的 M3 实验目录")
    parser.add_argument("--output", type=Path, required=True, help="新的比较 JSON，不覆盖既有文件")
    parser.add_argument("--device", default="cuda", help="计时设备，M2/M3 使用同一设备")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有比较结果: {args.output}")
    torch.set_num_threads(4)
    run = args.run.resolve()
    runs, deltas = collect_results(run)
    costs = measure_costs(run, args.device)
    for name in costs:
        runs[name]["trainable_parameters"] = costs[name]["trainable_parameters"]
    result = {
        "runs": runs, "M3_minus_M2_2": deltas, "inference": costs,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "torch": torch.__version__, "precision": "float32", "batch_selection": "Dev rows [0,512)",
        "measurement": "单模型独占设备，三轮交替 M2/M3 顺序；设备常驻输入前向，含同步，不含读取",
        "validation_history": read_json(run / "history.json"),
        "validation_guard": OmegaConf.to_container(OmegaConf.load(run / "config.yaml").validation_guard),
        "resume": read_json(run / "resume_info.json") if (run / "resume_info.json").exists() else None,
        "reference": (read_json(run / "validation_reference.json")
                      if (run / "validation_reference.json").exists() else None),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    for name, values in runs.items():
        print(name, json.dumps({key: values["metrics"]["overall"][key] for key in METRIC_NAMES}), flush=True)
    print("M3_minus_M2_2", json.dumps(deltas["overall"]), flush=True)
    print("costs", json.dumps(costs), flush=True)


if __name__ == "__main__":
    main()
