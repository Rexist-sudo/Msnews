"""排序模型的批量预测、推理计时和独立评估入口。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .checkpoint import load_model
from .ranking_metrics import evaluate
from ...dataset.DataReader.pl_dataloader import MINDDataModule
from ...dataset.FeaturesGenerator.ranking_support import load_support


@torch.no_grad()
def predict(model, data, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    scores = np.empty(len(data), dtype=np.float32)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        batch = (data.get_batch(np.arange(start, stop)) if hasattr(data, "get_batch")
                 else default_collate([data[i] for i in range(start, stop)]))
        batch = {name: value.to(device) for name, value in batch.items()}
        scores[start:stop] = model.inference(batch).reshape(-1).cpu().numpy()
    return scores


@torch.no_grad()
def benchmark(model, batch, repeats=100):
    model.eval()
    device = next(model.parameters()).device
    batch = {name: value.to(device) for name, value in batch.items()}
    for _ in range(20):
        model.inference(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.inference(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return {"batch_size": len(batch["label"]), "warmup": 20, "repeats": repeats,
            "median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "scope": "设备常驻输入的前向耗时，包含同步，不含数据读取"}


def main(model_class):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verify", action="store_true", help="与同目录已保存的预测逐行比较，不覆盖历史结果")
    parser.add_argument("--output", type=Path, help="可选：将当前口径的指标保存到新文件")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"不覆盖已有结果: {args.output}")
    torch.set_num_threads(4)
    model = load_model(model_class, args.checkpoint).to(args.device)
    data_module = MINDDataModule(model.config)
    data_module.setup("validate")
    data = data_module.val_dataset
    scores = predict(model, data)
    if args.verify:
        with np.load(args.checkpoint.parent / "best_predictions.npz") as saved:
            np.testing.assert_allclose(scores, saved["score"], rtol=1e-6, atol=1e-7)
            print(f"全量 {len(scores)} 条预测一致；最大误差 {np.max(np.abs(scores - saved['score'])):.3g}")
        if args.output is None:
            data_module.teardown("validate")
            return
    if hasattr(data, "_ensure_arrays"):
        arrays = data._ensure_arrays()
        rows = data.selected_indices if data.selected_indices is not None else slice(None)
    else:
        arrays = {name: np.empty(len(data), dtype=int) for name in ("user_id", "item_id", "label", "impression_id")}
        for i, row in enumerate(data):
            for name in arrays:
                arrays[name][i] = int(row[name])
        rows = slice(None)
    metrics = evaluate(arrays["user_id"][rows], arrays["item_id"][rows], arrays["label"][rows], scores,
                       arrays["impression_id"][rows], support=load_support(model.config))
    data_module.teardown("validate")
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
