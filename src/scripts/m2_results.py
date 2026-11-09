"""汇总 M2 对照、参数量及同设备前向成本。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset.DataReader.data_reader_mmap import DataReaderMmap
from src.model.model_utils.checkpoint import load_model
from src.model.model_utils.evaluation import benchmark
from src.model.model_utils.ranking_metrics import METRIC_NAMES
from src.model.sort_models.classic_sort_models.rankmixer.model import RankMixerDIN


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/ranking_study"


def auc_pairs(label, score):
    positive, negative = score[label == 1], np.sort(score[label == 0])
    wins = (np.searchsorted(negative, positive, side="left").sum()
            + np.searchsorted(negative, positive, side="right").sum()) / 2
    return wins, len(positive) * len(negative)


def add_pair_attribution():
    """在相同用户和分母下拆分同曝光桶、跨曝光桶对整体 GAUC 的贡献。"""
    # 与直接枚举核对同分、全对、全错及缺少一类标签的情况。
    for label, score in (([1, 0, 1, 0], [.5, .5, .9, .1]), ([1, 0], [.1, .9]),
                         ([1, 0], [.9, .1]), ([0, 0], [.1, .9])):
        label, score = np.array(label), np.array(score)
        direct = [(float(a > b) + .5 * float(a == b)) for a in score[label == 1] for b in score[label == 0]]
        assert auc_pairs(label, score) == (sum(direct), len(direct))
    path = OUTPUT / "m2_comparison.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    with np.load(ROOT / result["runs"]["M2_2"]["source"]) as saved:
        user, item, label = [saved[name].reshape(-1) for name in ("user_id", "item_id", "label")]
    directory = ROOT / "src/tmp/extractored_feature/train_features_mmap"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    train_item = np.load(directory / manifest["parts"][0]["path"] / "item_id.npy", mmap_mode="r").reshape(-1)
    exposure = np.bincount(train_item, minlength=int(item.max()) + 1)[item]
    bucket = np.searchsorted([1, 10, 100], exposure, side="right")
    order = np.argsort(user, kind="stable")
    user, label, bucket = user[order], label[order], bucket[order]
    bounds = np.r_[0, np.flatnonzero(user[1:] != user[:-1]) + 1, len(user)]
    attribution = {"scope": "逐用户按全部正负对归一化，同桶和跨桶贡献之和等于整体 GAUC；曝光桶为 0、1–9、10–99、≥100", "runs": {}}
    for name in ("M1", "M2_1", "M2_2"):
        with np.load(ROOT / result["runs"][name]["source"]) as saved:
            score = saved["score"].reshape(-1)[order]
        contributions = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            y, s, b = label[start:stop], score[start:stop], bucket[start:stop]
            total_wins, total_pairs = auc_pairs(y, s)
            if not total_pairs:
                continue
            within_wins, within_pairs = np.sum([auc_pairs(y[b == k], s[b == k]) for k in range(4)], axis=0)
            contributions.append((within_wins / total_pairs, (total_wins - within_wins) / total_pairs,
                                  within_pairs / total_pairs))
        within, cross, fraction = np.mean(contributions, axis=0)
        expected = result["runs"][name]["metrics"]["overall"]
        assert len(contributions) == expected["users_with_auc"]
        assert abs(within + cross - expected["GAUC"]) < 1e-12
        attribution["runs"][name] = {"within_bucket": within, "cross_bucket": cross,
                                     "within_pair_fraction": fraction, "users": len(contributions)}
    result["pair_attribution"] = attribution
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("pair attribution", json.dumps(attribution, ensure_ascii=False), flush=True)


def main():
    torch.set_num_threads(4)
    baseline = json.loads((OUTPUT / "m2_eda_metrics.json").read_text(encoding="utf-8"))
    runs = {"DIN": baseline["DIN"], "M1": baseline["M1"]}
    for name, directory in (("M2_1", "m2_raw_seed42"), ("M2_2", "m2_smooth_seed42")):
        result = json.loads((OUTPUT / directory / "result.json").read_text(encoding="utf-8"))
        history = json.loads((OUTPUT / directory / "history.json").read_text(encoding="utf-8"))
        assert result["total_steps"] == 34239
        assert [row["step"] for row in history] == [5706, 11413, 17119, 22826, 28532, 34239]
        best = max(history, key=lambda row: (row["overall"]["GAUC"], -row["overall"]["LogLoss"]))
        assert best["step"] == result["best_step"]
        for metric in METRIC_NAMES:
            assert abs(result["metrics"]["overall"][metric] - best["overall"][metric]) < 1e-12
        predictions = OUTPUT / directory / "best_predictions.npz"
        runs[name] = {"metrics": result["metrics"], "trainable_parameters": result["cost"]["trainable_parameters"],
                      "best_step": result["best_step"], "fit_seconds": result["fit_seconds"],
                      "source": str(predictions.relative_to(ROOT)), "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()}
    costs = {}
    for name, checkpoint in (("M1", "m1_d128_k2_seed42/best.pt"), ("M2_1", "m2_raw_seed42/best.ckpt"),
                             ("M2_2", "m2_smooth_seed42/best.ckpt")):
        model = load_model(RankMixerDIN, OUTPUT / checkpoint).to("cuda")
        data = DataReaderMmap(model.config, ROOT / "src/tmp/extractored_feature/dev_features_mmap")
        batch = data.get_batch(np.arange(512))
        if name != "M1":
            with np.load(OUTPUT / Path(checkpoint).parent / "best_predictions.npz") as saved:
                for field in ("user_id", "item_id", "label", "impression_id"):
                    np.testing.assert_array_equal(saved[field].reshape(-1), data._ensure_arrays()[field].reshape(-1))
        measured = benchmark(model, batch)
        measured["statistics_parameters"] = sum(p.numel() for n, p in model.named_parameters() if n.startswith("statistics_"))
        measured["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        measured["frozen_parameters"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        measured["total_parameters"] = sum(p.numel() for p in model.parameters())
        assert measured["trainable_parameters"] == runs[name]["trainable_parameters"]
        costs[name] = measured
        data.close()
        del model, batch, data
        torch.cuda.empty_cache()
    deltas = {}
    for name, before in (("M2_1", "M1"), ("M2_2", "M2_1"), ("M2_2", "M1")):
        deltas[name + "_minus_" + before] = {}
        for bucket, values in runs[name]["metrics"].items():
            other = runs[before]["metrics"][bucket]
            for field in ("rows", "positive", "users_with_auc", "Impression_Count"):
                assert values[field] == other[field]
            deltas[name + "_minus_" + before][bucket] = {metric: values[metric] - other[metric] for metric in METRIC_NAMES}
    result = {"runs": runs, "deltas": deltas, "inference": costs, "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "statistics_cache": json.loads((ROOT / "src/tmp/extractored_feature/news_statistics.json").read_text(encoding="utf-8"))}
    (OUTPUT / "m2_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, data in runs.items():
        print(name, json.dumps(data["metrics"]["overall"], ensure_ascii=False), flush=True)
    print("inference", json.dumps(costs, ensure_ascii=False), flush=True)
    add_pair_attribution()


if __name__ == "__main__":
    main()
