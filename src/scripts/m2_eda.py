"""M2 前置 EDA；在项目根目录运行 python -m src.scripts.m2_eda。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.dataset.FeaturesGenerator.news_statistics import NewsStatistics
from src.dataset.FeaturesGenerator.ranking_support import load_support
from src.model.model_utils.ranking_metrics import evaluate


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data/MIND"
CACHE = ROOT / "src/tmp"
OUTPUT = ROOT / "experiments/ranking_study"
BUCKETS = ("0", "1-9", "10-99", ">=100")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding="utf-8")


def mmap(split, field):
    return np.load(CACHE / f"extractored_feature/{split}_features_mmap/part_000001/{field}.npy", mmap_mode="r")


def bucket(exposure):
    return np.searchsorted([1, 10, 100], exposure, side="right")


def sources():
    paths = [DATA / f"MINDsmall_{split}/behaviors.tsv" for split in ("train", "dev")]
    paths += [CACHE / f"preprocess/{name}" for name in ("news_id_map.json", "user_id_map.json", "all_news_preprocess.csv")]
    paths += [Path(__file__), ROOT / "src/dataset/FeaturesGenerator/news_statistics.py",
              ROOT / "src/model/model_utils/ranking_metrics.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def load_split(split, news_map, user_map):
    frame = pd.read_csv(DATA / f"MINDsmall_{split}/behaviors.tsv", sep="\t",
                        names=["impression", "user", "time", "history", "candidates"], keep_default_na=False)
    frame["time"] = pd.to_datetime(frame.time, format="%m/%d/%Y %I:%M:%S %p")
    frame = frame.sort_values("time").reset_index(drop=True)
    request_time = frame.time.astype("int64").to_numpy() // 10**9
    lengths = frame.candidates.str.count(" ").to_numpy() + 1
    size = int(lengths.sum())
    item, label = np.empty(size, np.int64), np.empty(size, np.int8)
    history = np.zeros((len(frame), 50), dtype=np.int64)
    cursor = 0
    for index, row in enumerate(frame.itertuples(index=False)):
        candidates = row.candidates.split()
        stop = cursor + len(candidates)
        item[cursor:stop] = [news_map[token[:-2]] for token in candidates]
        label[cursor:stop] = [int(token[-1]) for token in candidates]
        values = [news_map[value] for value in row.history.split()[-50:]]
        history[index, :len(values)] = values
        cursor = stop
    user = np.repeat(frame.user.map(user_map).to_numpy(), lengths)
    impression = np.repeat(frame.impression.to_numpy(), lengths)
    for name, values in (("item_id", item), ("label", label), ("user_id", user), ("impression_id", impression)):
        np.testing.assert_array_equal(values, mmap(split, name).reshape(-1))
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    np.testing.assert_array_equal(history, np.maximum(mmap(split, "user_history")[starts], 0))
    print(f"{split}: {len(frame):,} requests, {size:,} candidates, raw/mmap aligned", flush=True)
    return dict(item=item, label=label, time=np.repeat(request_time, lengths), user=user, impression=impression,
                request_time=request_time, history=history, request_id=frame.impression.to_numpy())


def features_by_chunk(statistics, item, timestamp, label=None):
    counts, clicks = np.zeros(26, np.int64), np.zeros(26, np.int64)
    for start in range(0, len(item), 200000):
        stop = start + 200000
        raw, smooth, state = statistics.features(item[start:stop], timestamp[start:stop])
        assert np.isfinite(raw).all() and np.isfinite(smooth).all()
        assert np.array_equal(state == 0, raw[:, 3] == 0)
        assert np.all((state >= 0) & (state <= 25))
        counts += np.bincount(state, minlength=26)
        if label is not None:
            clicks += np.bincount(state, weights=label[start:stop], minlength=26).astype(np.int64)
    return {"count": counts, "click": clicks, "occupied": int(np.count_nonzero(counts)),
            "unknown_fraction": float(counts[0] / counts.sum()), "max_known_cell_fraction": float(counts[1:].max() / counts[1:].sum())}


def calibration(statistics, split, cutoff):
    select = split["time"] >= cutoff
    values = statistics.query(split["item"][select], cutoff)
    label = split["label"][select]
    predictions = {"raw": np.where(values["exposure"] > 0, values["raw_ctr"], values["group_ctr"]),
                   "smooth": values["smooth_ctr"], "group": values["group_ctr"], "global": values["global_ctr"]}
    results = {}
    for name, pred in predictions.items():
        results[name] = {}
        for tag, mask in {"all": np.ones(len(label), bool), **{b: bucket(values["exposure"]) == i for i, b in enumerate(BUCKETS)}}.items():
            p, y = np.clip(pred[mask], 1e-6, 1 - 1e-6), label[mask]
            results[name][tag] = {"rows": len(y), "brier": float(np.mean((p - y)**2)),
                                  "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    return results


def analysis():
    started = time.perf_counter()
    news_map = json.loads((CACHE / "preprocess/news_id_map.json").read_text(encoding="utf-8"))
    user_map = json.loads((CACHE / "preprocess/user_id_map.json").read_text(encoding="utf-8"))
    train, dev = load_split("train", news_map, user_map), load_split("dev", news_map, user_map)
    assert train["time"].max() < dev["time"].min()
    news = pd.read_csv(CACHE / "preprocess/all_news_preprocess.csv", sep="\t", header=None, usecols=[0, 1, 2],
                       names=["item_id", "category", "subcategory"]).set_index("item_id").sort_index()
    size = len(news_map) + 1
    assert news.index.tolist() == list(range(1, size))
    group_code, groups = pd.factorize(pd.MultiIndex.from_frame(news[["category", "subcategory"]]), sort=True)
    group = np.r_[0, group_code + 1]
    statistics = NewsStatistics(train["item"], train["time"], train["label"], train["request_time"], group)
    statistics.fit_boundaries()
    end = statistics.query(np.arange(size), statistics.end)
    exposure, click = end["exposure"], end["click"]
    historical = np.bincount(train["history"][train["history"] > 0], minlength=size)
    dev_exposure = np.bincount(dev["item"], minlength=size)
    dev_click = np.bincount(dev["item"], weights=dev["label"], minlength=size).astype(np.int64)
    config = OmegaConf.load(ROOT / "src/model/sort_models/classic_sort_models/base_sort_conf.yaml")
    support = load_support(config)
    np.testing.assert_array_equal(exposure, support["candidate"])
    np.testing.assert_array_equal(exposure + historical, support["item"])
    np.testing.assert_array_equal(click, np.bincount(train["item"], weights=train["label"], minlength=size))
    sentinel = np.iinfo(np.int64).max
    first_candidate, first_history = np.full(size, sentinel), np.full(size, sentinel)
    np.minimum.at(first_candidate, train["item"], train["time"])
    valid = train["history"] > 0
    np.minimum.at(first_history, train["history"][valid], np.broadcast_to(train["request_time"][:, None], valid.shape)[valid])
    first_observed = np.minimum(first_candidate, first_history)
    news["raw_id"] = pd.Series({value: key for key, value in news_map.items()})
    for name, values in dict(exposure=exposure, click=click, history_support=historical, dev_exposure=dev_exposure,
                             dev_click=dev_click, epi=end["epi"], raw_ctr=end["raw_ctr"], smooth_ctr=end["smooth_ctr"],
                             group_exposure=end["group_exposure"]).items():
        news[name] = values[1:]
    news["first_candidate"] = pd.to_datetime(np.where(first_candidate[1:] == sentinel, np.nan, first_candidate[1:]), unit="s")
    news["first_observed"] = pd.to_datetime(np.where(first_observed[1:] == sentinel, np.nan, first_observed[1:]), unit="s")
    news.to_csv(OUTPUT / "m2_eda_news.csv", encoding="utf-8", float_format="%.10g")
    buckets = {}
    last_day = int(train["time"].max() // 86400 * 86400)
    for index, name in enumerate(BUCKETS):
        mask = bucket(exposure) == index
        mask[0] = False
        e, c = exposure[mask], click[mask]
        n, total = int(mask.sum()), int(dev_exposure[mask].sum())
        buckets[name] = {"news": n, "train_rows": int(e.sum()), "dev_rows": total, "dev_positive": int(dev_click[mask].sum()),
                         "train_share": float(e.sum() / len(train["item"])), "dev_share": total / len(dev["item"]),
                         "zero_click_news": int((c == 0).sum()), "zero_click_dev_share": float(dev_exposure[mask & (click == 0)].sum() / total),
                         "raw_ctr_zero_fraction": float(np.mean(c == 0)) if index else None,
                         "raw_ctr_one_fraction": float(np.mean(c == e)) if index else None,
                         "raw_ctr_std": float(np.std(end["raw_ctr"][mask])) if index else None,
                         "smooth_ctr_std": float(np.std(end["smooth_ctr"][mask])),
                         "first_candidate_last_day_dev_share": float(dev_exposure[mask & (first_candidate >= last_day) & (first_candidate < sentinel)].sum() / total),
                         "first_observed_last_day_dev_share": float(dev_exposure[mask & (first_observed >= last_day) & (first_observed < sentinel)].sum() / total),
                         "group_ge100_dev_share": float(dev_exposure[mask & (end["group_exposure"] >= 100)].sum() / total)}
    support_groups = {}
    for name, mask in {"unseen_item": (exposure + historical) == 0, "history_only": (exposure == 0) & (historical > 0),
                       "candidate_seen": exposure > 0}.items():
        mask[0] = False
        support_groups[name] = {"news": int(mask.sum()), "dev_rows": int(dev_exposure[mask].sum()),
                                "dev_share": float(dev_exposure[mask].sum() / len(dev["item"]))}
    coverage = []
    for level, columns in (("category", ["category"]), ("subcategory", ["category", "subcategory"])):
        grouped = news.groupby(columns, sort=True)
        for key, data in grouped:
            name = key if isinstance(key, str) else "/".join(key)
            coverage.append(dict(level=level, name=name, news=len(data), exposed_news=int((data.exposure > 0).sum()),
                                 clicked_news=int((data.click > 0).sum()), exposure=int(data.exposure.sum()),
                                 click=int(data.click.sum()), dev_exposure=int(data.dev_exposure.sum())))
    pd.DataFrame(coverage).to_csv(OUTPUT / "m2_eda_groups.csv", index=False, encoding="utf-8")
    states = {}
    for name, split in (("train", train), ("dev", dev)):
        states[name] = features_by_chunk(statistics, split["item"], split["time"], split["label"])
        valid = split["history"] > 0
        times = np.broadcast_to(split["request_time"][:, None], valid.shape)[valid]
        states[name + "_history"] = features_by_chunk(statistics, split["history"][valid], times)
        print(name, "causal features and states checked", flush=True)
    low = (exposure[dev["item"]] > 0) & (exposure[dev["item"]] < 10)
    dev_values = statistics.query(dev["item"], statistics.end)
    uniform_x = np.minimum((dev_values["epi"] * 5).astype(int), 4)
    uniform_y = np.minimum(((1 - dev_values["raw_ctr"]) * 5).astype(int), 4)
    uniform_state = np.where(dev_values["exposure"] > 0, 1 + 5 * uniform_x + uniform_y, 0)
    adaptive_state = statistics.features(dev["item"], statistics.end)[2]
    # 相同平滑响应下比较边界，再单独观察原始响应的均匀分桶。
    smooth_y = np.minimum(((1 - dev_values["smooth_ctr"]) * 5).astype(int), 4)
    uniform_smooth = np.where(dev_values["exposure"] > 0, 1 + 5 * uniform_x + smooth_y, 0)
    grids = {}
    for name, state in (("uniform_raw", uniform_state), ("uniform_smooth", uniform_smooth), ("adaptive_smooth", adaptive_state)):
        for subset, mask in (("all", np.ones(len(low), bool)), ("low", low)):
            counts = np.bincount(state[mask], minlength=26)
            clicks = np.bincount(state[mask], weights=dev["label"][mask], minlength=26).astype(np.int64)
            grids[name + "_" + subset] = {"count": counts, "click": clicks, "occupied": int(np.count_nonzero(counts[1:])),
                                           "max_cell_fraction": float(counts[1:].max() / counts[1:].sum())}
    hourly, changes = [], {name: [0.0, 0.0, 0] for name in BUCKETS[1:]}
    previous = None
    for hour_index, cutoff in enumerate(statistics.boundary_time):
        values = statistics.query(np.arange(size), cutoff)
        hour = statistics.boundary_time[max(0, hour_index - 1)]
        select = (train["time"] >= hour) & (train["time"] < cutoff)
        requests = (train["request_time"] >= hour) & (train["request_time"] < cutoff)
        hourly.append(dict(cutoff=str(pd.Timestamp(cutoff, unit="s")), requests=int(requests.sum()),
                           exposure=int(select.sum()), click=int(train["label"][select].sum()),
                           cumulative_exposure=int(values["exposure"].sum()), exposed_news=int((values["exposure"] > 0).sum()),
                           first_candidate_news=int(((first_candidate >= hour) & (first_candidate < cutoff)).sum()),
                           first_observed_news=int(((first_observed >= hour) & (first_observed < cutoff)).sum())))
        if previous is not None:
            updated = (values["exposure"] > previous["exposure"]) & (previous["exposure"] > 0)
            for index, name in enumerate(BUCKETS[1:], start=1):
                mask = updated & (bucket(values["exposure"]) == index)
                changes[name][0] += float(np.abs(values["raw_ctr"][mask] - previous["raw_ctr"][mask]).sum())
                changes[name][1] += float(np.abs(values["smooth_ctr"][mask] - previous["smooth_ctr"][mask]).sum())
                changes[name][2] += int(mask.sum())
        previous = values
    pd.DataFrame(hourly).to_csv(OUTPUT / "m2_eda_hourly.csv", index=False, encoding="utf-8")
    volatility = {name: {"updates": n, "raw_mean_abs_change": raw / n, "smooth_mean_abs_change": smooth / n}
                  for name, (raw, smooth, n) in changes.items()}
    assert sum(row["exposure"] for row in hourly) == len(train["item"])
    assert sum(row["click"] for row in hourly) == int(train["label"].sum())
    assert sum(row["requests"] for row in hourly) == len(train["request_time"])
    summary = dict(seed=42, sources=sources(), lambda_group=statistics.lambda_group, lambda_item=statistics.lambda_item,
                   boundary_rule="Hourly past-only news-weighted quantiles; Train-end frozen for Dev",
                   train={"requests": len(train["request_time"]), "rows": len(train["item"]), "clicks": int(train["label"].sum()),
                          "start": str(pd.Timestamp(statistics.origin, unit="s")), "end": str(pd.Timestamp(statistics.end - 1, unit="s"))},
                   dev={"requests": len(dev["request_time"]), "rows": len(dev["item"]), "clicks": int(dev["label"].sum()),
                        "start": str(pd.Timestamp(int(dev["time"].min()), unit="s")), "end": str(pd.Timestamp(int(dev["time"].max()), unit="s"))},
                   buckets=buckets, support=support_groups, states=states, grids=grids, volatility=volatility,
                   boundaries={"time": statistics.boundary_time, "epi": statistics.epi_edges, "av": statistics.av_edges},
                   calibration_train_last_day=calibration(statistics, train, last_day),
                   calibration_dev=calibration(statistics, dev, statistics.end),
                   seconds=time.perf_counter() - started)
    write_json(OUTPUT / "m2_eda_summary.json", summary)
    print(json.dumps({"buckets": buckets, "support": support_groups, "volatility": volatility}, indent=2), flush=True)
    make_plot_spec(summary, news)


def make_plot_spec(summary, news):
    known = news.exposure > 0
    grid = summary["grids"]
    style = {"font_size": 11, "axes_linewidth": 1.0, "font_family": ["Microsoft YaHei", "DejaVu Sans"]}
    panels = [dict(type="bar", title="候选曝光分布迁移", categories=list(BUCKETS), labels=["Train", "Dev"],
                   series=[[summary["buckets"][b][split + "_share"] * 100 for b in BUCKETS] for split in ["train", "dev"]],
                   colors=["#92B1D9", "#F6C8B6"], ylabel="候选占比 (%)", xlabel="Train 候选曝光 E", annotate=True, annotate_fmt="{:.2f}"),
              dict(type="scatter", title="已曝光新闻的机会与响应", x=np.log10(news.loc[known, "epi"]).tolist(),
                   y=(1 - news.loc[known, "raw_ctr"]).tolist(), xlabel="log10(EPI)", ylabel="Av = 1 - CTR",
                   color="#92B1D9", size=5, alpha=0.25)]
    for key, title in (("uniform_raw_low", "均匀网格 · 原始 Av"), ("adaptive_smooth_low", "分位网格 · 平滑 Av")):
        matrix = np.asarray(grid[key]["count"])[1:].reshape(5, 5).T / sum(grid[key]["count"]) * 100
        panels.append(dict(type="heatmap", title=title + "（低曝光 Dev）", matrix=matrix.tolist(), x_labels=["1", "2", "3", "4", "5"],
                           y_labels=["1", "2", "3", "4", "5"], xlabel="EPI 桶", ylabel="Av 桶", cmap="Blues",
                           annotate=True, annotate_fmt="{:.1f}", colorbar_label="候选占比 (%)"))
    write_json(CACHE / "m2_eda_figure.json", {"style": style, "layout": {"nrows": 2, "ncols": 2, "figsize": [12, 8], "tight_layout_pad": 2}, "panels": panels})
    click_panels = []
    for key, title in (("uniform_raw_all", "均匀网格 · 原始 Av"), ("adaptive_smooth_all", "分位网格 · 平滑 Av")):
        for measure, label in (("count", "候选数"), ("click", "点击数")):
            matrix = np.asarray(grid[key][measure])[1:].reshape(5, 5).T
            click_panels.append(dict(type="heatmap", title=title + " · " + label, matrix=matrix.tolist(), x_labels=list("12345"),
                                     y_labels=list("12345"), xlabel="EPI 桶", ylabel="Av 桶", cmap="Blues", annotate=True, annotate_fmt="{:.0f}"))
    write_json(CACHE / "m2_eda_grid_figure.json", {"style": style, "layout": {"nrows": 2, "ncols": 2, "figsize": [13, 9]}, "panels": click_panels})


def metrics():
    config = OmegaConf.load(ROOT / "src/model/sort_models/classic_sort_models/base_sort_conf.yaml")
    support = load_support(config)
    values = {name: mmap("dev", name).reshape(-1) for name in ("user_id", "item_id", "label", "impression_id")}
    result = {}
    for tag, directory in (("DIN", "din_seed42"), ("M1", "m1_d128_k2_seed42")):
        path = OUTPUT / directory / "best_predictions.npz"
        score = np.load(path)["score"]
        saved = json.loads((OUTPUT / directory / "result.json").read_text(encoding="utf-8"))
        computed = evaluate(values["user_id"], values["item_id"], values["label"], score, values["impression_id"], support=support)
        for key in ("AUC", "GAUC", "LogLoss"):
            assert abs(computed["overall"][key] - saved["metrics"]["overall"][key]) < 1e-7
        result[tag] = {"source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "metrics": computed, "trainable_parameters": saved["cost"]["trainable_parameters"]}
        print(tag, computed["overall"], flush=True)
    write_json(OUTPUT / "m2_eda_metrics.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    metrics() if args.metrics else analysis()
