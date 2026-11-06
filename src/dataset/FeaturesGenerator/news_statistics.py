"""按请求时间查询新闻曝光、响应和分层收缩状态。"""

import numpy as np


class NewsStatistics:
    """统计仅来自 Train；同秒请求共用更新前的计数。"""

    def __init__(self, item, time, label, request_time, group, lambda_group=100.0, lambda_item=10.0):
        item = np.asarray(item, dtype=np.int64)
        self.group = np.asarray(group, dtype=np.int64)
        self.lambda_group, self.lambda_item = lambda_group, lambda_item
        self.origin = int(np.min(request_time))
        self.end = int(np.max(request_time)) + 1
        self.span = self.end - self.origin + 1
        offset = np.asarray(time, dtype=np.int64) - self.origin
        self.item_table = self._build_table(item, offset, label, len(group))
        self.group_table = self._build_table(self.group[item], offset, label, int(self.group.max()) + 1)
        order = np.argsort(time, kind="stable")
        self.time = np.asarray(time)[order]
        self.click = np.r_[0, np.cumsum(np.asarray(label)[order], dtype=np.int64)]
        self.request_time = np.sort(request_time)

    def _build_table(self, entity, offset, label, size):
        key = entity * self.span + offset
        order = np.argsort(key, kind="stable")
        keys, first, counts = np.unique(key[order], return_index=True, return_counts=True)
        clicks = np.add.reduceat(np.asarray(label, dtype=np.int64)[order], first)
        # 前缀和按实体连续存储，查询时扣除该实体起点之前的累计值。
        return (keys, np.r_[0, np.cumsum(counts)], np.r_[0, np.cumsum(clicks)],
                np.searchsorted(keys, np.arange(size, dtype=np.int64) * self.span))

    def _lookup(self, table, entity, time):
        keys, exposure, click, starts = table
        offset = np.clip(time - self.origin, 0, self.span - 1)
        stop = np.searchsorted(keys, entity * self.span + offset, side="left")
        start = starts[entity]
        return exposure[stop] - exposure[start], click[stop] - click[start]

    def query(self, item, time):
        """支持候选或历史矩阵，padding 归入 ID 0；返回时刻 t 之前的统计。"""
        item, time = np.broadcast_arrays(np.asarray(item, dtype=np.int64), np.asarray(time, dtype=np.int64))
        item = np.maximum(item, 0)
        exposure, click = self._lookup(self.item_table, item, time)
        group_exposure, group_click = self._lookup(self.group_table, self.group[item], time)
        total_exposure = np.searchsorted(self.time, time, side="left")
        total_click = self.click[total_exposure]
        requests = np.searchsorted(self.request_time, time, side="left")
        # 首次请求尚无反馈，用中性先验初始化，并通过 M=0 与未知状态表达缺失。
        global_ctr = np.divide(total_click, total_exposure, out=np.full(item.shape, 0.5), where=total_exposure > 0)
        group_ctr = (group_click + self.lambda_group * global_ctr) / (group_exposure + self.lambda_group)
        smooth_ctr = (click + self.lambda_item * group_ctr) / (exposure + self.lambda_item)
        raw_ctr = np.divide(click, exposure, out=np.zeros(item.shape), where=exposure > 0)
        epi = np.divide(exposure, requests, out=np.zeros(item.shape), where=requests > 0)
        return {"exposure": exposure, "click": click, "requests": requests, "epi": epi,
                "raw_ctr": raw_ctr, "smooth_ctr": smooth_ctr, "group_ctr": group_ctr,
                "group_exposure": group_exposure, "global_ctr": global_ctr}

    def fit_boundaries(self, interval=3600):
        """每小时用此前已曝光新闻的分位数更新边界，Dev 沿用 Train 末边界。"""
        self.boundary_time = np.r_[np.arange(self.origin // interval * interval, self.end, interval), self.end]
        epi, av = [], []
        for cutoff in self.boundary_time:
            values = self.query(np.arange(len(self.group)), cutoff)
            known = values["exposure"] > 0
            epi.append(np.quantile(values["epi"][known], [0.2, 0.4, 0.6, 0.8]) if known.any() else np.zeros(4))
            av.append(np.quantile(1 - values["smooth_ctr"][known], [0.2, 0.4, 0.6, 0.8]) if known.any() else np.zeros(4))
        self.epi_edges, self.av_edges = np.asarray(epi), np.asarray(av)

    def features(self, item, time):
        """输出 [..., 4] 原始/平滑特征及 [...,] 联合状态，0 为零曝光。"""
        values = self.query(item, time)
        known = values["exposure"] > 0
        raw = np.stack((values["epi"], np.where(known, 1 - values["raw_ctr"], 0),
                        np.log1p(values["exposure"]), known), axis=-1).astype(np.float32)
        smooth = raw.copy()
        smooth[..., 1] = 1 - values["smooth_ctr"]
        boundary = np.searchsorted(self.boundary_time, time, side="right") - 1
        boundary = np.clip(boundary, 0, len(self.boundary_time) - 1)
        x = np.sum(values["epi"][..., None] >= self.epi_edges[boundary], axis=-1)
        y = np.sum((1 - values["smooth_ctr"])[..., None] >= self.av_edges[boundary], axis=-1)
        state = np.where(known, 1 + 5 * x + y, 0).astype(np.int64)
        return raw, smooth, state


def prepare_statistics(config, data_path="Data/MIND", overwrite=False):
    """生成两组模型共用的候选统计，以及按请求索引的有效历史统计。"""
    import hashlib
    import json
    import time
    from pathlib import Path

    import pandas as pd

    directory = Path(config.paths.out_basedir) / "extractored_feature"
    news_path = Path(config.paths.out_basedir) / "preprocess/all_news_preprocess.csv"
    metadata_path = directory / "news_statistics.json"
    length = int(config.features.array_max_length.user_history)
    paths, parts = [news_path, Path(__file__)], {}
    for split in ("train", "dev"):
        split_dir = directory / f"{split}_features_mmap"
        manifest_path = split_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest["parts"]) != 1:
            raise ValueError("统计生成与 DataReaderMmap 共用单 part 数据")
        parts[split] = split_dir / manifest["parts"][0]["path"]
        paths.extend([Path(data_path) / f"MINDsmall_{split}/behaviors.tsv", manifest_path])
        paths.extend(parts[split] / f"{name}.npy" for name in ("item_id", "impression_id", "user_history"))
    paths.append(parts["train"] / "label.npy")
    sources = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        sources[str(path)] = digest.hexdigest()
    names = ("item_statistics", "item_state", "history_statistics", "history_state")
    targets = [directory / f"{split}_{name}.npy" for split in ("train", "dev") for name in names]
    if metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["sources"] == sources and metadata["history_length"] == length and all(path.exists() for path in targets):
            return metadata
        raise ValueError("统计输入或缓存已变化，请使用 --overwrite 重新生成")

    started = time.perf_counter()
    requests, request_times, arrays = {}, {}, {}
    for split, part in parts.items():
        behavior = pd.read_csv(Path(data_path) / f"MINDsmall_{split}/behaviors.tsv", sep="\t", header=None, usecols=[0, 2])
        ids = behavior[0].to_numpy(dtype=np.int64)
        times = pd.to_datetime(behavior[2], format="%m/%d/%Y %I:%M:%S %p").astype("int64").to_numpy() // 10**9
        if len(np.unique(ids)) != len(ids) or ids.min() <= 0:
            raise ValueError("每个划分的 Impression ID 必须为唯一正整数")
        mapping = np.zeros(int(ids.max()) + 1, dtype=np.int64)
        mapping[ids] = times
        requests[split] = mapping
        request_times[split] = times
        arrays[split] = {name: np.load(part / f"{name}.npy", mmap_mode="r")
                         for name in ("item_id", "impression_id", "user_history")}
        np.testing.assert_array_equal(np.sort(ids), np.unique(arrays[split]["impression_id"]))
    train = arrays["train"]
    label = np.load(parts["train"] / "label.npy", mmap_mode="r").reshape(-1)
    news = pd.read_csv(news_path, sep="\t", header=None, usecols=[0, 1, 2])
    group = np.zeros(int(config.embeddings.embedding_table_size.item_id), dtype=np.int64)
    group[news[0].to_numpy()] = pd.factorize(pd.MultiIndex.from_frame(news[[1, 2]]), sort=True)[0] + 1
    timestamp = requests["train"][train["impression_id"].reshape(-1)]
    statistics = NewsStatistics(train["item_id"].reshape(-1), timestamp, label, request_times["train"], group)
    statistics.fit_boundaries()
    if statistics.end > request_times["dev"].min():
        raise ValueError("Dev 请求时间应晚于 Train 统计截止时刻")
    metadata = dict(sources=sources, history_length=length, lambda_group=statistics.lambda_group,
                    lambda_item=statistics.lambda_item, boundary_interval=3600, state_count=26, splits={})
    for split, values in arrays.items():
        item, impression = values["item_id"].reshape(-1), values["impression_id"].reshape(-1)
        ids, first = np.unique(impression, return_index=True)
        shapes = {"item_statistics": (len(item), 5), "item_state": (len(item),),
                  "history_statistics": (len(requests[split]), length, 5), "history_state": (len(requests[split]), length)}
        output = {name: np.lib.format.open_memmap(directory / f"{split}_{name}.tmp.npy", mode="w+",
                                                dtype=np.uint8 if name.endswith("state") else np.float32, shape=shape)
                  for name, shape in shapes.items()}
        for start in range(0, len(item), 200000):
            stop = start + 200000
            raw, smooth, state = statistics.features(item[start:stop], requests[split][impression[start:stop]])
            output["item_statistics"][start:stop] = np.concatenate((raw, smooth[:, 1:2]), axis=-1)
            output["item_state"][start:stop] = state
        for start in range(0, len(ids), 4096):
            current = ids[start:start + 4096]
            history = values["user_history"][first[start:start + 4096]]
            if history.shape[1] != length:
                raise ValueError("历史长度与统计配置不一致")
            raw, smooth, state = statistics.features(history, requests[split][current, None])
            packed = np.concatenate((raw, smooth[..., 1:2]), axis=-1)
            packed[history <= 0] = 0
            output["history_statistics"][current] = packed
            output["history_state"][current] = state
        for name, value in output.items():
            value.flush()
            value._mmap.close()
            (directory / f"{split}_{name}.tmp.npy").replace(directory / f"{split}_{name}.npy")
        metadata["splits"][split] = {"rows": len(item), "requests": len(ids)}
        print(f"{split}: {len(item):,} candidates, {len(ids):,} request histories", flush=True)
    metadata["build_seconds"] = time.perf_counter() - started
    metadata["cache_bytes"] = sum(path.stat().st_size for path in targets)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata


if __name__ == "__main__":
    import argparse
    import json
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/model/sort_models/classic_sort_models/base_sort_conf.yaml")
    parser.add_argument("--data-path", default="Data/MIND")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = prepare_statistics(OmegaConf.load(args.config), args.data_path, args.overwrite)
    print(json.dumps({key: value for key, value in metadata.items() if key != "sources"}, ensure_ascii=False, indent=2))
