"""从原始请求核对末 50 条历史，并生成 M3 按历史和匹配位置的分析分组。"""

import csv
import json
from pathlib import Path

import numpy as np

from ..DataReader.data_reader_mmap import DataReaderMmap


HISTORY_FIELDS = ("user_history", "user_history_clicked_category", "user_history_clicked_subcategory")


def audit_history(config, data_path: str = "Data/MIND") -> dict:
    """逐请求核对原始顺序，逐候选行核对三路历史、mask 和统计缓存索引。

    包含全部 Train/Dev 行，不依赖抽样。位置由有效位反向计数，padding=0。
    统计数值及其严格时间来源由 prepare_statistics 的来源摘要检查负责。
    """
    base = Path(config.paths.out_basedir)
    features = base / "extractored_feature"
    news_map = json.loads((base / "preprocess/news_id_map.json").read_text(encoding="utf-8"))
    user_map = json.loads((base / "preprocess/user_id_map.json").read_text(encoding="utf-8"))
    mapping = json.loads((features / "original_val_2_embedding_idx_dict.json").read_text(encoding="utf-8"))
    item_count = int(config.embeddings.embedding_table_size.item_id)
    categories, subcategories = np.zeros(item_count, dtype=int), np.zeros(item_count, dtype=int)
    with (base / "preprocess/all_news_preprocess.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            categories[int(row[0])] = mapping["category"][0][row[1]]
            subcategories[int(row[0])] = mapping["subcategory"][0][row[2]]
    results = {}
    for split in ("train", "dev"):
        reader = DataReaderMmap(config, features / f"{split}_features_mmap")
        arrays = reader._ensure_arrays()
        impression = arrays["impression_id"].reshape(-1)
        size = int(impression.max()) + 1
        ids, first, candidate_counts = np.unique(impression, return_index=True, return_counts=True)
        first_by_id, count_by_id = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
        first_by_id[ids], count_by_id[ids] = first, candidate_counts
        histories = np.full((size, 50), -1, dtype=np.int64)
        lengths = np.zeros(size, dtype=np.int64)
        offset, requests = 0, 0
        with (Path(data_path) / f"MINDsmall_{split}/behaviors.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                request, uid = int(row[0]), user_map[row[1]]
                history = [news_map[name] for name in row[3].split()][-50:]
                count = len(history)
                histories[request, :count] = history
                lengths[request] = count
                candidates = [token.rsplit("-", 1) for token in row[4].split()]
                # 特征按请求时间排序，原始 behaviors.tsv 的行序并非时间序。
                start = first_by_id[request]
                stop = start + len(candidates)
                if count_by_id[request] != len(candidates) or not np.all(impression[start:stop] == request):
                    raise ValueError(f"{split} 请求 {request} 的候选顺序与缓存不一致")
                if not np.all(arrays["user_id"][start:stop].reshape(-1) == uid):
                    raise ValueError(f"{split} 请求 {request} 用户不一致")
                np.testing.assert_array_equal(arrays["item_id"][start:stop].reshape(-1),
                                              [news_map[name] for name, _ in candidates])
                np.testing.assert_array_equal(arrays["label"][start:stop].reshape(-1),
                                              [int(label) for _, label in candidates])
                offset, requests = offset + len(candidates), requests + 1
        if offset != len(reader):
            raise ValueError(f"{split} 原始候选数 {offset} 与缓存 {len(reader)} 不一致")
        for start in range(0, len(reader), 8192):
            current = impression[start:start + 8192]
            history = histories[current]
            valid = history > 0
            for name, table in zip(HISTORY_FIELDS, (None, categories, subcategories)):
                expected = history if table is None else np.where(valid, table[np.maximum(history, 0)], -1)
                np.testing.assert_array_equal(arrays[name][start:start + 8192], expected)
                np.testing.assert_array_equal(arrays[name + "_mask"][start:start + 8192] > 0, valid)
        history_stats = arrays["history_statistics"]
        history_state = arrays["history_state"]
        for start in range(0, size, 8192):
            padding = histories[start:start + 8192] <= 0
            # 未出现的 request=0 不属于真实请求；其 open_memmap 初值不作语义检查。
            if start == 0:
                padding[0] = False
            if np.any(history_stats[start:start + 8192][padding] != 0):
                raise ValueError(f"{split} padding 的历史统计应为 0")
            if np.any(history_state[start:start + 8192][padding] != 0):
                raise ValueError(f"{split} padding 的历史状态应为 0")
        ids = np.unique(impression)
        counts = lengths[ids]
        groups = {"empty": counts == 0, "1_9": (counts >= 1) & (counts < 10),
                  "10_19": (counts >= 10) & (counts < 20), "20_50": counts >= 20}
        candidate_lengths = lengths[impression]
        labels = arrays["label"].reshape(-1)
        results[split] = {"rows": len(reader), "requests": requests, "history_groups": {}}
        for name, mask in groups.items():
            keep = np.isin(impression, ids[mask])
            results[split]["history_groups"][name] = {
                "requests": int(mask.sum()), "candidates": int(keep.sum()),
                "clicks": int(labels[keep].sum()),
            }
        results[split]["effective_positions"] = [int((counts >= pos).sum()) for pos in range(1, 51)]
        results[split]["candidate_mean_history_length"] = float(candidate_lengths.mean())
        reader.close()
        print(f"历史全量核验通过: {split}, {requests:,} 请求, {offset:,} 候选", flush=True)
    return results


def history_group_masks(reader: DataReaderMmap) -> dict[str, np.ndarray]:
    """按候选返回历史长度与类别匹配组；近期为 1–5，较早为 6–50。

    各字段匹配组互斥：recent_only、earlier_only、both、neither。匹配位置
    由有效历史顺序决定，请求分组评估时沿用完整原始 Impression ID。
    """
    arrays = reader._ensure_arrays()
    rows = len(reader)
    lengths = np.asarray(arrays["user_history_mask"].sum(axis=1)).reshape(-1)
    masks = {"history_empty": lengths == 0, "history_1_9": (lengths >= 1) & (lengths < 10),
             "history_10_19": (lengths >= 10) & (lengths < 20), "history_20_50": lengths >= 20}
    for field, history_name in (("category", HISTORY_FIELDS[1]), ("subcategory", HISTORY_FIELDS[2])):
        recent, earlier = np.zeros(rows, dtype=bool), np.zeros(rows, dtype=bool)
        for start in range(0, rows, 16384):
            stop = min(start + 16384, rows)
            valid = arrays[history_name + "_mask"][start:stop] > 0
            positions = np.cumsum(valid[:, ::-1], axis=1)[:, ::-1] * valid
            matches = arrays[history_name][start:stop] == arrays[field][start:stop].reshape(-1, 1)
            recent[start:stop] = (matches & valid & (positions <= 5)).any(axis=1)
            earlier[start:stop] = (matches & valid & (positions > 5)).any(axis=1)
        masks.update({f"{field}_recent_only": recent & ~earlier,
                      f"{field}_earlier_only": earlier & ~recent,
                      f"{field}_both": recent & earlier,
                      f"{field}_neither": ~recent & ~earlier})
    return masks
