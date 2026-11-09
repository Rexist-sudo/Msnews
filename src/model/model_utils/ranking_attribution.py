"""将按用户平均的 GAUC 分解为同/跨曝光桶贡献，保留并列分数的半分。"""

import numpy as np


def support_pair_contributions(user, label, score, bucket) -> tuple[dict, dict]:
    """返回四桶的正负候选配对贡献与按正样本归属的逐行贡献。

    每个用户的分母为 P_u*N_u；累加排序正确对（同分计 0.5）后，再除以
    含正负候选的用户数。matrix[i,j] 表示正样本在桶 i、负样本在桶 j
    的 GAUC 贡献，矩阵总和等于公共评估器的整体 GAUC。row_weights
    以正候选归属划分收益，适用于历史长度/兴趣匹配的互斥联合切片。
    """
    user, label, score, bucket = [np.asarray(value).reshape(-1) for value in (user, label, score, bucket)]
    if not all(len(value) == len(user) for value in (label, score, bucket)):
        raise ValueError("GAUC 归因输入长度不一致")
    if np.any((bucket < 0) | (bucket > 3)):
        raise ValueError("bucket 必须为 0、1、2、3")
    order = np.argsort(user, kind="stable")
    starts = np.r_[0, np.flatnonzero(user[order][1:] != user[order][:-1]) + 1, len(order)]
    matrix, mass = np.zeros((4, 4)), np.zeros((4, 4))
    weights = {name: np.zeros(len(user), dtype=np.float64) for name in ("overall", "same_bucket", "pair_mass")}
    eligible = 0
    for start, stop in zip(starts[:-1], starts[1:]):
        rows = order[start:stop]
        positive = label[rows] > 0
        positive_rows, negative_rows = rows[positive], rows[~positive]
        if not len(positive_rows) or not len(negative_rows):
            continue
        eligible += 1
        denominator = len(positive_rows) * len(negative_rows)
        negative_scores = np.sort(score[negative_rows])
        values = score[positive_rows]
        low = np.searchsorted(negative_scores, values, side="left")
        high = np.searchsorted(negative_scores, values, side="right")
        weights["overall"][positive_rows] = (low + high) / (2 * denominator)
        weights["pair_mass"][positive_rows] = len(negative_rows) / denominator
        for i in range(4):
            current = positive_rows[bucket[positive_rows] == i]
            if not len(current):
                continue
            for j in range(4):
                negatives = np.sort(score[negative_rows[bucket[negative_rows] == j]])
                if not len(negatives):
                    continue
                low = np.searchsorted(negatives, score[current], side="left")
                high = np.searchsorted(negatives, score[current], side="right")
                wins = (low + high) / (2 * denominator)
                matrix[i, j] += wins.sum()
                mass[i, j] += len(current) * len(negatives) / denominator
                if i == j:
                    weights["same_bucket"][current] = wins
    if eligible:
        matrix /= eligible
        mass /= eligible
        for values in weights.values():
            values /= eligible
    weights["cross_bucket"] = weights["overall"] - weights["same_bucket"]
    same = float(np.trace(matrix))
    return {"users_with_auc": eligible, "overall": float(matrix.sum()), "same_bucket": same,
            "cross_bucket": float(matrix.sum() - same), "matrix": matrix.tolist(),
            "pair_mass": mass.tolist(), "same_pair_mass": float(np.trace(mass)),
            "bucket_order": ["0", "1–9", "10–99", "≥100"]}, weights


def request_pair_accuracy(impression, label, score, mask) -> dict:
    """同一原始请求内、组内正负候选的排序正确率，按有效请求平均。"""
    impression, label, score, mask = [np.asarray(value).reshape(-1) for value in (impression, label, score, mask)]
    order = np.argsort(impression, kind="stable")
    order = order[mask[order]]
    if not len(order):
        return {"accuracy": 0.0, "requests_with_pairs": 0, "pairs": 0}
    ids = impression[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    total, requests, pairs = 0.0, 0, 0
    for start, stop in zip(starts[:-1], starts[1:]):
        rows = order[start:stop]
        positive, negative = score[rows][label[rows] > 0], np.sort(score[rows][label[rows] == 0])
        if not len(positive) or not len(negative):
            continue
        low = np.searchsorted(negative, positive, side="left")
        high = np.searchsorted(negative, positive, side="right")
        count = len(positive) * len(negative)
        total += float((low + high).sum()) / (2 * count)
        pairs += count
        requests += 1
    return {"accuracy": total / requests if requests else 0.0,
            "requests_with_pairs": requests, "pairs": pairs}
