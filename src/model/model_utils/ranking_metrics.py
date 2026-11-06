"""排序统一评估：GAUC 按用户，HR/NDCG/MRR 按 Impression。"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score


METRIC_NAMES = ("AUC", "GAUC", "LogLoss", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR")


def _group_task(task):
    group_id, labels, scores, k_list = task
    row = _compute_user_metrics(group_id, list(zip(scores, labels)), k_list, False)
    return row["auc"], row["hr"], row["ndcg"], row["mrr"]


def _group_results(group, label, score, k_list, executor):
    order = np.argsort(group, kind="stable")
    ids, labels, scores = group[order], label[order], score[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    tasks = ((int(ids[start]), labels[start:stop], scores[start:stop], k_list)
             for start, stop in zip(starts[:-1], starts[1:]))
    return (executor.map(_group_task, tasks, chunksize=256) if executor is not None
            else map(_group_task, tasks))


def calculate_metrics(user, label, score, impression, executor=None):
    user, label, score, impression = [np.asarray(values).reshape(-1)
                                      for values in (user, label, score, impression)]
    score = score.astype(np.float64)
    if not (len(user) == len(label) == len(score) == len(impression)):
        raise ValueError("用户、Impression、标签和预测长度不一致")
    if not np.isfinite(score).all():
        raise ValueError("预测包含 NaN / Inf")
    result = dict.fromkeys(METRIC_NAMES, 0.0)
    result.update(rows=len(label), positive=int(label.sum()), users_with_auc=0,
                  User_Count=len(np.unique(user)), Impression_Count=0)
    if not len(label):
        return result
    if 0 < label.sum() < len(label):
        result["AUC"] = float(roc_auc_score(label, score))
    clipped = np.clip(score, 1e-15, 1 - 1e-15)
    result["LogLoss"] = float(-np.mean(label * np.log(clipped) + (1 - label) * np.log(1 - clipped)))
    aucs = [row[0] for row in _group_results(user, label, score, [], executor) if row[0] is not None]
    result["GAUC"] = float(np.mean(aucs)) if aucs else 0.0
    result["users_with_auc"] = len(aucs)
    # 同一请求可跨 batch；不同用户的同名请求不合并。同分候选保持输入顺序。
    _, groups = np.unique(np.column_stack((user, impression)), axis=0, return_inverse=True)
    ranking = list(_group_results(groups, label, score, [5, 10], executor))
    result["Impression_Count"] = len(ranking)
    for k in (5, 10):
        result[f"HR@{k}"] = float(np.mean([row[1][k] for row in ranking]))
        result[f"NDCG@{k}"] = float(np.mean([row[2][k] for row in ranking]))
    result["MRR"] = float(np.mean([row[3] for row in ranking]))
    return result


def evaluate(user, item, label, score, impression, support=None, train_users=None, executor=None):
    """整体、已见/未见与候选曝光分桶共用同一计算方式。"""
    user, item, label, score, impression = [np.asarray(values).reshape(-1)
                                           for values in (user, item, label, score, impression)]
    if not all(len(values) == len(user) for values in (item, label, score, impression)):
        raise ValueError("用户、新闻、Impression、标签和预测长度不一致")
    if executor is None and len(user) >= 10000:
        with ProcessPoolExecutor(max_workers=4) as pool:
            return evaluate(user, item, label, score, impression, support, train_users, pool)
    masks = {"overall": np.ones(len(user), dtype=bool)}
    if support is not None:
        seen_user = support["user"][user] > 0
        seen_item = support["item"][item] > 0
        exposure = support["candidate"][item]
        masks.update(seen_uid=seen_user, unseen_uid=~seen_user, seen_item=seen_item, unseen_item=~seen_item,
                     exposure_0=exposure == 0, exposure_1_9=(exposure >= 1) & (exposure < 10),
                     exposure_10_99=(exposure >= 10) & (exposure < 100), exposure_100_plus=exposure >= 100)
    elif train_users is not None:
        seen_ids = [uid for uid in np.unique(user) if uid in train_users or str(uid) in train_users]
        seen_user = np.isin(user, seen_ids)
        masks.update(seen_uid=seen_user, unseen_uid=~seen_user)
    result = {}
    for name, mask in masks.items():
        result[name] = calculate_metrics(user[mask], label[mask], score[mask], impression[mask], executor)
        result[name]["row_coverage"] = float(mask.mean()) if len(mask) else 0.0
        overall = result["overall"]
        for count in ("users_with_auc", "Impression_Count"):
            result[name][count + "_coverage"] = result[name][count] / overall[count] if overall[count] else 0.0
    return result


def _compute_user_metrics(uid, items, k_list, is_cold):
    if not items:
        return None

    preds = [float(x[0]) for x in items]
    labels = [int(x[1]) for x in items]
    
    # 默认值
    res = {
        'uid': uid,
        'preds': preds,
        'labels': labels,
        'is_cold': is_cold,
        'auc': None,
        'mrr': 0.0,
        'hr': {k: 0.0 for k in k_list},
        'ndcg': {k: 0.0 for k in k_list},
    }

    # --- AUC ---
    if len(set(labels)) > 1:
        try:
            res['auc'] = roc_auc_score(labels, preds)
        except ValueError: 
            pass

    # --- Sorting ---
    # items is list of (score, label)
    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    all_sorted_labels = np.array([x[1] for x in items_sorted])
    
    num_positives = sum(labels)
    if num_positives == 0:
        return res

    # --- Global MRR (Mean Reciprocal Rank over full list) ---
    hits = np.where(all_sorted_labels == 1)[0]
    if len(hits) > 0:
        # hits[0] 是第一个命中位置的索引(0-based)，排名为 hits[0] + 1
        res['mrr'] = 1.0 / (hits[0] + 1)
        
    # --- TopK Metrics (HR@K, NDCG@K) ---
    for k in k_list:
        top_k_labels = all_sorted_labels[:k]
        
        # HR@K
        if np.sum(top_k_labels) > 0:
            res['hr'][k] = 1.0
        
        # NDCG@K
        ranks = np.arange(1, len(top_k_labels) + 1)
        discounts = 1.0 / np.log2(ranks + 1)
        dcg = np.sum(top_k_labels * discounts)
        
        n_ideal = min(num_positives, k)
        ideal_ranks = np.arange(1, n_ideal + 1)
        idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
        
        if idcg > 0:
            res['ndcg'][k] = dcg / idcg
            
    return res

