"""M3 零输出初始化与 M2_2 逐层等价性核验，不更新训练权重。"""

import torch

from ..rankmixer.model import RankMixerDIN
from .....dataset.DataReader.data_reader_mmap import DataReaderMmap


@torch.no_grad()
def verify_equivalence(model, batch: dict) -> dict:
    """在同一 batch 比较 logit、三个兴趣 Token、两层输出和概率。

    公共参数取自当前 M3。要求 ContextFFN 为零输出初始化；浮点加法顺序
    不同，使用 atol=2e-6、rtol=1e-5，同时记录真实最大绝对误差。
    """
    was_training = model.training
    model.eval()
    # 核验用对照模型不消耗正式训练后续的随机数序列。
    with torch.random.fork_rng(devices=[]):
        baseline = RankMixerDIN(model.config).to(next(model.parameters()).device).eval()
    baseline.load_state_dict(model.initialization_state_dict(), strict=True)
    fields = baseline.get_inp_embedding(batch).split(baseline.field_dims, dim=1)
    tokens = torch.stack([projection(value) for projection, value in zip(baseline.field_projections, fields)], dim=1)
    logits, trace = model.forward_with_trace(batch)
    errors = {}

    def compare(name, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        errors[name] = float((actual - expected).abs().max())

    try:
        for index in model.interest_indices:
            compare(f"interest_token_{model.field_names[index]}", trace["layers"][0]["interest_tokens"][:, index], tokens[:, index])
        for number, (layer, block) in enumerate(zip(trace["layers"], baseline.interaction), start=1):
            tokens = block(tokens)
            compare(f"rankmixer_{number}", layer["tokens"], tokens)
            compare(f"history_{number}", layer["history"], trace["initial_history"])
        compare("logits", logits, baseline.score_head(tokens.mean(dim=1)))
        compare("probabilities", torch.sigmoid(logits), baseline(batch))
        for number, delta in enumerate(trace["layers"][1]["interest_deltas"]):
            compare(f"second_interest_delta_{number}", delta, torch.zeros_like(delta))
    finally:
        model.train(was_training)
    return {"passed": True, "rows": len(logits), "atol": 2e-6, "rtol": 1e-5,
            "max_absolute_errors": errors}


def check_training_initialization(model) -> dict:
    """训练启动前用固定真实候选核验，结果写入 initialization.json。"""
    reader = DataReaderMmap(model.config, f"{model.config.paths.out_basedir}/extractored_feature/dev_features_mmap")
    try:
        return verify_equivalence(model, reader.get_batch(list(range(min(512, len(reader))))))
    finally:
        reader.close()
