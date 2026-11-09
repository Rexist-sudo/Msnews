"""M3 手动训练前检查：公共初始化、M2 最佳权重等价性及可选全量重放。"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

from .analysis import mechanism_diagnostics
from .initialization import verify_equivalence
from .model import CoInteractionRankMixer
from ..rankmixer.model import RankMixerDIN
from ....model_utils.checkpoint import load_initial_parameters, load_model
from ....model_utils.evaluation import benchmark, predict
from ....model_utils.ranking_metrics import METRIC_NAMES, calculate_metrics
from ....model_utils.training import write_json
from .....dataset.DataReader.data_reader_mmap import DataReaderMmap
from .....dataset.FeaturesGenerator.history_analysis import audit_history
from .....dataset.FeaturesGenerator.news_statistics import prepare_statistics


def load_config():
    """与训练入口使用相同的四级配置，不复制默认值。"""
    directory = Path(__file__).resolve().parent
    return OmegaConf.merge(*(OmegaConf.load(path) for path in (
        directory.parent / "base_sort_conf.yaml", directory.parent / "din/din_conf.yaml",
        directory.parent / "rankmixer/rankmixer_conf.yaml", directory / "configs/co_interaction_conf.yaml",
    )))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-parameters", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/ranking_study/m2_smooth_seed42/best.ckpt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-evaluation", action="store_true", help="全量重放并核对八项整体指标")
    parser.add_argument("--audit-history", action="store_true", help="逐请求核对 Train/Dev 原始历史与三路 mmap")
    parser.add_argument("--output", type=Path, help="可选检查记录；不创建正式训练目录")
    args = parser.parse_args()
    L.seed_everything(42, workers=True)
    torch.set_num_threads(4)
    config = load_config()
    print("核对严格历史统计缓存来源…", flush=True)
    statistics = prepare_statistics(config)
    reader = DataReaderMmap(config, Path(config.paths.out_basedir) / "extractored_feature/dev_features_mmap")
    model = CoInteractionRankMixer(config)
    initialization = load_initial_parameters(model, args.initial_parameters or config.initial_parameters)
    model.to(args.device).eval()
    result = {"formal_training_started": False, "initialization": initialization,
              "statistics": {name: statistics[name] for name in ("history_length", "state_count", "splits")}}
    try:
        batch = {name: value.to(args.device) for name, value in reader.get_batch(np.arange(512)).items()}
        result["common_initialization_equivalence"] = verify_equivalence(model, batch)
        print("公共随机初始化逐层核验通过", flush=True)
        if args.checkpoint.exists():
            baseline = load_model(RankMixerDIN, args.checkpoint).to(args.device).eval()
            model.load_state_dict(baseline.state_dict(), strict=False)
            result["best_checkpoint_equivalence"] = verify_equivalence(model, batch)
            result["cost"] = {"m2_parameters": sum(p.numel() for p in baseline.parameters()),
                              "m3_parameters": sum(p.numel() for p in model.parameters()),
                              "m2_trainable_parameters": sum(p.numel() for p in baseline.parameters() if p.requires_grad),
                              "m3_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                              "new_context_parameters": sum(p.numel() for p in model.co_blocks.parameters()),
                              "m2_inference": benchmark(baseline, batch), "m3_inference": benchmark(model, batch)}
            if args.full_evaluation:
                print(f"全量重放 {len(reader):,} 条 Dev 候选，分别计算 M3 和 M2_2 预测…", flush=True)
                scores = predict(model, reader)
                baseline_scores = predict(baseline, reader)
                np.testing.assert_allclose(scores, baseline_scores, atol=2e-6, rtol=1e-5)
                arrays = reader._ensure_arrays()
                print("预测对齐通过，核对公共八项指标…", flush=True)
                with ProcessPoolExecutor(max_workers=4) as pool:
                    metrics = calculate_metrics(arrays["user_id"], arrays["label"], scores, arrays["impression_id"], pool)
                    baseline_metrics = calculate_metrics(arrays["user_id"], arrays["label"], baseline_scores, arrays["impression_id"], pool)
                differences = {name: metrics[name] - baseline_metrics[name] for name in METRIC_NAMES}
                if max(abs(value) for value in differences.values()) > 2e-6:
                    raise AssertionError(f"初始化全量指标超出 2e-6 容差: {differences}")
                result["full_evaluation"] = {"rows": len(scores), "m2": baseline_metrics, "m3": metrics,
                                             "delta": differences, "max_score_error": float(np.max(np.abs(scores - baseline_scores)))}
        elif args.full_evaluation:
            raise FileNotFoundError(f"全量等价性核验需要 M2_2 checkpoint: {args.checkpoint}")
        else:
            result["best_checkpoint_equivalence"] = {"available": False, "path": str(args.checkpoint)}
        result["mechanism"] = mechanism_diagnostics(model, reader)
        if args.audit_history:
            print("逐请求审计 Train/Dev 原始历史…", flush=True)
            result["history_audit"] = audit_history(config)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
