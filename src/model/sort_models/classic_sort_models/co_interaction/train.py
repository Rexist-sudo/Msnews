"""M3 训练入口，配置按 Base→DIN→RankMixer→Co-Interaction 合并。"""

from pathlib import Path

from .model import CoInteractionRankMixer
from .analysis import analyze_experiment
from .initialization import check_training_initialization
from ....model_utils.training import main


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    main(CoInteractionRankMixer, directory, [
        directory.parent / "din/din_conf.yaml",
        directory.parent / "rankmixer/rankmixer_conf.yaml",
        directory / "configs/co_interaction_conf.yaml",
    ], initialization_check=check_training_initialization, analysis_function=analyze_experiment)
