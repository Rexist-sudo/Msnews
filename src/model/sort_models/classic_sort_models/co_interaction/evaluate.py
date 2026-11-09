"""M3 最佳 checkpoint 独立评估与逐候选预测核对。"""

from .model import CoInteractionRankMixer
from .analysis import analyze_experiment
from ....model_utils.evaluation import main


if __name__ == "__main__":
    main(CoInteractionRankMixer, analysis_function=analyze_experiment)
