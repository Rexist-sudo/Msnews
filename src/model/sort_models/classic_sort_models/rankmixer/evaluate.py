"""RankMixerDIN checkpoint 评估与预测复核。"""

from .model import RankMixerDIN
from ....model_utils.evaluation import main


if __name__ == "__main__":
    main(RankMixerDIN)
