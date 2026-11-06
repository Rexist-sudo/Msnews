"""DIN checkpoint 评估与预测复核。"""

from .model import DIN
from ....model_utils.evaluation import main


if __name__ == "__main__":
    main(DIN)
