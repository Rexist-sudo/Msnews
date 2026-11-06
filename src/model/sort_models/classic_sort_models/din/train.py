"""DIN 训练入口，复用排序公共训练流程。"""

from pathlib import Path

from .model import DIN
from ....model_utils.training import main


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    main(DIN, directory, [directory / "din_conf.yaml"])
