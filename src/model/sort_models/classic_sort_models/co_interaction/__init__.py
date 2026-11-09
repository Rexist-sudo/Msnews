"""M3 双层序列—字段交互主干。"""

from .layers import CoBlock, ContextFFN
from .model import CoInteractionRankMixer

__all__ = ["CoInteractionRankMixer", "CoBlock", "ContextFFN"]
