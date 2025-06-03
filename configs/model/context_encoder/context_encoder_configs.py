from dataclasses import dataclass, field
from configs.model.mask_predictor import mask_predictor_config
from configs import BaseConfig

@dataclass
class default_context_encoder_config(BaseConfig):
    NAME: str = "MTREncoder"

    NUM_OF_ATTN_NEIGHBORS: int = 16
    NUM_INPUT_ATTR_AGENT: int = 30
    NUM_INPUT_ATTR_MAP: int = 13

    NUM_CHANNEL_IN_MLP_AGENT: int = 256
    NUM_CHANNEL_IN_MLP_MAP: int = 64
    NUM_LAYER_IN_MLP_AGENT: int = 3
    NUM_LAYER_IN_MLP_MAP: int = 5
    NUM_LAYER_IN_PRE_MLP_MAP: int = 3

    D_MODEL: int = 128
    NUM_ATTN_LAYERS: int = 6
    NUM_ATTN_HEAD: int = 8
    DROPOUT_OF_ATTN: float = 0.1

    USE_LOCAL_ATTN: bool = True
    MASK_PREDICTOR = None

    mask_predictor_all: mask_predictor_config = field(default_factory=mask_predictor_config)

    def __post_init__(self):
        self.MASK_PREDICTOR = self.mask_predictor_all.configs["default"]