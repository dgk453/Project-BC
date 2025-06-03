from dataclasses import dataclass, field
from configs.model.mask_predictor.mask_predictor_configs import default_mask_predictor_config
from configs import BaseConfig

@dataclass
class mask_predictor_config(BaseConfig):
    default: default_mask_predictor_config = field(default_factory=default_mask_predictor_config)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
        }
