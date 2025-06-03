from dataclasses import dataclass, field
from configs.model.context_encoder.context_encoder_configs import default_context_encoder_config
from configs import BaseConfig

@dataclass
class context_encoder_config(BaseConfig):
    default: default_context_encoder_config = field(default_factory=default_context_encoder_config)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
        }
