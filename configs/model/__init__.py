from dataclasses import dataclass, field
from configs.model.model_configs import default_model_config
from configs import BaseConfig

@dataclass
class model_config(BaseConfig):
    default: default_model_config = field(default_factory=default_model_config)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
        }
