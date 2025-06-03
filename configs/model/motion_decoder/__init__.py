from dataclasses import dataclass, field
from configs.model.motion_decoder.motion_decoder_configs import default_motion_decoder_config
from configs import BaseConfig

@dataclass
class motion_decoder_config(BaseConfig):
    default: default_motion_decoder_config = field(default_factory=default_motion_decoder_config)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
        }
