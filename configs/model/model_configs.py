from dataclasses import dataclass, field
from configs.model.context_encoder import context_encoder_config
from configs.model.motion_decoder import motion_decoder_config
from configs import BaseConfig

@dataclass
class default_model_config(BaseConfig): 
    context_encoder_all: context_encoder_config = field(default_factory=context_encoder_config)
    motion_decoder_all: motion_decoder_config = field(default_factory=motion_decoder_config)

    def __post_init__(self):
        self.CONTEXT_ENCODER = self.context_encoder_all.configs["default"]
        self.MOTION_DECODER = self.motion_decoder_all.configs["default"]