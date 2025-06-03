from dataclasses import dataclass, field
from configs.data.dataloader_configs import default_dataloader_config, mini_dataloader_config
from configs import BaseConfig

@dataclass
class dataloader_config(BaseConfig):
    default: default_dataloader_config = field(default_factory=default_dataloader_config)
    mini_dataloader: mini_dataloader_config = field(default_factory=mini_dataloader_config)
    def __post_init__(self):
        self.configs = {
            "default": self.default,
            "mini_500_000": self.mini_dataloader
        }
