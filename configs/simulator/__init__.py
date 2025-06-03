from dataclasses import dataclass, field
from configs.simulator.simulator_config import default_simulator_config
from configs import BaseConfig

@dataclass
class simulator_config(BaseConfig):
    default: default_simulator_config = field(default_factory=default_simulator_config)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
        }
