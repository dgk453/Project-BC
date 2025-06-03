from dataclasses import dataclass, field
from configs import BaseConfig

@dataclass
class default_simulator_config(BaseConfig): 
    data_path: str = "gpudrive/data/processed/v1.1/training"
    dynamics_model: str = "delta_local"
    max_num_object: int = 64
    num_envs: int = 4
    sample_with_replacement = False
    action_type: str = "continuous"
    dataset_size: int = 1000