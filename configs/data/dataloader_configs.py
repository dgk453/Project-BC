from dataclasses import dataclass, field
from configs import BaseConfig

@dataclass
class default_dataloader_config(BaseConfig): 
    batch_size: int = 24
    prior_frame: int = 20
    future_frame: int = 5
    data_path: str = "data/train/" # make sure to not have a '/' prefix in the path
    num_total_samples: int = 10_000_000 # Number of samples to train on / N
                                         # Leave as None if you want all the samples from the files specified in sim config
    num_samples_per_file: int = 350 # Number of samples each file will contain
    batch_size: int = 24
    num_batch_in_rollout: int = 5
    torch_data_loader_kwarg: dict = field(default_factory=dict)

@dataclass
class mini_dataloader_config(BaseConfig): 
    batch_size: int = 24
    prior_frame: int = 20
    future_frame: int = 5
    data_path: str = "data/train/" # make sure to not have a '/' prefix in the path
    num_total_samples: int = 500_000 # Number of samples to train on / N
                                         # Leave as None if you want all the samples from the files specified in sim config
    num_samples_per_file: int = 350 # Number of samples each file will contain
    batch_size: int = 24
    num_batch_in_rollout: int = 5
    torch_data_loader_kwarg: dict = field(default_factory=dict)
