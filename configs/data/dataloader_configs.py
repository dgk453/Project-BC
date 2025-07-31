from dataclasses import dataclass, field
from configs import BaseConfig

@dataclass
class default_dataloader_config(BaseConfig): 
    batch_size: int = 24
    prior_frame: int = 11  # Make this 11
    future_frame: int = 10 # Make this 10
    data_path: str = "data/train/" # make sure to not have a '/' prefix in the path
    num_total_samples: int = 10_000_000 # Number of samples to train on / N
                                         # Leave as None if you want all the samples from the files specified in sim config
    num_samples_per_file: int = 350 # Number of samples each file will contain
    batch_size: int = 24
    num_batch_in_rollout: int = 5
    torch_data_loader_kwarg: dict = field(default_factory=dict)

@dataclass
class mini_dataloader_config(BaseConfig): 
    batch_size: int = 64
    prior_frame: int = 11 # 1 second prior
    future_frame: int = 10 # 1 second
    # data_path: str = "data/1.1s_1s_mini_train" # make sure to not have a '/' prefix in the path
    num_total_samples: int = 200_000 # Number of samples to train on / N
                                     # Leave as None if you want all the samples from the files specified in sim config
    num_samples_per_file: int = 500 # Number of samples each file will contain
    torch_data_loader_kwarg: dict = field(default_factory=dict)

    def __post_init__(self): 
        self.data_path = f"data/{self.prior_frame}_{self.future_frame}_mini_train"