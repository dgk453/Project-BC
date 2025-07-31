from dataclasses import dataclass, field
from configs.simulator import simulator_config
from configs.data import dataloader_config
from configs.model import model_config
from configs.model.optim_configs import optim_config
from configs import BaseConfig
import datetime
import string
import random

def random_string(length):
    pool = string.ascii_letters + string.digits
    return ''.join(random.choice(pool) for i in range(length))


@dataclass
class Config(BaseConfig): 
    '''
    Data (dataloader), model, simulator
    '''

    OPTIMIZATION: optim_config = field(default_factory=optim_config)

    dataloader_type: str = "default"
    simulator_type: str = "default"
    model_type: str = "default"

    automatic_fix: bool = True

    root: str = "/scratch/cluster/abitpal/CausalMTR-BC" # No backslash at the end

    simulator_config_all: simulator_config = field(default_factory=simulator_config)
    dataloader_config_all: dataloader_config = field(default_factory=dataloader_config)
    model_config_all: model_config = field(default_factory=model_config)

    # Model misc attributes
    model_device: str = "cuda:0"
    model_device_id: int = 0

    # Training attributes
    log_interval: int = 5
    plot_interval: int = 150
    BC_attributes: dict = field(default_factory=dict)
    wandb_project_name: str = field(default_factory=lambda: f"cmtr-bc-training-{datetime.date.today()}")

    def __post_init__(self):
        self.SIMULATOR = self.simulator_config_all.configs[self.simulator_type]
        self.DATALOADER = self.dataloader_config_all.configs[self.dataloader_type]
        self.MODEL = self.model_config_all.configs[self.model_type]

        self.model_save_path = f"{self.root}/models/{self.wandb_project_name}-{random_string(5)}.pth" 

        if (self.automatic_fix): 
            self.MODEL.CONTEXT_ENCODER.NUM_INPUT_ATTR_AGENT = 18 + self.DATALOADER.prior_frame
            self.MODEL.MOTION_DECODER.NUM_FUTURE_FRAMES = self.DATALOADER.future_frame
        
        # # Ensure that the model config is compatible with the simulator config
        # assert self.model_config.simulator_type == self.simulator_type, \
        #     f"Model config {self.model_type} is not compatible with simulator type {self.simulator_type}"



    


