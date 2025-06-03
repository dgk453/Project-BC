from dataclasses import dataclass, field
from configs.simulator import simulator_config
from configs.data import dataloader_config
from configs.model import model_config
from configs import BaseConfig

@dataclass
class Config(BaseConfig): 
    '''
    Data (dataloader), model, simulator
    '''
    dataloader_type: str = "default"
    simulator_type: str = "default"
    model_type: str = "default"

    automatic_fix: bool = True

    simulator_config_all: simulator_config = field(default_factory=simulator_config)
    dataloader_config_all: dataloader_config = field(default_factory=dataloader_config)
    model_config_all: model_config = field(default_factory=model_config)

    # Model misc attributes
    model_device: str = "cuda:0"
    model_device_id: int = 0

    # Training attributes
    n_epochs : int = 100
    log_interval: int = 10
    BC_attributes: dict = field(default_factory=dict)


    def __post_init__(self):
        self.SIMULATOR = self.simulator_config_all.configs[self.simulator_type]
        self.DATALOADER = self.dataloader_config_all.configs[self.dataloader_type]
        self.MODEL = self.model_config_all.configs[self.model_type]

        if (self.automatic_fix): 
            self.MODEL.CONTEXT_ENCODER.NUM_INPUT_ATTR_AGENT = 18 + self.DATALOADER.prior_frame
            self.MODEL.MOTION_DECODER.NUM_FUTURE_FRAMES = self.DATALOADER.future_frame
        
        # # Ensure that the model config is compatible with the simulator config
        # assert self.model_config.simulator_type == self.simulator_type, \
        #     f"Model config {self.model_type} is not compatible with simulator type {self.simulator_type}"



    


