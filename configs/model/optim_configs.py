from dataclasses import dataclass, field
from configs import BaseConfig

@dataclass
class optim_config(BaseConfig): 
    BATCH_SIZE_PER_GPU: int = 10
    NUM_EPOCHS: int = 30

    OPTIMIZER: str = "AdamW"
    LR: float = 0.0001
    WEIGHT_DECAY: float = 0.01

    SCHEDULER: str = "lambdaLR"
    DECAY_STEP_LIST: list = field(default_factory=lambda: [22, 24, 26, 28]) 
    LR_DECAY: int = 0.5
    LR_CLIP: float = 0.000001

    GRAD_NORM_CLIP: float = 5.0