from dataclasses import dataclass, field
from typing import List, Dict
from configs.model.mask_predictor import mask_predictor_config
from configs import BaseConfig

@dataclass
class default_motion_decoder_config(BaseConfig): 
    NAME: str = "MTRDecoder"

    OBJECT_TYPE: List[str] = field(default_factory=lambda: ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'])
    CENTER_OFFSET_OF_MAP: List[float] = field(default_factory=lambda: [30.0, 0])

    NUM_FUTURE_FRAMES: int = 73
    NUM_MOTION_MODES: int = 6

    INTENTION_POINTS_FILE: str = "data/waymo/cluster_64_center_dict.pkl"

    D_MODEL: int = 256
    NUM_DECODER_LAYERS: int = 6
    NUM_ATTN_HEAD: int = 8
    MAP_D_MODEL: int = 256
    DROPOUT_OF_ATTN: float = 0.1

    NUM_BASE_MAP_POLYLINES: int = 256
    NUM_WAYPOINT_MAP_POLYLINES: int = 128

    LOSS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'cls': 1.0,
        'reg': 1.0,
        'vel': 0.5
    })

    NMS_DIST_THRESH: float = 2.5

    mask_predictor_all: mask_predictor_config = field(default_factory=mask_predictor_config)

    def __post_init__(self):
        self.MASK_PREDICTOR = self.mask_predictor_all.configs["default"]
