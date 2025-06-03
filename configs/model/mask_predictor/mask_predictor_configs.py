from dataclasses import dataclass, field
from typing import Optional
from configs import BaseConfig

@dataclass
class EncoderMaskConfig(BaseConfig):
    USE_KNN_ATTN_MASK: bool = True
    ONLY_OBJ_AS_QUERY: bool = True

@dataclass
class EntmaxAlphaSchedule(BaseConfig):
    schedule: str = "linear"  # "constant", "linear", "exponential"
    initial_value: float = 1.3
    final_value: float = 1.5
    start_epoch: int = 10
    final_epoch: int = 20


@dataclass
class EntmaxConfig(BaseConfig):
    alpha_learnable: bool = False
    use_holdout: bool = False
    straight_through: bool = True
    alpha: EntmaxAlphaSchedule = field(default_factory=EntmaxAlphaSchedule)


@dataclass
class BernoulliCoefSchedule(BaseConfig):
    schedule: str = "linear"
    initial_value: float = 0.0
    final_value: float = 0.01
    start_epoch: int = 10
    final_epoch: int = 30


@dataclass
class BernoulliRegularization(BaseConfig):
    use_masked_attn_score: bool = False
    use_supervision_loss: bool = False
    threshold: float = 0.1
    coef: BernoulliCoefSchedule = field(default_factory=BernoulliCoefSchedule)


@dataclass
class BernoulliConfig(BaseConfig):
    regularization: BernoulliRegularization = field(default_factory=BernoulliRegularization)


@dataclass
class ThresholdConfig(BaseConfig):
    threshold: float = 0.2
    dynamic_threshold: bool = False
    straight_through: bool = True


@dataclass
class SparsificationConfig(BaseConfig):
    type: str = "bernoulli"  # "topk", "entmax", "threshold", "bernoulli"
    use_mask_label_final_epoch: int = 20
    entmax_config: EntmaxConfig = field(default_factory=EntmaxConfig)
    bernoulli_config: BernoulliConfig = field(default_factory=BernoulliConfig)
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)


@dataclass
class AttnConfig(BaseConfig):
    embed_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.0
    mlp_hidden_dim: int = 256


@dataclass
class CompetitionConfig(BaseConfig):
    iters: int = 2
    use_rnn: bool = False
    update_key: bool = False


@dataclass
class DefaultMaskPredictorModelConfig(BaseConfig):
    attn_config: AttnConfig = field(default_factory=AttnConfig)
    competition_config: CompetitionConfig = field(default_factory=CompetitionConfig)
    sparsification_config: SparsificationConfig = field(default_factory=SparsificationConfig)


@dataclass
class default_mask_predictor_config(BaseConfig):
    USE_MASK_PREDICTOR: bool = True
    ENCODER_MASK_PREDICTOR_CFG_FNAME: Optional[str] = None
    USE_FJMP_LABELS: bool = False
    ENCODER: EncoderMaskConfig = field(default_factory=EncoderMaskConfig)
    ENC_DEC_USE_SAME_MASK: bool = True
    MODEL: DefaultMaskPredictorModelConfig = field(default_factory=DefaultMaskPredictorModelConfig)
