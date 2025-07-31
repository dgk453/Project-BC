import dataclasses
import itertools
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import numpy as np
import tqdm
from stable_baselines3.common import utils, vec_env
from imitation.algorithms import base as algo_base
from imitation.data import rollout, types
import wandb
import torch
import yaml
from pathlib import Path
from configs.model.optim_configs import optim_config
from cmtr_bc.batch_dict_visualization import plot_scenario


@dataclasses.dataclass(frozen=True)
class BatchIteratorWithEpochEndCallback:
    """Loops through batches from a batch loader and calls a callback after every epoch.

    Will throw an exception when an epoch contains no batches.
    """

    batch_loader: torch.utils.data.DataLoader
    n_epochs: Optional[int]
    n_batches: Optional[int]
    on_epoch_end: Optional[Callable[[int], None]]

    def __post_init__(self) -> None:
        epochs_and_batches_specified = (
            self.n_epochs is not None and self.n_batches is not None
        )
        neither_epochs_nor_batches_specified = (
            self.n_epochs is None and self.n_batches is None
        )
        if epochs_and_batches_specified or neither_epochs_nor_batches_specified:
            raise ValueError(
                "Must provide exactly one of `n_epochs` and `n_batches` arguments.",
            )

    def __iter__(self) -> Iterator[types.TransitionMapping]:
        def batch_iterator() -> Iterator[types.TransitionMapping]:
            # Note: the islice here ensures we do not exceed self.n_epochs
            for epoch_num in itertools.islice(itertools.count(), self.n_epochs):
                some_batch_was_yielded = False
                for batch in self.batch_loader:
                    yield batch
                    some_batch_was_yielded = True

                if not some_batch_was_yielded:
                    raise AssertionError(
                        f"Data loader returned no data during epoch "
                        f"{epoch_num} -- did it reset correctly?",
                    )
                if self.on_epoch_end is not None:
                    self.on_epoch_end(epoch_num)

        # Note: the islice here ensures we do not exceed self.n_batches
        return itertools.islice(batch_iterator(), self.n_batches)

def make_tensor_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_tensor_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_tensor_safe(x) for x in obj]
    else:
        return obj

@dataclasses.dataclass(frozen=True)
class BCTrainingMetrics:
    """Container for the different components of behavior cloning loss."""
    loss: torch.Tensor
    tb_dict: dict
    disp_dict: dict
    batch_dict: dict
    l2_norm: torch.Tensor
    log_enable: list = dataclasses.field(
        default_factory=lambda: ["loss", "tb_dict", "disp_dict", "l2_norm"]
    )



@dataclasses.dataclass(frozen=True)
class BehaviorCloningLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float

    def __call__(
        self,
        policy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, torch.Tensor],
        ],
    ) -> BCTrainingMetrics:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        # tensor_obs = types.map_maybe_dict(
        #     util.safe_to_tensor,
        #     types.maybe_unwrap_dictobs(obs),
        # )

        loss, tb_dict, disp_dict, batch_dict = policy(obs)
        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # l2_loss = self.l2_weight * l2_norm
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)
        # return policy(obs)
        return BCTrainingMetrics(
            loss=loss,
            tb_dict=tb_dict,
            disp_dict=disp_dict,
            batch_dict=batch_dict,
            l2_norm=l2_norm,
        )

def enumerate_batches(
    batch_it: torch.utils.data.DataLoader,
) -> Iterable[Tuple[Tuple[int, int, int], types.TransitionMapping]]:
    """Prepends batch stats before the batches of a batch iterator."""
    num_samples_so_far = 0
    for num_batches, batch in enumerate(batch_it):
        batch_size = batch['batch_size']
        num_samples_so_far += batch_size
        yield (num_batches, batch_size, num_samples_so_far), batch


@dataclasses.dataclass(frozen=True)
class RolloutStatsComputer:
    """Computes statistics about rollouts.

    Args:
        venv: The vectorized environment in which to compute the rollouts.
        n_episodes: The number of episodes to base the statistics on.
    """

    venv: Optional[vec_env.VecEnv]
    n_episodes: int

    # TODO(shwang): Maybe instead use a callback that can be shared between
    #   all algorithms' `.train()` for generating rollout stats.
    #   EvalCallback could be a good fit:
    #   https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback

    def __call__(
        self,
        policy,
        rng: np.random.Generator,
    ) -> Mapping[str, float]:
        if self.venv is not None and self.n_episodes > 0:
            trajs = rollout.generate_trajectories(
                policy,
                self.venv,
                rollout.make_min_episodes(self.n_episodes),
                rng=rng,
            )
            return rollout.rollout_stats(trajs)
        else:
            return dict()



class BCLogger:
    """Utility class to help logging information relevant to Behavior Cloning."""

    def __init__(self, project_name: str, config: Optional[Dict[str, Any]] = None, wandb_on = True, **wandb_kwargs):
        """Create new BC logger with wandb initialization.

        Args:
            project_name: The wandb project name.
            config: Optional configuration dictionary to log to wandb.
            **wandb_kwargs: Additional keyword arguments passed to wandb.init().
        """
        if (wandb_on):
            wandb.init(project=project_name, config=config, **wandb_kwargs)
        self.wandb_on = wandb_on
        self._step = 0
        self._current_epoch = 0

    def reset_steps(self):
        self._step = 0

    def log_epoch(self, epoch_number):
        self._current_epoch = epoch_number

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        num_samples_so_far: int,
        training_metrics: BCTrainingMetrics,
        rollout_stats: Mapping[str, float],
    ):
        # Prepare wandb log dict
        wandb_log = {
            "batch_size": batch_size,
            "bc/epoch": self._current_epoch,
            "bc/batch": batch_num,
            "bc/samples_so_far": num_samples_so_far,
        }

        # Log training metrics
        for k in training_metrics.log_enable:
            wandb_log[f"bc/{k}"] = make_tensor_safe(getattr(training_metrics, k, None))

        # Log rollout stats
        for k, v in rollout_stats.items():
            if "return" in k and "monitor" not in k:
                wandb_log["rollout/" + k] = v

        # Log to wandb
        if (self.wandb_on):
            wandb.log(wandb_log, step=self._step)

        self._step += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        return state


class BC(algo_base.DemonstrationAlgorithm):
    """Custom Behavioral cloning (BC) --> Hijacked imitation's BC for cmtr
    Recovers a policy via supervised learning from observation-action pairs.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        policy: torch.nn.Module,
        demonstrations: torch.utils.data.DataLoader,
        optimizer_cfg: optim_config = optim_config(),
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, torch.device] = "auto",
        project_name: str = "BC",
        wandb_config = None,
        wandb_on: bool = True,
        **wandb_kwargs
    ):
        """Builds BC.

        Args:
            rng: the random state to use for the random number generator.
            policy: a Stable Baselines3 policy; if unspecified,
                defaults to `FeedForward32Policy`.
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead), or if the batch size is not a multiple
                of the minibatch size.
        """
        self._demo_data_loader: Optional[torch.utils.data.DataLoader] = demonstrations
        super().__init__(
            demonstrations=demonstrations,
        )
        self._bc_logger = BCLogger(project_name, wandb_config, wandb_on, **wandb_kwargs)
        self.rng = rng
        self._policy = policy.to(utils.get_device(device))
        self.optimizer_cfg = optimizer_cfg
        self.build_optimizer()
        self.loss_calculator = BehaviorCloningLossCalculator(ent_weight, l2_weight)

    def build_optimizer(self) :
        opt_cfg = self.optimizer_cfg
        if opt_cfg.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(
                [each[1] for each in self.policy.named_parameters()],
                lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            )
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(self.policy.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0))
        else:
            assert False
        self.optimizer = optimizer

    def build_scheduler(self, total_epochs):
        optimizer = self.optimizer
        dataloader = self._demo_data_loader
        opt_cfg = self.optimizer_cfg
        decay_epochs = opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])
        total_iters_each_epoch = len(self._demo_data_loader.dataset)
        last_epoch = opt_cfg.get('LAST_EPOCH', -1)
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_epoch in decay_epochs:
                if cur_epoch >= decay_epoch:
                    cur_decay = cur_decay * opt_cfg.LR_DECAY
            return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

        if opt_cfg.get('SCHEDULER', None) == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=2 * len(dataloader),
                T_mult=1,
                eta_min=max(1e-2 * opt_cfg.LR, 1e-6),
                last_epoch=-1,
            )
        elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
        elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
            total_iters = total_iters_each_epoch * total_epochs
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=opt_cfg.LR_CLIP / opt_cfg.LR,
                                        total_iters=total_iters, last_epoch=last_epoch)
        else:
            scheduler = None

        self.scheduler = scheduler


    @property
    def policy(self) -> torch.nn.Module:
        return self._policy

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        assert(self._demo_data_loader is not None)

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        num_batch_in_mini_batch: Optional[int] = 1,
        on_epoch_end: Optional[Callable[[int], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        plot_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_logger: bool = True,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`. Note, that when you specify
        `n_batches` smaller than the number of batches in an epoch, the `on_epoch_end`
        callback will never be called.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            on_batch_end: Optional callback with no parameters to run at the end of each
                batch.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode length. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_logger: If True, then start plotting to wandb from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """
        if reset_logger:
            self._bc_logger.reset_steps()
        self._bc_logger.log_epoch(0)

        self.build_scheduler(n_epochs)

        compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(
                    f"Epoch {epoch_number} {total_num_epochs_str}",
                    pos=1,
                )
            self._bc_logger.log_epoch(epoch_number + 1)
            if on_epoch_end is not None:
                on_epoch_end(epoch_number)

        # n_minibatches = n_batches * mini_per_batch if n_batches is not None else None

        assert self._demo_data_loader is not None
        demonstration_batches = BatchIteratorWithEpochEndCallback(
            self._demo_data_loader,
            n_epochs,
            None,
            _on_epoch_end,
        )
        batches_with_stats = enumerate_batches(demonstration_batches)
        tqdm_progress_bar: Optional[tqdm.tqdm] = None

        if progress_bar:
            batches_with_stats = tqdm.tqdm(
                batches_with_stats,
                unit="batch",
                total=n_epochs * len(self._demo_data_loader.dataset) // self._demo_data_loader.batch_size,
                dynamic_ncols=True
            )
            tqdm_progress_bar = batches_with_stats

        def process_batch():

            # clip gradients
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.optimizer_cfg.GRAD_NORM_CLIP)

            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch_num % log_interval == 0:
                rollout_stats = compute_rollout_stats(self.policy, self.rng)
                self._bc_logger.log_batch(
                    batch_num,
                    minibatch_size,
                    num_samples_so_far,
                    training_metrics,
                    rollout_stats,
                )
            if self.scheduler is not None:
                self.scheduler.step()

            if on_batch_end is not None:
                on_batch_end()

            torch.cuda.empty_cache()

        def visualize():
            if batch_num % plot_interval == plot_interval - 1 and self._bc_logger.wandb_on:
                output_dict = training_metrics.batch_dict
                figs = plot_scenario(
                    input_dict=output_dict['input_dict'],
                    forward_ret_dict=output_dict['forward_ret_dict'],
                    num_samples=5,
                    plot_object_history=True,
                    plot_object_gt_future=True,
                    plot_ego_object_pred_future=True,
                )
                for fig in figs:
                    if (self._bc_logger.wandb_on):
                        wandb.log({"plot": wandb.Image(fig)}, self._bc_logger._step)
                    self._bc_logger._step += 1


        self.optimizer.zero_grad()
        for (
            batch_num,
            minibatch_size,
            num_samples_so_far,
        ), batch in batches_with_stats:
            # obs_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]
            # # unwraps the observation if it's a dictobs and converts arrays to tensors
            # obs_tensor = types.map_maybe_dict(
            #     lambda x: util.safe_to_tensor(x, device=self.policy.device),
            #     types.maybe_unwrap_dictobs(batch["obs"]),
            # )
            training_metrics = self.loss_calculator(self.policy, batch)
            # Renormalise the loss to be averaged over the whole
            # batch size instead of the minibatch size.
            # If there is an incomplete batch, its gradients will be
            # smaller, which may be helpful for stability.
            loss = training_metrics.loss * 1 / num_batch_in_mini_batch
            loss.backward()
            process_batch()
            visualize()

    def save(self, path, config=None):
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), path / "model.pth")
        if (config is not None): 
            with open(path / "config.yaml", "w") as f: 
                yaml.dump(dataclasses.asdict(config), f)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
