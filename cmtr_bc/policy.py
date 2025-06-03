import torch
import torch.nn as nn
from typing import Iterable, Tuple, Union, Any

class BCPolicy(nn.Module):
    """
    A policy network for Behavioral Cloning.
    """
    def __init__(self, observation_space: Any, action_space: Any, module: Any):
        """
        Initializes the policy network.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.module = module

    @property
    def device(self):
        return next(self.parameters()).device

    def evaluate_actions(self, obs: torch.Tensor, acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        """
        Evaluates the log probabilities and entropy of given actions under the policy.

        Args:
            obs: The observations.
            acts: The actions.

        Returns:
            A tuple containing the mean, log probabilities, and entropy of the actions.
        """
        distribution = self.module.dist(obs)
        log_prob = self.module.log_prob(distribution, acts)
        entropy = self.module.entropy(distribution)
        # logging.info(f"Action: {action}")
        return log_prob, entropy

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Predicts an action given an observation.

        Args:
            observation: The observation.
            deterministic: Whether to return a deterministic action.

        Returns:
            The predicted action.
        """
        # Returns the predicted action
        return self.module(observation, deterministic)