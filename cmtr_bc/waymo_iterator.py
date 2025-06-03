"""
Description: Imitation-compatible (https://imitation.readthedocs.io/)
iterator for generating expert trajectories in Waymo scenes.
"""
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from cmtr_bc.imitation_data_generation import generate_state_action_pairs
from cmtr_bc.waymo_dataset import WaymoDataset
from imitation.data.types import Trajectory
import logging
import os
import torch

# Global setting
logging.basicConfig(level="DEBUG")
from torch.utils.data import IterableDataset


class TrajectoryIterator(IterableDataset):
    def __init__(self, env : GPUDriveTorchEnv, scene_data_loader : SceneDataLoader, cmtr=False, waymo_dataset=None, prior_frame=None, future_frame=None, simple=False):
        """Imitation-compatible iterator for generating expert trajectories in Waymo scenes.
        Args:
            env (BaseEnv): Environment to use for generating trajectories.
            - SceneLoader is required: This determines the scenes + batch size
        """
        self.env = env
        self.scene_data_loader = scene_data_loader
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.cmtr = cmtr
        self.waymo_dataset = waymo_dataset
        self.future_frame = future_frame
        self.prior_frame = prior_frame
        self.simple = simple

    def __iter__(self):
        """Return an (expert_state, expert_action) iterable."""
        return iter(self._get_trajectories()) #this returns a batch
    
    def collate_batch(batch_data): 
        observations = []
        for obs in batch_data: 
            observations.append(obs)

        return WaymoDataset.collate_batch(None, observations)

    
    def _get_trajectories(self):
        """Load scenes, preprocess and return trajectories.""" 
        while True: 
            for batch in self.scene_data_loader: 
                self.env.swap_data_batch(batch)
                if (not self.cmtr): 
                    expert_obs, expert_acts, expert_next_obs, expert_dones = generate_state_action_pairs(env=self.env, 
                                                                                                        device="cuda", 
                                                                                                        action_space_type="continuous", 
                                                                                                        use_action_indices=True,  
                                                                                                        make_video=False,
                                                                                                        render_index=[2, 0],
                                                                                                        save_path="use_discr_actions_fix"
                                                                                                        ) 
                    for obs, act, next_obs, done in zip(expert_obs, expert_acts, expert_next_obs, expert_dones):
                        yield(obs, act, next_obs, done)
                else: 
                    observations = generate_state_action_pairs(env=self.env, 
                                                       device="cuda", 
                                                       action_space_type="continuous", 
                                                       use_action_indices=True,  
                                                       make_video=False,
                                                       render_index=[2, 0],
                                                       save_path="use_discr_actions_fix",
                                                       CMTR=True,
                                                       waymo_dataset=self.waymo_dataset,
                                                       prior_frame=self.prior_frame, 
                                                       future_frame=self.future_frame,
                                                       simple=self.simple
                                                       ) 
                    # return individual outputs of waymo_dataset --> collate batches later
                    for obs in observations:  
                        yield obs

        '''
        Notes: 
            return expert_obs, expert_acts, expert_next_obs, expert_dones
            this thing above is going to be easy to implement
            self._step_through_scene --> "generate_state_action_pairs"[:-2]
            go through cmtr preprocessing stuff --> info
            for the aggregator, change the shapes and do whatever
        '''


class ProcessedTrajectoryIterator(IterableDataset): 
    def __init__(self, root):
        self.data_path = root
        self.files = []
        try:
            self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))])
        except: 
            raise(f"Directory now found: {root}")
        
    def __iter__(self): 
        # Creating an infinite dataset
        for file in self.files: 
            saved_samples = torch.load(file, weights_only=False)
            for sample in saved_samples: 
                yield sample