import tyro
from configs.config import Config
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from cmtr_bc.waymo_dataset import WaymoDataset
import logging
from cmtr_bc.waymo_iterator import TrajectoryIterator
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from tqdm import tqdm
import os

def main(config: Config): 
    logger = logging.getLogger(__name__)
    sim_cfg = config.SIMULATOR; 
    dataloader_cfg = config.DATALOADER
    gpudrive_env_cfg = EnvConfig(dynamics_model=sim_cfg.dynamics_model)
    gpudrive_scene_loader = SceneDataLoader(
        root=sim_cfg.data_path,
        batch_size=sim_cfg.num_envs,
        dataset_size=sim_cfg.dataset_size,
        sample_with_replacement=sim_cfg.sample_with_replacement,
    )

    logger.info("Making the environment")
    env = GPUDriveTorchEnv(
        config=gpudrive_env_cfg,
        data_loader=gpudrive_scene_loader,
        max_cont_agents=sim_cfg.max_num_object, # Maximum number of agents to control per scenario
        device="cuda", 
        action_type="continuous" # "continuous" or "discrete"
    )

    waymo_dataset = WaymoDataset(dataloader_cfg, test_mode=1, logger=logger)
    traj_iterator = iter(
        TrajectoryIterator(env, gpudrive_scene_loader, cmtr=True, waymo_dataset=waymo_dataset, 
                                          prior_frame=dataloader_cfg.prior_frame, future_frame=dataloader_cfg.future_frame, 
                                          simple=False)
    )
    # train_iterator = iter(DataLoader(traj_iterator, batch_size=dataloader_cfg.batch_size, collate_fn=traj_iterator.collate_batch))
    
    n_samples = dataloader_cfg.num_total_samples
    n_samples_per_file = dataloader_cfg.num_samples_per_file

    logging.info("Beginning to process & save samples")

    save_path = os.getcwd() / Path(dataloader_cfg.data_path)
    os.makedirs(save_path, exist_ok=True)

    saved_samples = []
    idx = 0
    for idx in tqdm(range(n_samples)):
        samples = next(traj_iterator)
        saved_samples.append(samples)
        if (len(saved_samples) == n_samples_per_file): 
            torch.save(saved_samples, save_path / f"saved_samples_{(idx+1):04d}.pt")
            saved_samples = []

    if (len(saved_samples) > 0): 
        torch.save(saved_samples, save_path / f"saved_samples_{(idx+1):04d}.pt")

    logging.info("Finished saving batches")


if __name__ == '__main__':
    # print(os.getcwd())
    config = tyro.cli(Config)
    main(config)
