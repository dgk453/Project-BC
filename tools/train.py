import logging
logging.basicConfig(level=logging.INFO)
import os
import tyro
import torch
import numpy as np
from pathlib import Path
from configs.config import Config
from torch.utils.data import DataLoader
from mtr.models import model as model_utils
from cmtr_bc.waymo_iterator import ProcessedTrajectoryIterator, TrajectoryIterator
from cmtr_bc.bc import BC

def main(config: Config):
    dataloader_cfg = config.DATALOADER
    model_cfg = config.MODEL

    logging.info("(1/3) Create iterator...")

    saved_samples_path = os.getcwd() / Path(dataloader_cfg.data_path)
    data_iterator = ProcessedTrajectoryIterator(saved_samples_path, num_total_samples=dataloader_cfg.num_total_samples,
                                                prefetch_length=20, shuffle=True)
    dataloader = DataLoader(data_iterator, dataloader_cfg.batch_size, collate_fn=TrajectoryIterator.collate_batch,
                            **dataloader_cfg.torch_data_loader_kwarg)

    rng = np.random.default_rng()

    logging.info("(2/3) Initialize model")

    torch.cuda.set_device(config.model_device_id)
    model = model_utils.MotionTransformer(config=model_cfg).to(config.model_device)
    model.set_epoch(0)

    logging.info("(3/3) Beginning Training...")

    bc_trainer = BC(
        policy=model,
        demonstrations=dataloader,
        rng=rng,
        device=torch.device(config.model_device),
        project_name=config.wandb_project_name,
        wandb_on = config.wandb_on,
        wandb_config=config.__dict__,
        optimizer_cfg=config.OPTIMIZATION,
    )

    bc_trainer.train(
        n_epochs=config.OPTIMIZATION.NUM_EPOCHS,
        log_interval=config.log_interval,
        plot_interval=config.plot_interval,
        on_epoch_end=lambda epoch: model.set_epoch(epoch),
    )

    bc_trainer.save(path=config.model_save_path, config=config)

    logging.info("Finished training")


if __name__ == '__main__':
    # print(os.getcwd())
    config = tyro.cli(Config)
    main(config)
