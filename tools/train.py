import os
import tyro
import torch
import logging
import numpy as np
from pathlib import Path
from configs.config import Config
from torch.utils.data import DataLoader
from mtr.models import model as model_utils
from cmtr_bc.waymo_iterator import ProcessedTrajectoryIterator, TrajectoryIterator
from cmtr_bc.bc import BC

def main(config: Config): 
    logger = logging.getLogger(__name__)
    dataloader_cfg = config.DATALOADER   
    model_cfg = config.MODEL 

    logging.info("(1/3) Create iterator...")

    saved_samples_path = os.getcwd() / Path(dataloader_cfg.data_path) 
    data_iterator = ProcessedTrajectoryIterator(saved_samples_path)
    dataloader = DataLoader(data_iterator, dataloader_cfg.batch_size, collate_fn=TrajectoryIterator.collate_batch, 
                            **dataloader_cfg.torch_data_loader_kwarg)

    rng = np.random.default_rng()

    logging.info("(2/3) Initialize model")

    torch.cuda.set_device(config.model_device_id)
    model = model_utils.MotionTransformer(config=model_cfg).to(config.model_device)
    model.set_epoch(0)

    logging.info("(3/3) Beginngin Training...")

    bc_trainer = BC(
        policy=model,
        demonstrations=dataloader,
        rng=rng,
        device=torch.device(config.model_device),
    )

    # def _on_epoch_end(epoch): 
    #     func_list = [model.set_epoch]

    # Train
    bc_trainer.train(
        n_batches=config.n_epochs,
        log_interval=config.log_interval,
        on_epoch_end=lambda epoch: model.set_epoch(epoch),
    )

    logging.info("Finished training")


if __name__ == '__main__':
    # print(os.getcwd())
    config = tyro.cli(Config)
    main(config)
