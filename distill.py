import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.carla_datamodule import CarlaDataModule
from pl_modules.carla_feat_datamodule import CarlaFeatDataModule
from pl_modules.distillation_module import DistillationModule
from pl_modules.distillation_feat_module import DistillationFeatModule

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="distill", version_base=None)
def main(cfg):
    teacher_ckpt = cfg.teacher_ckpt

    # Setup Result Directory
    result_dir = os.path.join(cfg.project.result_dir, f"distill_{cfg.project.run_name}")
    os.makedirs(result_dir, exist_ok=True)

    # Initialize Data and Distillation Model
    # Initialize the DataModule
    if cfg.data.type == 'carla':
        datamodule = CarlaDataModule(cfg)
        model = DistillationModule(cfg, teacher_ckpt)
    elif cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
        model = DistillationModule(cfg, teacher_ckpt)
    elif cfg.data.type == 'carla_feat':
        datamodule = CarlaFeatDataModule(cfg)
        model = DistillationFeatModule(cfg, teacher_ckpt)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.type}")


    # Initialize logger
    logger = None  # Default to no logger

    # Check if logging with Wandb is enabled in config
    use_wandb = cfg.logging.enable_wandb

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger  # Import here to handle ImportError
            wandb_logger = WandbLogger(
                project=cfg.project.name,
                name=cfg.project.run_name,
                save_dir=result_dir
            )
            logger = wandb_logger
            print("WandbLogger initialized.")
        except ImportError:
            print("Wandb is not installed. Skipping Wandb logging.")



    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(result_dir, 'checkpoints'),
        save_last=True,
        save_top_k=1,
        monitor='val/distill_loss',
    )


    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        trainer = pl.Trainer(
            default_root_dir=result_dir,
            max_epochs=cfg.training.max_epochs,
            logger=logger,  # Pass the logger (WandbLogger or None)
            devices=num_gpu,
            precision=32 if cfg.training.amp else 32,
            accelerator='gpu',
            callbacks=[
                checkpoint_callback,
                pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
            ],
            log_every_n_steps=1,
            strategy=DDPStrategy(find_unused_parameters=False)
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=result_dir,
            max_epochs=cfg.training.max_epochs,
            logger=logger,  # Pass the logger (WandbLogger or None)
            devices=num_gpu,
            precision=32 if cfg.training.amp else 32,
            accelerator='gpu',
            callbacks=[
                checkpoint_callback,
                pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
            ],
            log_every_n_steps=1
        )

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
