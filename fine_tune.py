# fine_tune.py

import pytorch_lightning as pl
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.teleop_datamodule import TeleopDataModule
from pl_modules.citywalker_module import CityWalkerModule
from pl_modules.citywalker_feat_module import CityWalkerFeatModule
import torch

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="goal_agnostic", version_base=None)
def main(cfg):
    checkpoint_path = cfg.checkpoint

    # Create result directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir  # Update result_dir in cfg

    # Save config file in result directory
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    # Initialize the DataModule
    if cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
    elif cfg.data.type == 'teleop':
        datamodule = TeleopDataModule(cfg)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.type}")

    # Initialize the model
    if cfg.model.type == 'citywalker':
        model = CityWalkerModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    elif cfg.model.type == 'citywalker_feat':
        model = CityWalkerFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        raise ValueError(f"Invalid model: {cfg.model.type}")
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(pl.utilities.model_summary.ModelSummary(model, max_depth=2))

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
        monitor='val/direction_loss',
    )

    num_gpu = 1

    # Set up Trainer
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        max_epochs=cfg.training.max_epochs,
        logger=logger,  # Pass the logger (WandbLogger or None)
        devices=num_gpu,
        precision='16-mixed' if cfg.training.amp else 32,
        accelerator='gpu' if num_gpu > 0 else 'cpu',
        callbacks=[
            checkpoint_callback,
            pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
        ],
        log_every_n_steps=1,
    )

    # Start fine-tuning
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
