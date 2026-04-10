# main.py

import pytorch_lightning as pl
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.teleop_datamodule import TeleopDataModule
from pl_modules.citywalker_module import CityWalkerModule
from pl_modules.citywalker_feat_module import CityWalkerFeatModule
from pl_modules.carla_datamodule import CarlaDataModule
from pl_modules.carla_feat_datamodule import CarlaFeatDataModule
from pl_modules.citywalk_feat_datamodule import CityWalkFeatDataModule
from pl_modules.urban_nav_feat_mixture_datamodule import UrbanNavFeatMixtureDataModule
from pytorch_lightning.strategies import DDPStrategy
import torch
import glob
torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

from pl_modules.citywalker_crop_module import CityWalkerCropModule
from pl_modules.citywalk_crop_datamodule import CityWalkCropDataModule
from pl_modules.flow_matching_module import FlowMatchingModule
from pl_modules.flow_matching_feat_module import FlowMatchingFeatModule
from pl_modules.flow_matching_feat_simple_module import FlowMatchingFeatSimpleModule

import hydra
from omegaconf import OmegaConf


def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint in the given directory based on modification time.

    Args:
        checkpoint_dir (str): Path to the directory containing checkpoints.

    Returns:
        str: Path to the latest checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    print(checkpoint_dir)
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")

    # Sort checkpoints by modification time (latest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return latest_checkpoint

@hydra.main(config_path="config", config_name="urban_nav", version_base=None)
def main(cfg):
    # Create result directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir  # Update result_dir in cfg

    # Save config file in result directory
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    # Initialize the DataModule
    if cfg.data.type == 'carla':
        datamodule = CarlaDataModule(cfg)
    elif cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
    elif cfg.data.type == 'teleop':
        datamodule = TeleopDataModule(cfg)
    elif cfg.data.type == 'citywalk_crop':
        datamodule = CityWalkCropDataModule(cfg)
    elif cfg.data.type == 'carla_feat':
        datamodule = CarlaFeatDataModule(cfg)
    elif cfg.data.type == 'citywalk_feat':
        datamodule = CityWalkFeatDataModule(cfg)
    elif cfg.data.type == 'urban_nav_feat_mixture':
        datamodule = UrbanNavFeatMixtureDataModule(cfg)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

    # Initialize the model
    if cfg.model.type == 'flow_matching':
        model = FlowMatchingModule(cfg)
    elif cfg.model.type == 'citywalker':
        model = CityWalkerModule(cfg)
    elif cfg.model.type == 'citywalker_crop':
        model = CityWalkerCropModule(cfg)
    elif cfg.model.type == 'citywalker_feat':
        model = CityWalkerFeatModule(cfg)
    elif cfg.model.type == 'flow_matching_feat':
        model = FlowMatchingFeatModule(cfg)
    elif cfg.model.type == 'flow_matching_feat_simple':
        model = FlowMatchingFeatSimpleModule(cfg)
    elif cfg.model.type == 'citywalker_fm':
        model = FlowMatchingModule(cfg)
    else:
        raise ValueError(f"Invalid model: {cfg.model.type}")
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
        monitor='val/velocity_loss',
    )

    num_gpu = torch.cuda.device_count()
    # num_gpu = 1

    # Set up Trainer
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
            strategy=DDPStrategy(find_unused_parameters=True)
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
            log_every_n_steps=1,
        )

    checkpoint_path = cfg.get("checkpoint", None)

    if cfg.training.resume:
        # Determine the checkpoint path
        try:
            if checkpoint_path:
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            else:
                # Automatically find the latest checkpoint
                checkpoint_dir = os.path.join(cfg.project.result_dir, 'checkpoints')
                if not os.path.isdir(checkpoint_dir):
                    raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
                checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError()
                else:
                    print(f"No checkpoint specified. Using the latest checkpoint: {checkpoint_path}")
            print(f"Training resume from checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            print("No checkpoint found. Training from scratch.")
            checkpoint_path = None
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        # Start training
        trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
