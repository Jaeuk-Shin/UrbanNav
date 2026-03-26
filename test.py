# test.py

import pytorch_lightning as pl
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.teleop_datamodule import TeleopDataModule
from pl_modules.citywalker_module import CityWalkerModule
from pl_modules.citywalker_feat_module import CityWalkerFeatModule
import torch
import glob

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

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
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")

    # Sort checkpoints by modification time (latest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return latest_checkpoint


@hydra.main(config_path="config", config_name="goal_agnostic", version_base=None)
def main(cfg):
    # Create a directory for test results
    test_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # Initialize the DataModule
    if cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
    elif cfg.data.type == 'teleop':
        datamodule = TeleopDataModule(cfg)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

    checkpoint_path = cfg.get("checkpoint", None)

    # Determine the checkpoint path
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    else:
        # Automatically find the latest checkpoint
        checkpoint_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        print(f"No checkpoint specified. Using the latest checkpoint: {checkpoint_path}")

    # Load the model from the checkpoint
    if cfg.model.type == 'citywalker':
        model = CityWalkerModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    elif cfg.model.type == 'citywalker_feat':
        model = CityWalkerFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        raise ValueError(f"Invalid model: {cfg.model.type}")
    model.result_dir = test_dir
    print(f"Loaded model from checkpoint: {checkpoint_path}")

    # Initialize Trainer
    trainer = pl.Trainer(
        default_root_dir=test_dir,
        devices=cfg.training.gpus,
        precision='bf16-mixed' if cfg.training.amp else 32,
        accelerator='ddp' if cfg.training.gpus > 1 else 'gpu',
        logger=False
        # You can add more Trainer arguments if needed
    )

    # Run testing
    trainer.test(model, datamodule=datamodule, verbose=True)


if __name__ == '__main__':
    main()
