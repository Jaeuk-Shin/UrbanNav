"""Validation-only script — runs validation and saves visualizations without training.

Usage:
    python validate.py --config config/goal_agnostic_fm.yaml --checkpoint path/to/best.ckpt
    python validate.py --config config/goal_agnostic_fm.yaml --checkpoint path/to/best.ckpt \
        model.type=flow_matching_feat data.type=carla_feat
    python validate.py --config config/goal_agnostic_fm.yaml --checkpoint path/to/best.ckpt \
        validation.num_visualize=100

Outputs are saved to {result_dir}/val_vis/epoch_0/.
"""

import os
import sys

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.carla_datamodule import CarlaDataModule
from pl_modules.carla_feat_datamodule import CarlaFeatDataModule
from pl_modules.citywalk_feat_datamodule import CityWalkFeatDataModule
from pl_modules.urban_nav_feat_mixture_datamodule import UrbanNavFeatMixtureDataModule
from pl_modules.flow_matching_module import FlowMatchingModule
from pl_modules.flow_matching_feat_module import FlowMatchingFeatModule

import hydra
from omegaconf import OmegaConf


def build_datamodule(cfg):
    constructors = {
        'carla': CarlaDataModule,
        'citywalk': CityWalkDataModule,
        'carla_feat': CarlaFeatDataModule,
        'citywalk_feat': CityWalkFeatDataModule,
        'urban_nav_feat_mixture': UrbanNavFeatMixtureDataModule,
    }
    if cfg.data.type not in constructors:
        raise ValueError(f"Unsupported data type for validation: {cfg.data.type}")
    return constructors[cfg.data.type](cfg)


def load_model(cfg, checkpoint_path):
    if cfg.model.type in ('flow_matching', 'citywalker_fm'):
        model = FlowMatchingModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    elif cfg.model.type == 'flow_matching_feat':
        model = FlowMatchingFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        raise ValueError(f"Unsupported model type for validation: {cfg.model.type}")
    return model


@hydra.main(config_path="config", config_name="urban_nav", version_base=None)
def main(cfg):
    checkpoint_path = cfg.get("checkpoint", None)
    if not checkpoint_path:
        print("Error: --checkpoint is required. Usage:")
        print("  python validate.py --config config/goal_agnostic_fm.yaml --checkpoint path/to/best.ckpt")
        sys.exit(1)
    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Set up output directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {result_dir}/val_vis/epoch_0/")
    print(f"Model type : {cfg.model.type}")
    print(f"Data type  : {cfg.data.type}")
    print(f"Num vis    : {cfg.validation.num_visualize}")

    datamodule = build_datamodule(cfg)
    model = load_model(cfg, checkpoint_path)
    model.eval()

    print(pl.utilities.model_summary.ModelSummary(model, max_depth=2))

    num_gpu = torch.cuda.device_count()
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        devices=max(num_gpu, 1),
        accelerator='gpu' if num_gpu > 0 else 'cpu',
        logger=False,
        enable_checkpointing=False,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=10)],
    )

    results = trainer.validate(model, datamodule=datamodule)

    print("\n--- Validation Results ---")
    for k, v in results[0].items():
        print(f"  {k}: {v:.6f}")
    print(f"\nVisualizations saved to: {result_dir}/val_vis/epoch_0/")


if __name__ == '__main__':
    main()
