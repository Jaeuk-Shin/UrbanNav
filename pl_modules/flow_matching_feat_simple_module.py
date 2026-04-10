import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.flow_matching_feat_simple import FlowMatchingFeatSimple
from pl_modules.flow_matching_feat_module import FlowMatchingFeatModule
from diffusion_policy.model.diffusion.ema_model import EMAModel


class FlowMatchingFeatSimpleModule(FlowMatchingFeatModule):
    """
    Lightning module for training FlowMatchingFeatSimple (FiLM-conditioned MLP
    velocity field) on precomputed features.

    Inherits visualization, helpers, and test logic from FlowMatchingFeatModule
    but:
      * instantiates FlowMatchingFeatSimple instead of FlowMatchingFeat;
      * maintains an Exponential Moving Average (EMA) copy of the model that
        is used for validation, test, and deployment. EMA decay ramps up via
        the diffusion-policy warmup schedule. Disable with cfg.training.use_ema=false.
    """

    def __init__(self, cfg):
        # Skip the parent __init__ so we don't build the wrong model,
        # but still run the LightningModule initializer.
        pl.LightningModule.__init__(self)
        self.cfg = cfg
        self.model = FlowMatchingFeatSimple(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size

        # --- EMA setup ---------------------------------------------------
        training_cfg = cfg.training
        self.use_ema = bool(getattr(training_cfg, 'use_ema', True))
        if self.use_ema:
            ema_cfg = getattr(training_cfg, 'ema', None)

            def _ema_get(key, default):
                if ema_cfg is None:
                    return default
                return getattr(ema_cfg, key, default)

            # Register the averaged weights as a regular submodule so they
            # are moved to the correct device and saved in the Lightning
            # checkpoint state_dict automatically.
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

            self.ema = EMAModel(
                self.ema_model,
                update_after_step=_ema_get('update_after_step', 0),
                inv_gamma=_ema_get('inv_gamma', 1.0),
                power=_ema_get('power', 0.75),
                min_value=_ema_get('min_value', 0.0),
                max_value=_ema_get('max_value', 0.9999),
            )
        else:
            self.ema_model = None
            self.ema = None

    # ------------------------------------------------------------------
    # Keep EMA copy frozen and in eval mode through Lightning's
    # train()/eval() transitions (parent calls train() on every epoch).
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.ema_model is not None:
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
        return self

    # ------------------------------------------------------------------
    # EMA weight update — called after each optimizer step.
    # ------------------------------------------------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step(self.model)

    # ------------------------------------------------------------------
    # Save / restore the EMA bookkeeping scalars (optimization_step, decay).
    # The EMA weights themselves are already in state_dict via self.ema_model.
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint['ema_optimization_step'] = self.ema.optimization_step
            checkpoint['ema_decay'] = self.ema.decay

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None:
            self.ema.optimization_step = checkpoint.get('ema_optimization_step', 0)
            self.ema.decay = checkpoint.get('ema_decay', 0.0)

    # ------------------------------------------------------------------
    # Validation / test use the EMA copy when available.
    # ------------------------------------------------------------------
    def _eval_model(self):
        return self.ema_model if self.ema_model is not None else self.model

    def validation_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']

        eval_model = self._eval_model()
        wp_pred, v_pred, ut = eval_model(obs_features, cord, batch['waypoints'])
        velocity_loss = F.mse_loss(v_pred, ut)
        self.log('val/velocity_loss', velocity_loss,
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        wp_preds, info = eval_model.sample(obs_features, cord, num_samples=200)
        wp_preds = wp_preds * batch['step_scale'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        self.process_visualization(
            mode='val', batch=batch, wp_pred=wp_preds, noise=info['noise']
        )
        return velocity_loss

    def test_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']

        eval_model = self._eval_model()
        wp_pred, v_pred, ut = eval_model(obs_features, cord, batch['waypoints'])
        velocity_loss = F.mse_loss(v_pred, ut)
        self.log('test/velocity_loss', velocity_loss,
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.process_visualization(mode='test', batch=batch, wp_pred=wp_pred)
        return velocity_loss

    # ------------------------------------------------------------------
    # Only the online model gets gradients. Exclude the EMA copy from the
    # optimizer parameter list so it is not updated via backprop.
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)
        params = self.model.parameters()

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                params, lr=lr, weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                params, lr=lr, weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_cfg = self.cfg.scheduler
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.gamma,
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.training.max_epochs,
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")
