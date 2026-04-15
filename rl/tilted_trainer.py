"""
Train an exponential-tilted noise proposal for the distilled trajectory model.

Learns q_phi(z | ctx) = N(mu_phi(ctx), sigma_phi^2 I) in the noise space of a
frozen distilled model, such that f(ctx, z), z ~ q_phi, samples from

    p_tilted(x | ctx) proportional to exp(-beta R(x)) p_base(x | ctx)

where R is the MPC optimal tracking cost under unicycle dynamics.

Usage:
    python -m rl.tilted_trainer \
        --config config/distill.yaml \
        --distill_ckpt path/to/distill/checkpoints/last.ckpt \
        --beta 0.1 --num_samples 8 --max_epochs 5

See rl/docs/exponential_tilted_model.md for full derivation.
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from rl.models.noise_proposal import NoiseProposal
from rl.envs.mpc.mpc import MPC


# ── MPC evaluation helpers ──────────────────────────────────────────────


def _solve_single(wp_np, mpc_horizon, dt, n_skips, max_speed):
    """Solve MPC for a single waypoint set (runs in worker process)."""
    ulb = np.array([-max_speed, -0.8])
    uub = np.array([ max_speed,  0.8])
    mpc = _solve_single._mpc
    if mpc is None:
        mpc = MPC(mpc_horizon, dt, ulb, uub, max_wall_time=2.0 * dt)
        _solve_single._mpc = mpc

    wp_up = np.repeat(wp_np, n_skips, axis=0)[:mpc_horizon]
    cw = np.zeros(mpc_horizon)
    cw[n_skips-1::n_skips] = 1.0
    initial_pose = np.array([0.0, 0.0, np.pi / 2.0])
    try:
        _, _, stats = mpc.solve(initial_pose, wp_up, cw)
        return stats["optimal_cost"]
    except Exception:
        return 1e6  # fallback for infeasible problems


def _init_worker(mpc_horizon, dt, n_skips, max_speed):
    """Initialise per-worker MPC solver (JIT compiled once per process)."""
    ulb = np.array([-max_speed, -0.8])
    uub = np.array([max_speed, 0.8])
    _solve_single._mpc = MPC(mpc_horizon, dt, ulb, uub, max_wall_time=2.0 * dt)


_solve_single._mpc = None  # placeholder for the per-process solver


def evaluate_mpc_batch(wp_batch_np, mpc_horizon, dt, n_skips, max_speed,
                       pool=None):
    """
    Evaluate MPC cost for a batch of waypoint sets.

    Parameters
    ----------
    wp_batch_np : (N, T, 2) numpy array
    pool : multiprocessing.Pool or None (sequential if None)

    Returns
    -------
    costs : (N,) numpy array
    """
    fn = partial(_solve_single, mpc_horizon=mpc_horizon, dt=dt,
                 n_skips=n_skips, max_speed=max_speed)
    wps = [wp_batch_np[i] for i in range(wp_batch_np.shape[0])]
    if pool is not None:
        costs = pool.map(fn, wps)
    else:
        costs = [fn(w) for w in wps]
    return np.array(costs, dtype=np.float32)


# ── Frozen encoder helpers ──────────────────────────────────────────────


@torch.no_grad()
def encode_context(distilled_model, obs, cord):
    """Run the frozen image-based teacher encoder and return dec_out (B, feat_dim)."""
    import torchvision.transforms.functional as TF

    teacher = distilled_model.teacher
    B, N, _, H, W = obs.shape
    obs_flat = obs.view(B * N, 3, H, W)

    if teacher.do_rgb_normalize:
        obs_flat = (obs_flat - teacher.mean) / teacher.std
    if teacher.do_resize:
        obs_flat = TF.center_crop(obs_flat, teacher.crop)
        obs_flat = TF.resize(obs_flat, teacher.resize)

    obs_enc = teacher.obs_encoder(obs_flat)
    obs_enc = teacher.compress_obs_enc(obs_enc).view(B, N, -1)
    cord_enc = teacher.cord_embedding(cord).view(B, -1)
    cord_enc = teacher.compress_goal_enc(cord_enc).view(B, 1, -1)
    tokens = torch.cat([obs_enc, cord_enc], dim=1)
    tokens = teacher.positional_encoding(tokens)
    dec_out = teacher.sa_decoder(tokens).mean(dim=1)
    return dec_out


@torch.no_grad()
def encode_context_feat(distilled_model, obs_features, cord):
    """Run the frozen feature-based teacher encoder and return dec_out (B, feat_dim)."""
    teacher = distilled_model.teacher
    B = obs_features.shape[0]
    obs_enc = teacher.compress_obs_enc(obs_features)
    cord_enc = teacher.cord_embedding(cord).view(B, -1)
    cord_enc = teacher.compress_goal_enc(cord_enc).view(B, 1, -1)
    tokens = torch.cat([obs_enc, cord_enc], dim=1)
    tokens = teacher.positional_encoding(tokens)
    return teacher.sa_decoder(tokens).mean(dim=1)


# ── Training step ───────────────────────────────────────────────────────


def train_step(proposal, distilled_model, dec_out, optimizer,
               beta, num_samples, mpc_horizon, dt, n_skips, max_speed,
               baseline_state, baseline_ema, pool, device,
               noise_traj_len=12):
    """
    One gradient step of the noise-space variational tilting.

    Loss = KL(q_phi || N(0,I)) + beta * REINFORCE_estimate(E_{q_phi}[R])
    """
    B = dec_out.shape[0]
    K = num_samples
    len_traj_pred = distilled_model.len_traj_pred

    # 1. Sample noise from proposal
    z, log_q, kl = proposal(dec_out, num_samples=K)
    # z: (B, K, 24), log_q: (B, K), kl: (B,)

    # 2. Generate trajectories through frozen generator
    z_flat = z.view(B * K, noise_traj_len, 2)
    dec_rep = dec_out.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

    with torch.no_grad():
        deltas = distilled_model.forward_student(dec_rep, z_flat)
        wp = torch.cumsum(deltas, dim=1)  # (B*K, T, 2)

    # 3. Evaluate MPC cost (no gradients — runs on CPU)
    wp_np = wp.detach().cpu().numpy()
    costs_np = evaluate_mpc_batch(wp_np, mpc_horizon, dt, n_skips, max_speed,
                                  pool=pool)
    costs = torch.tensor(costs_np, dtype=torch.float32, device=device)
    costs = costs.view(B, K)

    # 4. REINFORCE with baseline
    baseline_val = baseline_state["value"]
    advantage = costs - baseline_val
    # Update running baseline
    baseline_state["value"] = (baseline_ema * baseline_val
                               + (1 - baseline_ema) * costs.mean().item())

    reinforce_loss = (advantage.detach() * log_q).mean()
    kl_loss = kl.mean()
    loss = kl_loss + beta * reinforce_loss

    # 5. Optimise
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(proposal.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "kl": kl_loss.item(),
        "reinforce": reinforce_loss.item(),
        "mean_cost": costs.mean().item(),
        "min_cost": costs.min().item(),
        "std_cost": costs.std().item(),
        "baseline": baseline_state["value"],
    }


# ── Validation ──────────────────────────────────────────────────────────


@torch.no_grad()
def validate(proposal, distilled_model, val_loader, beta,
             num_samples, mpc_horizon, dt, n_skips, max_speed,
             pool, device, max_batches=20,
             use_feat=False, noise_traj_len=12):
    """Compute mean tilted loss and cost on validation data."""
    proposal.eval()
    total_kl, total_cost, count = 0.0, 0.0, 0
    K = num_samples
    len_traj_pred = distilled_model.len_traj_pred

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        cord = batch["input_positions"].to(device)
        if use_feat:
            obs_features = batch["obs_features"].to(device)
            dec_out = encode_context_feat(distilled_model, obs_features, cord)
        else:
            obs = batch["video_frames"].to(device)
            dec_out = encode_context(distilled_model, obs, cord)
        B = dec_out.shape[0]

        z, log_q, kl = proposal(dec_out, num_samples=K)
        z_flat = z.view(B * K, noise_traj_len, 2)
        dec_rep = dec_out.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        deltas = distilled_model.forward_student(dec_rep, z_flat)
        wp = torch.cumsum(deltas, dim=1)

        wp_np = wp.cpu().numpy()
        costs_np = evaluate_mpc_batch(wp_np, mpc_horizon, dt, n_skips,
                                      max_speed, pool=pool)

        total_kl += kl.sum().item()
        total_cost += costs_np.sum()
        count += B

    proposal.train()
    return {
        "val/kl": total_kl / max(count, 1),
        "val/mean_cost": total_cost / max(count * K, 1),
    }


# ── Main ────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train exponential-tilted noise proposal")
    p.add_argument("--config", type=str, required=True,
                   help="Path to distillation config YAML (same as distill.py)")
    p.add_argument("--distill_ckpt", type=str, required=True,
                   help="Path to distilled model checkpoint (DistillationModule)")
    p.add_argument("--result_dir", type=str, default=None,
                   help="Output directory (default: <config.result_dir>/tilted)")
    # Tilting hyperparameters
    p.add_argument("--beta", type=float, default=0.1,
                   help="Tilt strength (higher = stronger preference for low MPC cost)")
    p.add_argument("--num_samples", type=int, default=8,
                   help="Noise samples per context during training")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override config batch size (default: use config)")
    p.add_argument("--baseline_ema", type=float, default=0.99,
                   help="EMA decay for REINFORCE baseline")
    # MPC parameters
    p.add_argument("--n_skips", type=int, default=5,
                   help="Sub-steps per waypoint for MPC horizon upsampling")
    p.add_argument("--dt", type=float, default=0.2,
                   help="MPC simulation timestep")
    p.add_argument("--max_speed", type=float, default=1.4,
                   help="Maximum linear velocity for unicycle model")
    # NoiseProposal architecture
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Hidden dimension of the proposal network")
    # Workers
    p.add_argument("--mpc_workers", type=int, default=4,
                   help="Number of parallel MPC solver workers (0 = sequential)")
    # Logging
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print metrics every N steps")
    p.add_argument("--val_interval", type=int, default=200,
                   help="Run validation every N steps")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="tilted-model")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load config ──
    cfg = OmegaConf.load(args.config)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    # ── Result directory ──
    result_dir = args.result_dir or os.path.join(
        cfg.project.result_dir, "tilted"
    )
    os.makedirs(result_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Determine model variant ──
    data_type = cfg.data.type
    model_type = cfg.model.type
    use_feat = data_type in ('carla_feat', 'urban_nav_feat_mixture')

    # ── Load distilled model (frozen) ──
    print(f"Loading distilled model from {args.distill_ckpt}")
    if use_feat:
        if model_type == 'flow_matching_feat_simple':
            from pl_modules.distillation_feat_simple_module import DistillationFeatSimpleModule
            distill_module = DistillationFeatSimpleModule.load_from_checkpoint(
                args.distill_ckpt, cfg=cfg
            )
        else:
            from pl_modules.distillation_feat_module import DistillationFeatModule
            distill_module = DistillationFeatModule.load_from_checkpoint(
                args.distill_ckpt, cfg=cfg
            )
    else:
        from pl_modules.distillation_module import DistillationModule
        distill_module = DistillationModule.load_from_checkpoint(
            args.distill_ckpt, cfg=cfg
        )
    distilled_model = distill_module.model
    distilled_model.eval()
    for p in distilled_model.parameters():
        p.requires_grad = False
    distilled_model.to(device)

    encoder_feat_dim = distilled_model.teacher.encoder_feat_dim
    len_traj_pred = distilled_model.len_traj_pred
    # Simple models don't zero-pad; UNet-based models pad noise to 12 timesteps
    if model_type == 'flow_matching_feat_simple':
        noise_traj_len = len_traj_pred
    else:
        noise_traj_len = 12
    noise_dim = noise_traj_len * 2
    print(f"  encoder_feat_dim={encoder_feat_dim}, len_traj_pred={len_traj_pred}, "
          f"noise_dim={noise_dim}")

    # ── MPC parameters ──
    mpc_horizon = len_traj_pred * args.n_skips

    # ── Noise proposal ──
    proposal = NoiseProposal(
        context_dim=encoder_feat_dim,
        noise_dim=noise_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(proposal.parameters(), lr=args.lr)
    print(f"  NoiseProposal params: {sum(p.numel() for p in proposal.parameters()):,}")

    # ── Data ──
    if data_type == "carla":
        from pl_modules.carla_datamodule import CarlaDataModule
        datamodule = CarlaDataModule(cfg)
    elif data_type == "citywalk":
        from pl_modules.citywalk_datamodule import CityWalkDataModule
        datamodule = CityWalkDataModule(cfg)
    elif data_type == "carla_feat":
        from pl_modules.carla_feat_datamodule import CarlaFeatDataModule
        datamodule = CarlaFeatDataModule(cfg)
    elif data_type == "urban_nav_feat_mixture":
        from pl_modules.urban_nav_feat_mixture_datamodule import UrbanNavFeatMixtureDataModule
        datamodule = UrbanNavFeatMixtureDataModule(cfg)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # ── MPC worker pool ──
    pool = None
    if args.mpc_workers > 0:
        pool = mp.Pool(
            args.mpc_workers,
            initializer=_init_worker,
            initargs=(mpc_horizon, args.dt, args.n_skips, args.max_speed),
        )
        print(f"  MPC worker pool: {args.mpc_workers} processes")

    # ── W&B ──
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config={
                    "beta": args.beta,
                    "num_samples": args.num_samples,
                    "lr": args.lr,
                    "hidden_dim": args.hidden_dim,
                    "n_skips": args.n_skips,
                    "dt": args.dt,
                    "max_speed": args.max_speed,
                    "mpc_horizon": mpc_horizon,
                    "batch_size": cfg.training.batch_size,
                },
                dir=result_dir,
            )
        except ImportError:
            print("wandb not installed, skipping")

    # ── Training loop ──
    baseline_state = {"value": 0.0}
    global_step = 0
    best_val_cost = float("inf")

    print(f"\nTraining tilted model: beta={args.beta}, K={args.num_samples}, "
          f"lr={args.lr}, mpc_horizon={mpc_horizon}")
    print(f"  {args.max_epochs} epochs, {len(train_loader)} batches/epoch\n")

    for epoch in range(args.max_epochs):
        t_epoch = time.time()
        epoch_costs = []

        for batch_idx, batch in enumerate(train_loader):
            cord = batch["input_positions"].to(device)

            # Encode context (frozen)
            if use_feat:
                obs_features = batch["obs_features"].to(device)
                dec_out = encode_context_feat(distilled_model, obs_features, cord)
            else:
                obs = batch["video_frames"].to(device)
                dec_out = encode_context(distilled_model, obs, cord)

            metrics = train_step(
                proposal, distilled_model, dec_out, optimizer,
                beta=args.beta,
                num_samples=args.num_samples,
                mpc_horizon=mpc_horizon,
                dt=args.dt,
                n_skips=args.n_skips,
                max_speed=args.max_speed,
                baseline_state=baseline_state,
                baseline_ema=args.baseline_ema,
                pool=pool,
                device=device,
                noise_traj_len=noise_traj_len,
            )
            epoch_costs.append(metrics["mean_cost"])
            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    f"  [{epoch+1}/{args.max_epochs}] step {global_step:5d} | "
                    f"loss={metrics['loss']:.4f}  kl={metrics['kl']:.4f}  "
                    f"cost={metrics['mean_cost']:.2f} (min={metrics['min_cost']:.2f})  "
                    f"baseline={metrics['baseline']:.2f}"
                )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=global_step,
                )

            # Validation
            if global_step % args.val_interval == 0:
                val_metrics = validate(
                    proposal, distilled_model, val_loader, args.beta,
                    num_samples=args.num_samples,
                    mpc_horizon=mpc_horizon, dt=args.dt,
                    n_skips=args.n_skips, max_speed=args.max_speed,
                    pool=pool, device=device,
                    use_feat=use_feat, noise_traj_len=noise_traj_len,
                )
                print(
                    f"  ── val: kl={val_metrics['val/kl']:.4f}  "
                    f"cost={val_metrics['val/mean_cost']:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log(val_metrics, step=global_step)

                if val_metrics["val/mean_cost"] < best_val_cost:
                    best_val_cost = val_metrics["val/mean_cost"]
                    torch.save(
                        {"proposal": proposal.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "step": global_step,
                         "beta": args.beta,
                         "best_val_cost": best_val_cost},
                        os.path.join(ckpt_dir, "best.pt"),
                    )

        # End-of-epoch checkpoint
        torch.save(
            {"proposal": proposal.state_dict(),
             "optimizer": optimizer.state_dict(),
             "step": global_step,
             "epoch": epoch,
             "beta": args.beta},
            os.path.join(ckpt_dir, "last.pt"),
        )
        elapsed = time.time() - t_epoch
        mean_cost = np.mean(epoch_costs) if epoch_costs else float("nan")
        print(
            f"Epoch {epoch+1}/{args.max_epochs} done in {elapsed:.1f}s  "
            f"mean_cost={mean_cost:.2f}  best_val_cost={best_val_cost:.2f}\n"
        )

    # ── Cleanup ──
    if pool is not None:
        pool.close()
        pool.join()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"Done. Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
