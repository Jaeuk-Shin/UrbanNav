import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import OmegaConf

from data.carla_feat_dataset import CarlaFeatDataset
from data.mixture_sampler import WeightedMixtureSampler


class UrbanNavFeatMixtureDataModule(pl.LightningDataModule):
    """DataModule that mixes multiple feature-based navigation datasets with
    configurable per-dataset sampling weights.

    Each entry in ``cfg.data.mixture`` specifies a dataset root directory
    (expected to contain ``pose/`` and ``dino/`` subdirectories by default),
    a sampling weight, and optional per-dataset overrides for episode counts
    and subdirectory names.

    Example config entry::

        mixture:
          - root: /data/carla_town01
            weight: 0.5
          - root: /data/real_world
            weight: 0.5
            pose_subdir: camera-to-world   # override default 'pose'
            feature_subdir: features        # override default 'dino'
            num_train: 500
            num_val: 25

    Training batches are drawn proportionally to the weights via
    ``WeightedMixtureSampler``.  Validation and test batches iterate over the
    full concatenation of all datasets without weighting.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _make_sub_cfg(self, entry):
        """Build a per-dataset OmegaConf by overriding data paths/counts."""
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        data = cfg_dict['data']

        root = entry['root']
        data['root_dir'] = root
        data['pose_dir'] = entry.get('pose_subdir', 'pose')
        data['feature_dir'] = entry.get(
            'feature_dir',
            f"{root}/{entry.get('feature_subdir', 'dino')}",
        )
        data['rgb_dir'] = entry.get(
            'rgb_dir',
            f"{root}/{entry.get('rgb_subdir', 'rgb')}",
        )

        # Per-dataset camera intrinsics (optional)
        if 'camera' in entry:
            data['camera'] = entry['camera']

        # Per-dataset keep-list (produced by filter_episodes.py)
        if 'keep_list' in entry:
            data['keep_list'] = entry['keep_list']

        # Per-dataset episode counts (fall back to top-level defaults)
        for key in ('num_train', 'num_val', 'num_test'):
            if key in entry:
                data[key] = entry[key]
            else:
                data.setdefault(key, 0)

        return OmegaConf.create(cfg_dict)

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        mixture = OmegaConf.to_container(self.cfg.data.mixture, resolve=True)
        weights = [entry['weight'] for entry in mixture]

        if stage == 'fit' or stage is None:
            train_datasets = []
            val_datasets = []
            for entry in mixture:
                sub_cfg = self._make_sub_cfg(entry)
                train_datasets.append(CarlaFeatDataset(sub_cfg, mode='train'))
                val_datasets.append(CarlaFeatDataset(sub_cfg, mode='val'))

            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            self.train_weights = weights

        if stage == 'test' or stage is None:
            test_datasets = []
            for entry in mixture:
                sub_cfg = self._make_sub_cfg(entry)
                test_datasets.append(CarlaFeatDataset(sub_cfg, mode='test'))
            self.test_dataset = ConcatDataset(test_datasets)

    def train_dataloader(self):
        sampler = WeightedMixtureSampler(
            dataset_lengths=[len(ds) for ds in self.train_dataset.datasets],
            weights=self.train_weights,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
