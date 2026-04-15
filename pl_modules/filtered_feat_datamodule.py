import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.filtered_feat_dataset import FilteredFeatDataset
from data.carla_dataset import CarlaSampler


class FilteredFeatDataModule(pl.LightningDataModule):
    """DataModule backed by pre-built filtered LUT files.

    Expects the config to contain::

        data:
          type: filtered_feat
          lut_train: /path/to/lut_train.npz
          lut_val:   /path/to/lut_val.npz
          lut_test:  /path/to/lut_test.npz   # optional
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = FilteredFeatDataset(
                lut_path=self.cfg.data.lut_train, cfg=self.cfg, mode='train',
            )
            self.val_dataset = FilteredFeatDataset(
                lut_path=self.cfg.data.lut_val, cfg=self.cfg, mode='val',
            )
        if stage == 'test' or stage is None:
            lut_test = getattr(self.cfg.data, 'lut_test', None)
            if lut_test is None:
                raise ValueError(
                    "data.lut_test must be set for test stage"
                )
            self.test_dataset = FilteredFeatDataset(
                lut_path=lut_test, cfg=self.cfg, mode='test',
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=CarlaSampler(self.train_dataset),
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
