import torch
from torch.utils.data import Sampler


class WeightedMixtureSampler(Sampler):
    """Samples from a ConcatDataset according to per-dataset mixture weights.

    At each draw, a dataset is chosen proportionally to its weight, then a
    sample is drawn uniformly at random from that dataset's index range within
    the ConcatDataset.

    Args:
        dataset_lengths: Length of each constituent dataset (in ConcatDataset order).
        weights: Unnormalized sampling weight for each dataset.
        total_samples: Number of draws per epoch.  Defaults to the sum of all
            dataset lengths so that one "epoch" has the same total draw count
            as seeing every sample once (though the actual per-dataset coverage
            is governed by the weights, not the sizes).
    """

    def __init__(self, dataset_lengths, weights, total_samples=None):
        self.weights = torch.tensor(weights, dtype=torch.float64)
        self.weights /= self.weights.sum()

        # ConcatDataset-style offsets
        self.offsets = []
        self.lengths = []
        offset = 0
        for length in dataset_lengths:
            self.offsets.append(offset)
            self.lengths.append(length)
            offset += length

        self.total_samples = total_samples or offset

    def __iter__(self):
        # Each call to __iter__ produces a fresh random permutation so that
        # successive epochs see different orderings.
        ds_choices = torch.multinomial(
            self.weights, self.total_samples, replacement=True
        )
        for ds_idx in ds_choices:
            ds_idx = ds_idx.item()
            local_idx = torch.randint(0, self.lengths[ds_idx], (1,)).item()
            yield self.offsets[ds_idx] + local_idx

    def __len__(self):
        return self.total_samples
