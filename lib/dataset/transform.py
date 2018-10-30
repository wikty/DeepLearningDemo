import torch
import numpy as np


class IdentityTransform(object):

    def __call__(self, sample):
        return sample


class ComposeTransform(object):

    def __init__(self, transforms=[]):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample

    def copy(self):
        """Return a copy of transform."""
        return ComposeTransform(self.transforms)

    def prepend(self, transform):
        """Insert `transform` on the first position."""
        self.transform.insert(0, transform)

    def append(self, transform):
        """Insert `transform` on the last position."""
        self.transforms.append(transform)

    def insert(self, i, transform):
        """Insert `transform` on the `i` position."""
        self.transforms.insert(i, transform)


class SplitTransform(object):
    """Split batch into input and output batch."""

    def __call__(self, batch):
        batch_input, batch_output = [], []
        for sample in batch:
            batch_input.append(sample[0])
            batch_output.append(sample[1])
        return batch_input, batch_output


class ToArrayTransform(object):
    """Convert list to numpy array."""

    def __init__(self, dtype=None):
        """
        Args:
            dtype (np.dtype): numpy dtype.
        """
        self.dtype = dtype

    def __call__(self, batch):
        batch_input, batch_target = batch
        return (np.array(batch_input, dtype=self.dtype), 
            np.array(batch_target, dtype=self.dtype))


class ToTensorTransform(object):

    def __init__(self, dtype=None, to_cuda=None):
        """
        Args:
            dtype (torch.dtype): torch dtype.
            to_cude (bool|device): flag to enable cude or the destination
                GPU device instance.
        """
        self.dtype = dtype
        self.to_cuda = to_cuda

    def __call__(self, batch):
        batch_input, batch_target = batch
        if not self.to_cuda:
            return (torch.tensor(batch_input, dtype=self.dtype), 
                torch.tensor(batch_target, dtype=self.dtype))
        elif isinstance(self.to_cuda, bool):
            return (torch.tensor(batch_input, dtype=self.dtype).cuda(),
                torch.tensor(batch_target, dtype=self.dtype).cuda())
        else:
            return (
                torch.tensor(batch_input, 
                             dtype=self.dtype).cuda(self.to_cuda),
                torch.tensor(batch_target, 
                             dtype=self.dtype).cuda(self.to_cuda)
            )
