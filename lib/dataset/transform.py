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