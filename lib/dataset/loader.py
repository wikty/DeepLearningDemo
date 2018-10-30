import random


class DatasetHandler(object):
    """The handler to read sample in the dataset."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __enter__(self):
        """init dataset resources"""
        return self

    def __exit__(self, type, value, trace):
        """clear dataset resources"""
        pass

    def read(self):
        """Return a sample, or None if there is no sample."""
        pass


class DatasetIterator(object):

    def __init__(self, name, size, handler, 
                 shuffle=False, transform=None):
        self.name = name
        self.size = size
        self.handler = handler
        self.shuffle = shuffle
        self.transform = transform
        if self.transform is None:
            self.transform = (lambda s: s)

    def __iter__(self):
        """Return a generator that returns a sample in each iteration."""
        with self.handler as handler:
            if shuffle:
                # load all samples into memory
                samples = []
                while True:
                    sample = handler.read()
                    if sample is None:
                        break
                    sample = self.transform(sample)
                    samples.append(sample)
                random.shuffle(samples)
                for sample in samples:
                    yield sample
            else:
                # lazy-loading mode
                while True:
                    sample = handler.read()
                    if sample is None:
                        break
                    sample = self.transform(sample)
                    yield sample


class BatchIterator(object):

    def __init__(self, dataset, batch_size=1, transform=None):
        """
        Args:
            dataset (DatasetIterator): the instance of DatasetIterator.
        """
        self.dataset = dataset
        self.size = batch_size
        self.transform = transform
        if self.transform is None:
            self.transform = (lambda s: s)

    @property
    def dataset_name(self):
        """The name of dataset."""
        return self.dataset.name

    @property
    def dataset_size(self):
        """The size of dataset."""
        return self.dataset.size
    
    @property
    def batch_size(self):
        """The size of batch."""
        return self.size

    def __iter__(self):
        """Return a iterator that returns a batch in each iteration."""
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.size:
                yield self.transform(batch)
                batch = []
        if batch:
            # the last batch may be less then batch size.
            yield self.transform(batch)


class Loader(object):

    def __init__(self, handler, 
                 sample_transform=None, batch_sample=None):
        pass

    def load(self, name, size, batch_size=None, shuffle=False):
        dataset = DatasetGenerator(name, size, self.handler, 
                                   shuffle=shuffle,
                                   transform=self.sample_transform)
        if batch_size is None:
            return dataset
        batches = BatchGenerator(dataset, 
                                 batch_size=batch_size, 
                                 transform=self.batch_transform)
        return generator