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
        raise NotImplementedError

    def read(self):
        """Return a sample, or None if there is no sample."""
        raise NotImplementedError


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
            if self.shuffle:
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


class BaseLoader(object):

    def __init__(self, sample_transform=None, batch_transform=None):
        self.sample_transform = sample_transform
        self.batch_transform = batch_transform

    def create_loader(self, *args, **kwargs):
        """Fix the parameters for loader"""
        def loader():
            return self.load(*args, **kwargs)
        return loader

    def load(self, handler, name, size, 
             batch_size=None, shuffle=False, 
             sample_transform=None, batch_transform=None):
        """Return dataset or batch iterator.

        Args:
            handler (handler): the handler to read dataset.
            name (str): the name of dataset.
            size (int): the size of dataset.
            batch_size (int or None): the size of batch, or None if want a
                dataset iterator.
            shuffle (bool): the flag to shuffle dataset
        """
        if sample_transform is None:
            sample_transform = self.sample_transform
        if batch_transform is None:
            batch_transform = self.batch_transform
        dataset = DatasetIterator(name, size, handler, 
                                   shuffle=shuffle,
                                   transform=sample_transform)
        if batch_size is None:
            return dataset
        batches = BatchIterator(dataset, 
                                 batch_size=batch_size, 
                                 transform=batch_transform)
        return batches