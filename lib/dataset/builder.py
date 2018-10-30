import random
import argparse


def get_parser(data_dir, train_factor, val_factor, test_factor,
               train_name, val_name, test_name):
    def tofloat(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in [0.0, 1.0]" % x)
        return x

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-dir', 
        default=data_dir,
        help="Directory for the dataset to save (default: %(default)s)")
    parser.add_argument('--data-factor',
        default=1.0,
        help="The factor of dataset to be builded (default: %(default)s)",
        type=tofloat)
    parser.add_argument('--train-factor', 
        default=train_factor,
        help="The factor of train dataset (default: %(default)s)", 
        type=tofloat)
    parser.add_argument('--val-factor', 
        default=val_factor,
        help="The factor of validation dataset (default: %(default)s)", 
        type=tofloat)
    parser.add_argument('--test-factor', 
        default=test_factor,
        help="The factor of test dataset (default: %(default)s)", 
        type=tofloat)
    parser.add_argument('--train-name',
        default=train_name,
        help="The name of train dataset (default: %(default)s)")
    parser.add_argument('--val-name',
        default=val_name,
        help="The name of validation dataset (default: %(default)s)")
    parser.add_argument('--test-name',
        default=test_name,
        help="The name of test dataset (default: %(default)s)")

    return parser


class Sample(object):
    """Sample is a wrapper of dict."""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        """Raise exception when the key not exists."""
        raise AttributeError("{} not exists in sample.".format(key))


class Dataset(object):
    """Dataset is a manager of samples."""

    def __init__(self, name, samples=[]):
        self._name = name
        self._samples = []
        self.extend(samples)

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        return self.get()

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return len(self._samples)

    def add(self, sample):
        assert isinstance(sample, Sample)
        self._samples.append(sample)

    def extend(self, samples=[]):
        for sample in samples:
            self.add(sample)

    def get(self, shuffle=False, key=None, **key_kwargs):
        """Return a iterator for dataset.
        
        Args:
            shuffle (bool): shuffle the dataset
            key (function): a filter function for dataset
            key_kwargs (dict): arguments for the key filter function
        """
        foo = list(range(len(self._samples)))
        if shuffle:
            random.shuffle(foo)
        for i in foo:
            sample = self._samples[i]
            if key is not None and (not key(sample, **key_kwargs)):
                continue
            yield sample


class BaseBuilder(object):
    """The builder for dataset."""

    def __init__(self, 
                 data_factor=1.0, train_factor=0.7, 
                 val_factor=0.15, test_factor=0.15, 
                 train_name='train', val_name='val', 
                 test_name='test', logger=None, 
                 *args, **kwargs):
        """
        Args:
            data_factor: the factor of the data to be builded.
            train_factor: the factor of train dataset.
            val_factor: the factor of validation dataset.
            test_factor: the factor of test dataset.
            train_name: the name of train dataset.
            val_name: the name of validation dataset.
            test_name: the name of test dataset.
        """
        self.data_factor = data_factor
        self.train_factor = train_factor
        self.val_factor = val_factor
        self.test_factor = test_factor
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.logger = logger
        self.samples = []

    def build(self, shuffle=True):
        total = int(len(self.samples)*self.data_factor)
        samples = self.samples[:total]  # only build part of data
        if shuffle:
            random.shuffle(samples)
        l = len(samples)
        i = int(l*self.train_factor)
        j = int(l*(self.train_factor+self.val_factor))
        return (
            Dataset(self.train_name, samples[:i]),
            Dataset(self.val_name, samples[i:j]),
            Dataset(self.test_name, samples[j:])
        )

    def add_sample(self, **kwargs):
        """Add a sample into builder."""
        self.samples.append(Sample(**kwargs))

    def load(self, *args, **kwargs):
        """Load data from disk and add them to builder.

        Note: You should implement this method for your project.
        """
        pass

    def dump(self, *args, **kwargs):
        """Dump data to disk and split them to train/val/test data.
    
        Note: You should implement this method for your project.
        """
        pass