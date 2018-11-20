import os


class DatasetCfg(object):

    def __init__(self, data_dir,
                 params_file='datasets.json',
                 log_file='datasets.log',
                 train_name='train',
                 val_name='val',
                 test_name='test',
                 train_factor=0.7,
                 val_factor=0.15,
                 test_factor=0.15):
        self._data_dir = data_dir
        self._params_file = params_file
        self._log_file = log_file
        self._train_name = train_name
        self._val_name = val_name
        self._test_name = test_name
        self._train_factor = train_factor
        self._val_factor = val_factor
        self._test_factor = test_factor

    def data_dir(self):
        return self._data_dir

    def params_file(self):
        return os.path.join(self._data_dir, self._params_file)

    def log_file(self):
        return os.path.join(self._data_dir, self._log_file)

    def train_name(self):
        return self._train_name

    def val_name(self):
        return self._val_name

    def test_name(self):
        return self._test_name

    def train_factor(self):
        return self._train_factor

    def val_factor(self):
        return self._val_factor

    def test_factor(self):
        return self._test_factor
