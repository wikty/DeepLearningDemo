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
        self._data_dir = None
        self._params_file = params_file
        self._log_file = log_file
        self._train_name = train_name
        self._val_name = val_name
        self._test_name = test_name
        self._train_factor = train_factor
        self._val_factor = val_factor
        self._test_factor = test_factor
        self.set_data_dir(data_dir)

    def set_data_dir(self, data_dir):
        msg = "Data directory not exists"
        assert os.path.isdir(data_dir), msg
        msg = "Dataset parameters file not exists"
        assert os.path.isfile(os.path.join(data_dir, 
                                           self._params_file)), msg
        self._data_dir = data_dir

    def params_filename(self):
        return self._params_file

    def log_filename(self):
        return self._log_file

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
