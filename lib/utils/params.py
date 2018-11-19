import copy
import json


class Params(object):
    """Load/dump parameters from/to json file."""

    def __init__(self, json_path=None, encoding='utf8', data={}):
        if json_path:
            self.load(json_path, encoding)
        if data and isinstance(data, dict):
            self.__dict__.update(data)

    def __getattr__(self, key):
        """Raise exception when the key not exists in the instance."""
        raise AttributeError("{} not exists in params.".format(key))

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        for key, value in self.__dict__.items():
            yield key, value

    @property
    def dict(self):
        """A copy with dict-like interface to Params instance."""
        return copy.deepcopy(self.__dict__)

    def load(self, json_path, encoding='utf8'):
        """Load and update parameters from json file."""
        with open(json_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            self.__dict__.update(data)

    def dump(self, json_path, encoding='utf8', indent=4):
        """Dump parameters to json file."""
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(self.__dict__, f, 
                      ensure_ascii=False, indent=indent)

    def items(self):
        """A interface like dict.items()."""
        return self.__dict__.items()

    def get(self, name, default=None):
        """Get a attribute"""
        return self.__dict__.get(name, default)

    def set(self, name, value):
        """Set a attribute for params."""
        setattr(self, name, value)

    def update(self, params):
        """Update parameters from another Params instance."""
        assert isinstance(params, Params)
        self.__dict__.update(params.dict)

    def copy(self):
        params = Params()
        params.update(self)
        return params

    def check_all(self, attrs=[]):
        """Check if all of attrs in the Params."""
        flag = True
        for name in attrs:
            if name not in self.__dict__:
                flag = False
                break
        return flag

    def check_any(self, attrs=[]):
        """Check if any of attrs in the Params."""
        flag = False
        for name in attrs:
            if name in self.__dict__:
                flag = True
                break
        return flag
